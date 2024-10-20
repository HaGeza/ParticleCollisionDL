import numpy as np
import torch
from torch import Tensor
from torch.multiprocessing import Pool

from src.Util import CoordinateSystemEnum
from src.Util import convert_to_cartesian


class IPairingStrategy:
    """
    Interface for the pairing strategy. A pairing strategy is used to pair up
    generated and ground-truth hits to calculate the reconstruction loss.
    """

    def create_pairs_in_batch(self, args: tuple[Tensor, Tensor, int, int]) -> Tensor:
        """
        Create pairs of generated and ground-truth hits within a batch.

        :param tuple args: Tuple of predicted and ground-truth hit sets and offsets, where
            predicted hit set has shape `[num_hits_batch_i, hit_dim]`, ground truth hit set
            has shape `[num_hits_next_batch_i, hit_dim]`, and offsets are the indices of the
            first hit in the batch.
        :return Tensor: Pairing tensor. Shape `[min(num_hits_batch_i, num_hits_next_batch_i), 2]`
        """

        raise NotImplementedError

    @staticmethod
    def _get_max_ind(gt_ind: Tensor, pred_ind: Tensor) -> int:
        """
        Get the maximum index from the ground truth and predicted indices.

        :param Tensor gt_ind: Ground truth indices.
        :param Tensor pred_ind: Predicted indices.
        :return int: The maximum index.
        """
        num_gt = gt_ind.max() if len(gt_ind) > 0 else 0
        num_pred = pred_ind.max() if len(pred_ind) else 0
        return (max(num_gt, num_pred) + 1).item()

    def create_pairs(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_batch_ind: Tensor,
        gt_batch_ind: Tensor,
        pred_part_ind: Tensor = None,
        gt_part_ind: Tensor = None,
        paired: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Create pairs of generated and ground-truth hits.

        :param Tensor pred: Predicted hit set. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit set. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_batch_ind: Predicted hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_batch_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor pred_part_ind: Predicted hit part index tensor. `None` or tensor with shape `[num_hits]`.
        :param Tensor gt_part_ind: Ground truth hit part index tensor. `None` or tensor with shape `[num_hits_next]`.
        :param bool paired: Whether the pairing is precomputed
        :return tuple[Tensor, Tensor]: A tuple containing two Tensors:
            1. Pairing tensor. Shape `[sum(min(num_hits_batch_i, num_hits_next_batch_i)), 2]`
            2. Number of pairs per batch or per batch-part combination. Shape `[num_batches]`
                or `[num_batches * num_parts]`.
        """

        if paired:
            if pred_part_ind is None or gt_part_ind is None:
                _, pair_counts = torch.unique(pred_batch_ind, return_counts=True)
            else:
                num_parts = IPairingStrategy._get_max_ind(gt_part_ind, pred_part_ind)
                _, pair_counts = torch.unique(pred_batch_ind * num_parts + pred_part_ind, return_counts=True)

            pairs = torch.arange(pair_counts.sum(), device=pred.device, dtype=torch.long).unsqueeze(1).repeat(1, 2)
            return pairs, pair_counts

        original_device = pred.device.type
        if original_device == "mps":
            pred = pred.cpu()
            gt = gt.cpu()
            pred_batch_ind = pred_batch_ind.cpu()
            gt_batch_ind = gt_batch_ind.cpu()

        # Create lists of batch predictions and ground truths
        num_batches = IPairingStrategy._get_max_ind(gt_batch_ind, pred_batch_ind)

        if pred_part_ind is None or gt_part_ind is None:
            sorted_pred_inds = torch.argsort(pred_batch_ind)
            sorted_gt_inds = torch.argsort(gt_batch_ind)

            pred_list = [pred[pred_batch_ind == i] for i in range(num_batches)]
            gt_list = [gt[gt_batch_ind == i] for i in range(num_batches)]
        else:
            num_parts = IPairingStrategy._get_max_ind(gt_part_ind, pred_part_ind)

            sorted_pred_inds = torch.argsort(pred_batch_ind * num_parts + pred_part_ind)
            sorted_gt_inds = torch.argsort(gt_batch_ind * num_parts + gt_part_ind)

            pred_list = [
                pred[(pred_batch_ind == i) & (pred_part_ind == j)]
                for i in range(num_batches)
                for j in range(num_parts)
            ]
            gt_list = [
                gt[(gt_batch_ind == i) & (gt_part_ind == j)] for i in range(num_batches) for j in range(num_parts)
            ]

        # Create list of offsets using cumulative sum of list lengths
        pred_offsets = list(np.cumsum([0] + [len(pred) for pred in pred_list[:-1]]))
        gt_offsets = list(np.cumsum([0] + [len(gt) for gt in gt_list[:-1]]))

        with Pool() as pool:
            pairs_list = pool.map(self.create_pairs_in_batch, zip(pred_list, gt_list, pred_offsets, gt_offsets))

        pairs = torch.cat(pairs_list, dim=0)

        pairs[sorted_pred_inds, 0] = pairs[:, 0]
        pairs[sorted_gt_inds, 1] = pairs[:, 1]

        if original_device == "mps":
            pred = pred.to("mps")
            gt = gt.to("mps")
            pred_batch_ind = pred_batch_ind.to("mps")
            gt_batch_ind = gt_batch_ind.to("mps")
            pairs = pairs.to("mps")

        return pairs, torch.tensor([len(p) for p in pairs_list], device=pred.device)

    def calculate_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_batch_ind: Tensor,
        gt_batch_ind: Tensor,
        pred_part_ind: Tensor = None,
        gt_part_ind: Tensor = None,
        coordinate_system: CoordinateSystemEnum = CoordinateSystemEnum.CARTESIAN,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Calculate the reconstruction loss.

        :param Tensor pred: Predicted hit set. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit set. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_batch_ind: Predicted hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_batch_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor pred_part_ind: Predicted hit part index tensor. `None` or tensor with shape `[num_hits]`.
        :param Tensor gt_part_ind: Ground truth hit part index tensor. `None` or tensor with shape `[num_hits_next]`.
        :param str reduction: Reduction method for the loss.
        :return Tensor: Reconstruction loss.
        """

        pred_cart = convert_to_cartesian(pred, coordinate_system)
        gt_cart = convert_to_cartesian(gt, coordinate_system)

        pairs, _ = self.create_pairs(pred_cart, gt_cart, pred_batch_ind, gt_batch_ind, pred_part_ind, gt_part_ind)
        diffs = torch.sum((pred_cart[pairs[:, 0]] - gt_cart[pairs[:, 1]]) ** 2, dim=1)
        if reduction == "mean":
            return diffs.mean() if len(diffs) > 0 else torch.tensor(0.0, device=pred.device)
        if reduction == "sum":
            return diffs.sum()
        return diffs
