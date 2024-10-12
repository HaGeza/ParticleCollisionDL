import multiprocessing
import numpy as np
import torch
from torch import Tensor

from src.Util import CoordinateSystemEnum
from src.Util import convert_to_cartesian, cartesian_squared_euclidean


class IPairingStrategy:
    """
    Interface for the pairing strategy. A pairing strategy is used to pair up
    generated and ground-truth hits to calculate the reconstruction loss.
    """

    MAX_PAIR_LOSS = 7000.0

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

    def _create_pairs(self, pred: Tensor, gt: Tensor, pred_ind: Tensor, gt_ind: Tensor) -> Tensor:
        """
        Create pairs of generated and ground-truth hits.

        :param Tensor pred: Predicted hit set. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit set. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :return Tensor: Pairing tensor. Shape `[sum(min(num_hits_batch_i, num_hits_next_batch_i)), 2]`
        """

        # Create lists of batch predictions and ground truths
        num_batches = max(gt_ind.max() if len(gt_ind) > 0 else 0, pred_ind.max() if len(pred_ind) else 0) + 1
        pred_list = [pred[pred_ind == i] for i in range(num_batches)]
        gt_list = [gt[gt_ind == i] for i in range(num_batches)]
        # Create list of offsets using cumulative sum of list lengths
        pred_offsets = list(np.cumsum([0] + [len(pred) for pred in pred_list[:-1]]))
        gt_offsets = list(np.cumsum([0] + [len(gt) for gt in gt_list[:-1]]))

        with multiprocessing.Pool() as pool:
            pairs_list = pool.map(self.create_pairs_in_batch, zip(pred_list, gt_list, pred_offsets, gt_offsets))
        # pairs_list = [
        #     self.create_pairs_in_batch((pred, gt, pred_offset, gt_offset))
        #     for pred, gt, pred_offset, gt_offset in zip(pred_list, gt_list, pred_offsets, gt_offsets)
        # ]

        pairs = torch.cat(pairs_list, dim=0)

        return pairs

    def calculate_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_ind: Tensor,
        gt_ind: Tensor,
        coordinate_system: CoordinateSystemEnum,
        reduction: str = "mean",
    ) -> Tensor:
        """
        Calculate the reconstruction loss.

        :param Tensor pred: Predicted hit set. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit set. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param str reduction: Reduction method for the loss.
        :return Tensor: Reconstruction loss.
        """

        pred_cart = convert_to_cartesian(pred, coordinate_system)
        gt_cart = convert_to_cartesian(gt, coordinate_system)

        pairs = self._create_pairs(pred_cart, gt_cart, pred_ind, gt_ind)
        diffs = cartesian_squared_euclidean(pred_cart[pairs[:, 0]], gt_cart[pairs[:, 1]])
        if reduction == "mean":
            return diffs.mean() if len(diffs) > 0 else torch.tensor(0.0, device=pred.device)
        elif reduction == "sum":
            return diffs.sum()
        return diffs