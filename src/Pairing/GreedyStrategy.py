import multiprocessing
import torch
from torch import Tensor

from src.Util import CoordinateSystemEnum
from .IPairingStrategy import IPairingStrategy


class GreedyStrategy(IPairingStrategy):
    """
    Pairing strategy that pairs up hits greedily based on distance. Time complexity: O(n^2)
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

        pred, gt, pred_ind, gt_ind = args

        num_pairs = min(pred.size(0), gt.size(0))
        pairs = torch.zeros((num_pairs, 2), device=pred.device, dtype=torch.long)
        unused = torch.ones(gt.size(0), device=pred.device, dtype=torch.bool)
        indices = torch.arange(gt.size(0), device=pred.device)

        for i in range(num_pairs):
            min_ind = torch.argmin(torch.sum((pred[i] - gt[unused]) ** 2, dim=1), dim=0)
            min_ind = indices[unused][min_ind]
            pairs[i] = torch.tensor([i + pred_ind, min_ind + gt_ind], device=pred.device)
            unused[min_ind] = False

        return pairs
