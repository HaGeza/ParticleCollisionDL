import numpy as np
import torch
from torch import Tensor
from sklearn.neighbors import KDTree

from src.Pairing import GreedyStrategy
from src.Util import CoordinateSystemEnum
from .IPairingStrategy import IPairingStrategy


class RepeatedKDTreeStrategy(IPairingStrategy):
    """
    Pairing strategy that repeatedly build KD-trees to pair points, until a stopping criterion is met.
    The remaining points are paired using `GreedyStrategy`. Assuming that there are at most `sqrt(n log n)`
    points remaining for the greedy-phase, time complexity is O(n log n).
    """

    def __init__(self, k: int = 20):
        """
        :param CoordinateSystemEnum coordinate_system: The coordinate system to use.
        :param int k: Number of nearest neighbors to query. The pairs are formed by
            randomly selecting one of the `k` nearest neighbors.
        """

        self.k = k
        self.fallback_strategy = GreedyStrategy()

    def create_pairs_in_batch(self, args: tuple[Tensor, Tensor, int, int]) -> Tensor:
        """
        Create pairs of generated and ground-truth hits within a batch.

        :param tuple args: Tuple of predicted and ground-truth hit sets and offsets, where
            predicted hit set has shape `[num_hits_batch_i, hit_dim]`, ground truth hit set
            has shape `[num_hits_next_batch_i, hit_dim]`, and offsets are the indices of the
            first hit in the batch.
        :return Tensor: Pairing tensor. Shape `[min(num_hits_batch_i, num_hits_next_batch_i), 2]`
        """

        pred, gt, pred_offset, gt_offset = args

        kd_iter = 0
        num_pairs_req = min(pred.size(0), gt.size(0))
        pairs = torch.zeros((0, 2), device=pred.device, dtype=torch.long)

        unused_pred = torch.ones(pred.size(0), device=pred.device, dtype=torch.bool)
        unused_gt = torch.ones(gt.size(0), device=pred.device, dtype=torch.bool)

        indices_pred = torch.arange(pred.size(0), device=pred.device)
        indices_gt = torch.arange(gt.size(0), device=pred.device)

        while num_pairs_req > self.k:
            kd_tree = KDTree(gt[unused_gt].cpu().numpy())
            _, selected_gt = kd_tree.query(pred[unused_pred].cpu().numpy(), k=self.k)
            neighbor_inds = np.random.randint(0, selected_gt.shape[1], size=selected_gt.shape[0])
            selected_gt = selected_gt[np.arange(selected_gt.shape[0]), neighbor_inds]

            selected_gt, selected_pred = np.unique(selected_gt.flatten(), return_index=True)

            selected_pred = indices_pred[unused_pred][selected_pred]
            selected_gt = indices_gt[unused_gt][selected_gt]
            new_pairs = torch.stack([selected_pred + pred_offset, selected_gt + gt_offset], dim=1)
            pairs = torch.cat((pairs, new_pairs), dim=0)

            unused_pred[selected_pred] = False
            unused_gt[selected_gt] = False

            num_pairs_req -= new_pairs.size(0)
            kd_iter += 1

            num_added = new_pairs.size(0)
            # num_added elements were added, and it took O(num_req * log(num_req)) time,
            # so adding 1 element takes O(num_req * log(num_req) / num_added) time with KD-tree,
            # and O(num_req^2 / num_req) = O(num_req) time with GreedyStrategy. If greedy is
            # estimated to be faster, switch to it.
            if num_pairs_req * np.log(num_pairs_req) / num_added > num_pairs_req:
                break

        if num_pairs_req > 0:
            new_pairs = self.fallback_strategy.create_pairs_in_batch(
                (pred[unused_pred], gt[unused_gt], pred_offset, gt_offset)
            )
            pairs = torch.cat((pairs, new_pairs), dim=0)

        return pairs
