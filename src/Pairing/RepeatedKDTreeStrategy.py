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

    def __init__(self, max_kd_tree_iter: int = 25, k: int = 10):
        """
        :param CoordinateSystemEnum coordinate_system: The coordinate system to use.
        :param int max_kd_tree_iter: Maximum number of KD-tree iterations.
        :param int k: Number of nearest neighbors to query. The pairs are formed by
            randomly selecting one of the `k` nearest neighbors.
        """

        self.max_kd_tree_iter = max_kd_tree_iter
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

        min_kd_tree_points = np.sqrt(num_pairs_req * np.log(num_pairs_req)) if num_pairs_req > 0 else 0

        while num_pairs_req > min_kd_tree_points and kd_iter < self.max_kd_tree_iter:
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

        if num_pairs_req > 0:
            new_pairs = self.fallback_strategy.create_pairs_in_batch(
                (pred[unused_pred], gt[unused_gt], pred_offset, gt_offset)
            )
            pairs = torch.cat((pairs, new_pairs), dim=0)

        return pairs
