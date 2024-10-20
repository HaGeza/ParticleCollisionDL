import torch
from torch import Tensor

from .IPairingStrategy import IPairingStrategy


class GreedyStrategy(IPairingStrategy):
    """
    Pairing strategy that pairs up hits greedily based on distance. Time complexity: O(n^2)
    """

    def create_pairs_in_batch(self, args: tuple[Tensor, Tensor, int, int]) -> Tensor:
        pred, gt, pred_offset, gt_offset = args

        num_pairs = min(pred.size(0), gt.size(0))
        pairs = torch.zeros((num_pairs, 2), device=pred.device, dtype=torch.long)
        unused = torch.ones(gt.size(0), device=pred.device, dtype=torch.bool)
        indices = torch.arange(gt.size(0), device=pred.device)

        for i in range(num_pairs):
            min_ind = torch.argmin(torch.sum((pred[i] - gt[unused]) ** 2, dim=1), dim=0)
            min_ind = indices[unused][min_ind]
            pairs[i] = torch.tensor([i + pred_offset, min_ind + gt_offset], device=pred.device)
            unused[min_ind] = False

        return pairs
