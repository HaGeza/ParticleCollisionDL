import numpy as np
from torch import Tensor
import torch
from scipy.optimize import linear_sum_assignment

from .IPairingStrategy import IPairingStrategy


class HungarianAlgorithmStrategy(IPairingStrategy):
    """
    Class for pairing hits using the Hungarian algorithm. Runtime is O(n^3),
    but it is guaranteed to find the optimal pairing.
    """

    def create_pairs_in_batch(self, args: tuple[Tensor, Tensor, int, int]) -> Tensor:
        pred, gt, pred_offset, gt_offset = args

        dists = torch.cdist(pred, gt)
        pred_ind, gt_ind = linear_sum_assignment(dists.detach().cpu().numpy())

        pairs = np.stack((pred_ind + pred_offset, gt_ind + gt_offset), axis=1)
        pairs = torch.tensor(pairs, device=pred.device, dtype=torch.long)
        return pairs
