import torch
from torch import Tensor
from .IPairingStrategy import IPairingStrategy


class VectorizedGreedyStrategy(IPairingStrategy):
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

        num_pairs_req = min(pred.size(0), gt.size(0))
        pairs = torch.zeros((0, 2), device=pred.device, dtype=torch.long)

        unused_pred = torch.ones(pred.size(0), device=pred.device, dtype=torch.bool)
        unused_gt = torch.ones(gt.size(0), device=pred.device, dtype=torch.bool)

        indices_pred = torch.arange(pred.size(0), device=pred.device)
        indices_gt = torch.arange(gt.size(0), device=pred.device)

        while num_pairs_req > 0:
            pred_gt_dist = torch.cdist(pred[unused_pred], gt[unused_gt])
            selected_gt = torch.argmin(pred_gt_dist, dim=1)

            selected_gt, inverse = torch.unique(selected_gt, return_inverse=True)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            selected_pred = inverse.new_empty(selected_gt.size(0)).scatter_(0, inverse, perm)

            selected_pred = indices_pred[unused_pred][selected_pred]
            selected_gt = indices_gt[unused_gt][selected_gt]
            new_pairs = torch.stack([selected_pred + pred_offset, selected_gt + gt_offset], dim=1)
            pairs = torch.cat((pairs, new_pairs), dim=0)

            unused_pred[selected_pred] = False
            unused_gt[selected_gt] = False

            num_pairs_req -= new_pairs.size(0)

        return pairs
