from torch import nn, Tensor


class IHitSetGenerator(nn.Module):
    """
    Interface for hit set generators.
    """

    def forward(self, x: Tensor, gt: Tensor, gt_ind: Tensor, size: Tensor) -> Tensor:
        raise NotImplementedError

    def calc_loss(self, pred_tensor: Tensor, gt_tensor: Tensor, gt_ind: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, size: Tensor) -> Tensor:
        raise NotImplementedError
