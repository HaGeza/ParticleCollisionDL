from torch import nn, Tensor


class IHitSetSizeGenerator(nn.Module):
    """
    Interface for hit set size generators.
    """

    def forward(self, z: Tensor, gt: Tensor, gt_ind: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(z: Tensor) -> int:
        raise NotImplementedError
