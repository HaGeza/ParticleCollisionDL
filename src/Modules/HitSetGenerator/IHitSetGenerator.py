from torch import nn, Tensor


class IHitSetGenerator(nn.Module):
    """
    Interface for hit set generators.
    """

    def forward(self, z: Tensor, gt: Tensor, size: int) -> Tensor:
        raise NotImplementedError

    def calc_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(self, z: Tensor, size: int) -> Tensor:
        raise NotImplementedError
