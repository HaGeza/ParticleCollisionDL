from torch import nn, Tensor


class IHitSetSizeGenerator(nn.Module):
    """
    Interface for hit set size generators.
    """

    def forward(self, x: Tensor, gt: Tensor) -> Tensor:
        raise NotImplementedError

    def generate(z: Tensor) -> int:
        raise NotImplementedError
