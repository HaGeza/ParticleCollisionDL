from torch import nn, Tensor


class IHitSetEncoder(nn.Module):
    """
    Interface for hit set encoders.
    """

    def forward(self, x: Tensor, x_ind: Tensor) -> Tensor:
        raise NotImplementedError
