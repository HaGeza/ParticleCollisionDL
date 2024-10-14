import torch
from torch import Tensor

from src.Modules.HitSetProcessor import IHitSetProcessor

from .IHitSetEncoder import IHitSetEncoder


class GlobalPoolingEncoder(IHitSetEncoder):
    """
    PointNet hit set encoder.
    """

    def __init__(self, processor: IHitSetProcessor, device: str = "cpu"):
        """
        Constructor for the PointNet encoder.

        :param IHitsSetProcessor processor: The processor to use before global pooling.
        :param str device: The device to run the encoder on.
        """
        super().__init__()

        self.processor = processor

    def forward(self, x: Tensor, x_ind: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        :param Tensor x: The input tensor containing the hit point-cloud. Shape: `[num_hits, input_dim]`.
        :param Tensor x_ind: The batch index tensor. Shape: `[num_hits]`.
        :return Tensor: The output tensor encoding information about the hit point-cloud.
        """
        x = self.processor(x)

        out_size = x_ind.max().item() + 1
        out = torch.full((out_size, x.size(1)), -float("inf"), device=x.device)

        for i in range(out_size):
            out[i] = x[x_ind == i].max(dim=0).values

        return out
