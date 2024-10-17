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

        self.device = device
        self.processor = processor

    def forward(self, x: Tensor, x_ind: Tensor, batch_size: int = 0) -> Tensor:
        x = self.processor(x)

        out_size = x_ind.max().item() + 1 if batch_size == 0 else batch_size
        out = torch.full((out_size, x.size(1)), -float("inf"), device=x.device)

        for i in range(out_size):
            batch = x[x_ind == i]
            if batch.size(0) > 0:
                out[i] = x[x_ind == i].max(dim=0).values
            else:
                out[i] = torch.zeros_like(out[i])

        return out

    def get_encoding_dim(self) -> int:
        return self.processor.get_output_dim()
