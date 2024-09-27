import torch
from torch import Tensor

from src.TimeStep import ITimeStep
from .IHitSetGenerator import IHitSetGenerator


class EquidistantSetGenerator(IHitSetGenerator):
    """
    Class for generating equidistant hit sets.
    """

    def __init__(self, time_step: ITimeStep, encoding_dim: int = 16, hit_dim: int = 3, device: str = "cpu"):
        super().__init__()

        self.device = device
        self.to(device)

        self.time_step = time_step
        self.encoding_dim = encoding_dim
        self.hit_dim = hit_dim

    def forward(self, x: Tensor, _gt: Tensor, _gt_ind: Tensor, size: Tensor) -> Tensor:
        """
        Forward pass of the equidistant set generator.

        :param Tensor x: Input tensor. Shape `[encoding_dim]`
        :param Tensor gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param int size: Size of the generated hit point-cloud.
        """

        return x

    def calc_loss(self, pred_tensor: Tensor, _gt_tensor: Tensor, _gt_ind: Tensor) -> Tensor:
        """
        Calculate the loss of the equidistant set generator.

        :param Tensor pred_tensor: Predicted hit tensor. Shape `[num_hits_pred, hit_dim]`
        :param Tensor gt_tensor: Ground truth hit tensor. Shape `[num_hits_act, hit_dim]`
        "param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_act]`
        """

        return torch.tensor(0.0, device=self.device)

    def generate(self, x: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor x: Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        """

        return x
