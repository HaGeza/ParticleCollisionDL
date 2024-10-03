import torch
from torch import Tensor

from src.TimeStep.ForEquidistant import ITimeStepForEquidistant
from .IHitSetGenerator import IHitSetGenerator


class EquidistantSetGenerator(IHitSetGenerator):
    """
    Class for generating placing hits according to some strategy, and learning to adjust them.
    """

    def __init__(
        self,
        t: int,
        time_step: ITimeStepForEquidistant,
        encoding_dim: int = 16,
        hit_dim: int = 3,
        device: str = "cpu",
    ):
        """
        Constructor for the equidistant set generator.

        :param int t: Time step to generate hits for.
        :param ITimeStepForEquidistant time_step: Time step object used for the original input hit set,
            and for generating the equidistant hit set.
        :param int encoding_dim: Dimension of the encoding.
        :param int hit_dim: Dimension of the hit.
        :param str device: Device to load the data on.
        """

        super().__init__()

        self.device = device
        self.to(device)

        self.t = t
        self.time_step = time_step
        self.encoding_dim = encoding_dim
        self.hit_dim = hit_dim

    def forward(self, _x: Tensor, _gt: Tensor, _gt_ind: Tensor, size: Tensor) -> Tensor:
        """
        Forward pass of the equidistant set generator.

        :param Tensor _x: Input tensor. Shape `[encoding_dim]`
        :param Tensor _gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor _gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        with torch.no_grad():
            return self.time_step.place_equidistant_hits(self.t, size, self.device)

    def generate(self, _x: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor _x: (UNUSED) Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        with torch.no_grad():
            return self.time_step.place_equidistant_hits(self.t, size, self.device)

    def calc_loss(self, pred_tensor: Tensor, gt_tensor: Tensor, pred_ind: Tensor, gt_ind: Tensor) -> Tensor:
        """
        Calculate the loss of the equidistant set generator.

        :param Tensor pred_tensor: Predicted hit tensor. Shape `[num_hits_pred, hit_dim]`
        :param Tensor gt_tensor: Ground truth hit tensor. Shape `[num_hits_act, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits_pred]`
        "param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_act]`
        """

        return torch.tensor(0.0, device=self.device)
