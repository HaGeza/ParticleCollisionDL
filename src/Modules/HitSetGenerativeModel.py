import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.Modules.HitSetEncoder import HitSetEncoderEnum, PointNetEncoder
from src.Modules.HitSetSizeGenerator import GaussianSizeGenerator, HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import EquidistantSetGenerator, HitSetGeneratorEnum
from src.TimeStep import ITimeStep


class HitSetGenerativeModel(nn.Module):
    """
    Model for generating hit sets at time t+1 given the hit set at time t.
    """

    def __init__(
        self,
        encoder_type: HitSetEncoderEnum,
        size_generator_type: HitSetSizeGeneratorEnum,
        set_generator_type: HitSetGeneratorEnum,
        time_step: ITimeStep,
        device: str,
    ):
        super().__init__()

        self.time_step = time_step
        self.device = device
        self.to(device)

        self.encoders = nn.ModuleList()
        self.size_generators = nn.ModuleList()
        self.set_generators = nn.ModuleList()

        for t in range(time_step.get_num_time_steps()):
            if encoder_type == HitSetEncoderEnum.POINT_NET:
                self.encoders.append(PointNetEncoder(device=device))

            if size_generator_type == HitSetSizeGeneratorEnum.GAUSSIAN:
                self.size_generators.append(GaussianSizeGenerator(device=device))

            if set_generator_type == HitSetGeneratorEnum.EQUIDISTANT:
                self.set_generators.append(EquidistantSetGenerator(t, time_step, device=device))

    def forward(self, x: Tensor, gt: Tensor, x_ind: Tensor, gt_ind: Tensor, t: int) -> tuple[int, Tensor]:
        z = self.encoders[t](x, x_ind)
        size = self.size_generators[t](z, gt, gt_ind)
        return size, self.set_generators[t](z, gt, gt_ind, size)

    def calc_loss(
        self,
        pred_size: Tensor,
        pred_tensor: Tensor,
        gt_size: Tensor,
        gt_tensor: Tensor,
        gt_ind: Tensor,
        t: int,
        size_loss_ratio: float = 0.25,
    ) -> Tensor:
        size_loss = F.mse_loss(pred_size, gt_size, reduction="mean") * size_loss_ratio
        set_loss = self.set_generators[t].calc_loss(pred_tensor, gt_tensor, gt_ind) * (1 - size_loss_ratio)
        return size_loss + set_loss

    def generate(self, x: Tensor, x_ind: Tensor, t: int) -> Tensor:
        x = self.encoders[t](x, x_ind)
        size = self.size_generators[t].generate(x, x_ind)
        return self.set_generators[t].generate(size, x, x_ind)
