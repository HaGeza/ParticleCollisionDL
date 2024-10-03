import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.Modules.HitSetEncoder import HitSetEncoderEnum, PointNetEncoder
from src.Modules.HitSetSizeGenerator import GaussianSizeGenerator, HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import AdjustingSetGenerator, HitSetGeneratorEnum
from src.Pairing import PairingStrategyEnum
from src.TimeStep import ITimeStep
from src.Util import CoordinateSystemEnum


class HitSetGenerativeModel(nn.Module):
    """
    Model for generating hit sets at time t+1 given the hit set at time t. The model consists of three parts:
    1. An encoder that encodes the input hit set.
    2. A size generator that generates the size of the hit set at time t+1, based on the encoded hit-set.
       At training time it may also optionally make use of the ground-truth point-cloud at the next time-step.
    3. A set generator that generates the hit set at time t+1, based on the encoded hit-set and the generated size.
       At training time it may also optionally make use of the ground-truth point-cloud at the next time-step.
    """

    def __init__(
        self,
        encoder_type: HitSetEncoderEnum,
        size_generator_type: HitSetSizeGeneratorEnum,
        set_generator_type: HitSetGeneratorEnum,
        time_step: ITimeStep,
        pairing_strategy_type: PairingStrategyEnum,
        coordinate_system: CoordinateSystemEnum,
        device: str,
    ):
        """
        :param HitSetEncoderEnum encoder_type: Type of encoder to use.
        :param HitSetSizeGeneratorEnum size_generator_type: Type of size generator to use.
        :param HitSetGeneratorEnum set_generator_type: Type of set generator to use.
        :param ITimeStep time_step: Time step object used to create the pseudo-time-step and to pass to the set generator.
        """

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

            if set_generator_type == HitSetGeneratorEnum.ADJUSTING:
                self.set_generators.append(
                    AdjustingSetGenerator(t, time_step, pairing_strategy_type, coordinate_system, device=device)
                )

    def forward(self, x: Tensor, gt: Tensor, x_ind: Tensor, gt_ind: Tensor, t: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the hit-set generative model.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :return tuple[Tensor, Tensor, Tensor]: Tuple containing the generated hit set size,
            the hit set sizes used for set generation (rounded to integers), and the generated hit set.
        """

        z = self.encoders[t](x, x_ind)
        size = self.size_generators[t](z, gt, gt_ind)
        with torch.no_grad():
            size_int = size.round().int()
        return size, size_int, self.set_generators[t](z, gt, gt_ind, size_int)

    def calc_loss(
        self,
        pred_size: Tensor,
        used_size: Tensor,
        pred_tensor: Tensor,
        gt_size: Tensor,
        gt_tensor: Tensor,
        gt_ind: Tensor,
        t: int,
        size_loss_ratio: float = 0.25,
    ) -> Tensor:
        """
        Calculate the loss of the hit-set generative model. The loss consists of two parts:
        1. Loss incurred by the size prediction (MAX_PAIR_LOSS for each missing / additional predicted hit)
        2. Loss incurred by the set prediction (uses the loss function of the set generator).

        :param Tensor pred_size: Predicted hit set size tensor. Shape `[batch_num]`
        :param Tensor used_size: Hit set size used for set generation (rounded to integers). Shape `[batch_num]`
        :param Tensor pred_tensor: Predicted hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor gt_size: Ground truth hit set size tensor. Shape `[batch_num]`
        :param Tensor gt_tensor: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param int t: Time step to calculate the loss for.
        :param float size_loss_ratio: Ratio of the size loss to the set loss.
        :return: Loss tensor.
        """

        size_loss = (torch.abs(pred_size - gt_size) * self.set_generators[t].max_pair_loss).mean()
        pred_ind = torch.repeat_interleave(torch.arange(len(used_size)), used_size)
        set_loss = self.set_generators[t].calc_loss(pred_tensor, gt_tensor, pred_ind, gt_ind)

        print(f"t= {t}: Size loss: {size_loss.item()}, Set loss: {set_loss.item()}")

        return size_loss * size_loss_ratio + set_loss * (1 - size_loss_ratio)

    def generate(self, x: Tensor, x_ind: Tensor, t: int) -> Tensor:
        """
        Generate the hit set at time t+1 given the hit set at time t.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param int t: Time step to generate the hit set for.
        :return: Generated hit set tensor. Shape `[num_hits_pred, hit_dim]`
        """

        x = self.encoders[t](x, x_ind)
        size = self.size_generators[t].generate(x, x_ind)
        return self.set_generators[t].generate(size.round().int(), x, x_ind)
