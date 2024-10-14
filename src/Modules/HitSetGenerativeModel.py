import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.Modules.HitSetEncoder import HitSetEncoderEnum, GlobalPoolingEncoder
from src.Modules.HitSetGenerator.DDPM import DDPMSetGenerator
from src.Modules.HitSetGenerator.DDPM.BetaSchedules import CosineBetaSchedule
from src.Modules.HitSetProcessor import PointNetProcessor
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
        use_shell_part_sizes: bool,
        size_loss_ratio: float = 0.25,
        input_dim: int = 3,
        encoding_dim: int = 16,
        device: str = "cpu",
    ):
        """
        :param HitSetEncoderEnum encoder_type: Type of encoder to use.
        :param HitSetSizeGeneratorEnum size_generator_type: Type of size generator to use.
        :param HitSetGeneratorEnum set_generator_type: Type of set generator to use.
        :param ITimeStep time_step: Time step object used to create the pseudo-time-step and to pass to the set generator.
        :param PairingStrategyEnum pairing_strategy_type: Pairing strategy to use for reconstruction loss.
        :param CoordinateSystemEnum coordinate_system: Coordinate system to use for input.
        :param bool use_shell_part_sizes: Whether to use shell part sizes for the loss calculation.
        :param float size_loss_ratio: Ratio of the size loss to the set loss.
        :param str device: Device to run the model on.
        """

        super().__init__()

        self.time_step = time_step
        self.coordinate_system = coordinate_system
        self.use_shell_part_sizes = use_shell_part_sizes
        self.size_loss_ratio = size_loss_ratio
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.device = device
        self.to(device)

        self.information = {
            "encoder": encoder_type.value,
            "size_generator": size_generator_type.value,
            "set_generator": set_generator_type.value,
            "time_step": time_step.__class__.__name__,
            "pairing_strategy": pairing_strategy_type.value,
            "coordinate_system": coordinate_system.value,
            "using_shell_part_sizes": use_shell_part_sizes,
        }

        self.encoders = nn.ModuleList()
        self.size_generators = nn.ModuleList()
        self.set_generators = nn.ModuleList()

        for t in range(time_step.get_num_time_steps() - 1):
            if encoder_type == HitSetEncoderEnum.POINT_NET:
                processor = PointNetProcessor(device=device)
                self.encoders.append(GlobalPoolingEncoder(processor, device=device))

            num_sizes = 1 if not use_shell_part_sizes else time_step.get_num_shell_parts(t + 1)
            if size_generator_type == HitSetSizeGeneratorEnum.GAUSSIAN:
                self.size_generators.append(GaussianSizeGenerator(num_size_samples=num_sizes, device=device))

            if set_generator_type == HitSetGeneratorEnum.ADJUSTING:
                self.set_generators.append(
                    AdjustingSetGenerator(t, time_step, pairing_strategy_type, coordinate_system, device=device)
                )
            elif set_generator_type == HitSetGeneratorEnum.NONE:
                self.set_generators.append(None)
            elif set_generator_type == HitSetGeneratorEnum.DDPM:
                num_steps = 50
                beta_schedule = CosineBetaSchedule(offset=0.002, num_steps=num_steps)
                denoising_processors = [
                    PointNetProcessor(input_dim=encoding_dim + input_dim, hidden_dim=2 * input_dim, device=device)
                    for _ in range(num_steps)
                ]
                decoder = PointNetProcessor(input_dim=encoding_dim + input_dim, hidden_dim=input_dim, device=device)

                self.set_generators.append(
                    DDPMSetGenerator(
                        t,
                        time_step,
                        pairing_strategy_type,
                        coordinate_system,
                        num_steps,
                        beta_schedule,
                        denoising_processors,
                        decoder,
                        device=device,
                    )
                )

    def forward(
        self, x: Tensor, gt: Tensor, x_ind: Tensor, gt_ind: Tensor, gt_size: Tensor, t: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the hit-set generative model.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor gt_size: Ground truth hit set size tensor. Shape `[num_batches]`
            or `[num_batches, num_parts_next]`
        :return tuple[Tensor, Tensor, Tensor]: Tuple containing the generated hit set size,
            the generated hit set and the loss.
        """

        z = self.encoders[t - 1](x, x_ind)
        pred_size = self.size_generators[t - 1](z, gt, gt_ind)

        if self.set_generators[t - 1] is not None:
            # Size loss is the average number of missing / additional hits per batch, times the max pair loss
            size_loss = (torch.abs(pred_size - gt_size) * self.set_generators[t - 1].max_pair_loss).mean()

            # Calculate the used size for set generation
            with torch.no_grad():
                used_size = torch.clamp(pred_size, min=0.0).round().int()

            # Calculate the indices for the pairing strategy
            flat_size = used_size if used_size.dim() == 1 else used_size.sum(dim=1)
            pred_ind = torch.repeat_interleave(torch.arange(len(flat_size), device=self.device), flat_size)

            pred_hits, set_loss = self.set_generators[t - 1](z, gt, pred_ind, gt_ind, used_size)
            loss = size_loss * self.size_loss_ratio + set_loss * (1 - self.size_loss_ratio)
        else:
            pred_hits, loss = Tensor([]), F.mse_loss(pred_size, gt_size)

        return pred_size, pred_hits, loss

    def generate(self, x: Tensor, x_ind: Tensor, t: int) -> Tensor | None:
        """
        Generate the hit set at time t+1 given the hit set at time t.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param int t: Time step to generate the hit set for.
        :return: Generated hit set tensor. Shape `[num_hits_pred, hit_dim]`. None if no set generator is defined.
        """

        x = self.encoders[t - 1](x, x_ind)
        size = self.size_generators[t - 1].generate(x, x_ind)
        return (
            self.set_generators[t - 1].generate(size.round().int(), x, x_ind)
            if self.set_generators[t - 1] is not None
            else None
        )

    def get_info(self) -> dict[str, str]:
        """
        Get information about the model.

        :return dict[str, str]: Information about the model.
        """

        return self.information
