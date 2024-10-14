from torch import Tensor, nn
import torch

from src.Modules.HitSetProcessor import IHitSetProcessor
from src.Pairing import PairingStrategyEnum
from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Util import CoordinateSystemEnum
from ..AdjustingSetGenerator import AdjustingSetGenerator
from .BetaSchedules import IBetaSchedule


class DDPMSetGenerator(AdjustingSetGenerator):
    def __init__(
        self,
        t: int,
        time_step: ITimeStepForAdjusting,
        pairing_strategy_type: PairingStrategyEnum,
        coordinate_system: CoordinateSystemEnum,
        num_steps: int,
        beta_schedule: IBetaSchedule,
        denoising_processors: list[IHitSetProcessor],
        decoder: IHitSetProcessor,
        device: str = "cpu",
    ):
        """
        Constructor for the adjusting set generator.

        :param int t: Time step to generate hits for.
        :param ITimeStepForAdjusting time_step: Time step object used for the original input hit set,
            and for generating the hit set.
        :param PairingStrategyEnum pairing_strategy_type: Pairing strategy to use.
        :param CoordinateSystemEnum coordinate_system: Coordinate system to use.
        :param int num_steps: Number of steps in the diffusion process.
        :param IBetaSchedule beta_schedule: Schedule for the beta values.
        :param list[IHitSetProcessor] denoising_processors: List of denoising processors.
        :param IHitSetProcessor decoder: Final denoising processor.
        :param str device: Device to load the data on.
        """

        super().__init__(
            t,
            time_step,
            pairing_strategy_type,
            coordinate_system,
            device,
        )

        self.num_steps = num_steps
        self.betas = torch.tensor(beta_schedule.get_betas())
        self.processors = nn.ModuleList(denoising_processors)
        self.decoder = decoder

    def _add_noise(self, x: Tensor, step: int) -> Tensor:
        """
        Add noise to the input tensor.

        :param Tensor x: Input tensor. Shape `[encoding_dim]`
        :param int step: Step of the diffusion process.
        :return Tensor: Noisy tensor. Shape `[encoding_dim]`
        """

        return torch.sqrt(1.0 - self.betas[step]) * x + torch.sqrt(self.betas[step]) * torch.rand_like(x)

    def forward(self, z: Tensor, gt: Tensor, pred_ind: Tensor, gt_ind: Tensor, size: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the adjusting set generator.

        :param Tensor z: Encoded input hit set. Shape `[encoding_dim]`
        :param Tensor gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits_pred]`.
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`.
        :param Tensor size: Size of the hit point-cloud to generate. Shape `[num_batches]`
            or `[num_batches, num_parts_next]`.
        :return: Generated hit set (Shape `[sum(size), hit_dim]`), and the loss
        """

        initial_points = super().generate(z, size)

        pairs, num_pairs_per_batch = self.pairing_strategy.create_pairs(initial_points, gt, pred_ind, gt_ind)
        diffs = initial_points[pairs[:, 0]] - gt[pairs[:, 1]]

        # create noisy steps
        latents = [self._add_noise(diffs, 0)]
        for i in range(1, self.num_steps):
            latents.append(self._add_noise(latents[-1], i))

        # predict mus and log_vars with denoising networks
        mus = [None for _ in range(self.num_steps)]
        log_vars = [None for _ in range(self.num_steps)]
        zs = torch.cat([z[i].repeat(num_pairs_per_batch[i], 1) for i in range(z.size(0))], dim=0)

        for i in range(self.num_steps - 1, -1, -1):
            out = self.processors[i](torch.cat([zs, latents[i - 1]], dim=1))
            mus[i], log_vars[i] = out[:, : latents[i].size(1)], out[:, latents[i].size(1) :]

        # predict final mu with final denoising network
        pred_diff = self.decoder(torch.cat([zs, latents[0]], dim=1))

        # calculate ELBO according to 2AMU20/A3

        # return the generated hit set and the loss

    def generate(self, z: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor z: (UNUSED) Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        initial_points = super().generate(z, size)

        # put standard normal around paired predicted points

        # denoise movement distributions

        # move points according to sample from denoised movement distributions

        # return the generated hit set
