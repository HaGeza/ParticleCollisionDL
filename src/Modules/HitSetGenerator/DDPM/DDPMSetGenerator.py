from torch import Tensor, nn
import torch

from src.Modules.HitSetProcessor import IHitSetProcessor
from src.Pairing import PairingStrategyEnum
from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import convert_to_cartesian
from src.Util.Distributions import log_normal_diag, log_standard_normal
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

    def _log_posterior(self, latent: Tensor, beta_ind: int) -> Tensor:
        """
        Compute log posterior.

        :param Tensor latent: Latent tensor. Shape `[encoding_dim]`
        :param int beta_ind: Index of the beta value.
        :return Tensor: Log posterior
        """

        return log_normal_diag(
            latent, torch.sqrt(1.0 - self.betas[beta_ind]) * latent, torch.log(self.betas[beta_ind])
        )

    def forward(
        self,
        z: Tensor,
        gt: Tensor,
        pred_ind: Tensor,
        gt_ind: Tensor,
        size: Tensor,
        initial_pred: Tensor = torch.tensor([]),
    ) -> tuple[Tensor, Tensor]:
        paired = initial_pred.size(0) > 0
        initial_pred = super().generate(z, size, initial_pred)

        initial_pred_cart = convert_to_cartesian(initial_pred, self.coordinate_system)
        gt_cart = convert_to_cartesian(gt, self.coordinate_system)

        pairs, num_pairs_per_batch = self.pairing_strategy.create_pairs(
            initial_pred_cart, gt_cart, pred_ind, gt_ind, paired=paired
        )
        diffs = gt_cart[pairs[:, 1]] - initial_pred_cart[pairs[:, 0]]
        input_dim = initial_pred.size(1)

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
            mus[i], log_vars[i] = out[:, :input_dim], out[:, input_dim:]

        # predict final mu with final denoising network
        pred_diffs = self.decoder(torch.cat([zs, latents[0]], dim=1))

        # calculate ELBO
        re_term = (log_standard_normal(diffs - pred_diffs)).sum(dim=1)

        kl_term = (self._log_posterior(latents[-1], -1) - log_standard_normal(latents[-1])).sum(dim=1)
        for i in range(self.num_steps - 1, -1, -1):
            kl_term_i = self._log_posterior(latents[i], i) - log_normal_diag(latents[i], mus[i], log_vars[i])
            kl_term = kl_term + kl_term_i.sum(dim=1)

        loss = -(re_term - kl_term).mean()

        return initial_pred[pairs[:, 0]] + pred_diffs, loss

    def generate(self, z: Tensor, size: Tensor, initial_pred: Tensor = torch.tensor([])) -> Tensor:
        hits_per_batch = size if size.dim() == 1 else size.sum(dim=1)
        initial_points = super().generate(z, size, initial_pred)
        input_dim = initial_points.size(1)

        # put standard normal around paired predicted points
        diffs = torch.randn_like(initial_points, device=self.device)

        # denoise movement distributions
        zs = torch.cat([z[i].repeat(hits_per_batch[i], 1) for i in range(z.size(0))], dim=0)
        for i in range(self.num_steps):
            out = self.processors[i](torch.cat([zs, diffs], dim=1))
            mu, log_var = out[:, :input_dim], out[:, input_dim:]
            diffs = mu + torch.randn_like(diffs, device=self.device) * torch.exp(0.5 * log_var)

        # move points according to sample from denoised movement distributions
        return initial_points + diffs
