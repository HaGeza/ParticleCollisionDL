from torch import Tensor, nn
import torch

from src.Modules.HitSetProcessor import IHitSetProcessor, LocalGNNProcessor
from src.Pairing import PairingStrategyEnum
from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import convert_to_cartesian, get_coord_differences
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
        gnn_processor_used: bool = False,
        use_reverse_posterior: bool = False,
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
        :param bool gnn_processor_used: Whether the GNN processor is used.
        :param bool use_reverse_posterior: Whether to use the reverse or forward posterior.
        :param str device: Device to load the data on.
        """

        super().__init__(
            t,
            time_step,
            pairing_strategy_type,
            coordinate_system,
            device,
        )

        assert num_steps > 0, "Number of denoising steps must be greater than 0"

        self.num_steps = num_steps
        self.betas = torch.tensor(beta_schedule.get_betas())
        self.alphas = 1 - self.betas

        self.use_reverse_posterior = use_reverse_posterior

        if use_reverse_posterior:
            self.alpha_bars = torch.cumprod(self.alphas, dim=0)

            self.x0_weights = self.alpha_bars[:-1].sqrt() * self.betas[1:] / (1 - self.alpha_bars[1:])
            self.xt_weights = self.alphas[1:].sqrt() * (1 - self.alpha_bars[:-1]) / (1 - self.alpha_bars[1:])
            self.log_beta_tilde = (
                (1 - self.alpha_bars[:-1]).log() - (1 - self.alpha_bars[1:]).log() + self.betas[1:].log()
            )

        self.processors = nn.ModuleList(denoising_processors)
        self.gnn_processor_used = gnn_processor_used

    def _add_noise(self, x: Tensor, step: int) -> Tensor:
        """
        Add noise to the input tensor.

        :param Tensor x: Input tensor. Shape `[encoding_dim]`
        :param int step: Step of the diffusion process.
        :return Tensor: Noisy tensor. Shape `[encoding_dim]`
        """

        return torch.sqrt(1.0 - self.betas[step]) * x + torch.sqrt(self.betas[step]) * torch.randn_like(x)

    def _log_posterior(self, latents: list[Tensor], x0: Tensor, step: int) -> Tensor:
        """
        Compute log posterior.

        :param Tensor latents: List of latent tensors, each with shape `[encoding_dim]`
        :param Tensor x0: Initial tensor. Shape `[encoding_dim]`
        :param int step: Step of the diffusion process.
        :return Tensor: Log posterior
        """

        if self.use_reverse_posterior:
            return log_normal_diag(
                latents[step],
                self.x0_weights[step] * x0 + self.xt_weights[step] * latents[step + 1],
                self.log_beta_tilde[step],
            )
        # use reverse posterior
        prev_latent = latents[step - 1] if step > 0 else x0
        return log_normal_diag(latents[step], torch.sqrt(self.alphas[step]) * prev_latent, torch.log(self.betas[step]))

    def get_gnn_potential_neighbors(
        self, x: Tensor, x_ind: Tensor, processor: LocalGNNProcessor, k_multiplier: int = 3
    ) -> tuple[Tensor, Tensor]:
        """
        Get the potential neighbors for the GNN processor.

        :param Tensor x: Input tensor. Shape `[num_hits, hit_dim]`.
        :param Tensor x_ind: Index tensor. Shape `[num_hits]`.
        :param LocalGNNProcessor processor: GNN processor (should be one of the denoising networks).
        :param int k_multiplier: Multiplier for the number of neighbors. The number of potential
            neighbors for each node is `processor.k * k_multiplier`.
        :return tuple[Tensor, Tensor]: Potential neighbor indices and potential neighbor differences,
            with shapes `[num_hits, num_potential]` and `[num_hits, num_potential, hit_dim]`,
            respectively, where `num_potential = processor.k * k_multiplier`.
        """
        num_potential = processor.k * k_multiplier
        potential_neighbor_inds = torch.empty([0, num_potential], device=self.device, dtype=torch.long)
        potential_neighbor_diffs = torch.empty([0, num_potential, x.size(1)], device=self.device)
        b_start = 0

        for b in range(x_ind.max().item() + 1):
            x_b = x[x_ind == b]

            padding_size = 0
            if x_b.size(0) < num_potential:
                padding_size = num_potential - x_b.size(0)
                x_b = torch.cat([x_b, torch.zeros([padding_size, x_b.size(1)], device=self.device)], dim=0)
            potential_n_inds_b, potential_n_diffs_b = processor.get_top_k_neighbors(x_b, k=processor.k * 3)

            # Take out padding columns
            potential_n_inds_b = potential_n_inds_b[: x_b.size(0)]
            potential_n_diffs_b = potential_n_diffs_b[: x_b.size(0)]
            # Change padding rows to zeros
            if padding_size > 0:
                potential_n_inds_b[:, -padding_size:] = 0
                potential_n_diffs_b[:, -padding_size:] = 0

            potential_neighbor_inds = torch.cat([potential_neighbor_inds, potential_n_inds_b + b_start], dim=0)
            potential_neighbor_diffs = torch.cat([potential_neighbor_diffs, potential_n_diffs_b], dim=0)

            b_start += x_b.size(0)

        return potential_neighbor_inds, potential_neighbor_diffs

    def pick_k_nearest(self, x: Tensor, potential_neighbor_inds: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """
        Pick the `k` nearest neighbors for each point in the input tensor.

        :param Tensor x: Input tensor. Shape `[num_hits, hit_dim]`.
        :param Tensor potential_neighbor_inds: Potential neighbor indices tensor.
            Shape `[num_hits, num_potential]`.
        :param int k: Number of neighbors to pick.
        :return tuple[Tensor, Tensor]: Indices of the `k` nearest neighbors and their differences,
            with shapes `[num_hits, k]` and `[num_hits, k, hit_dim]`, respectively.
        """

        neighbors = x[potential_neighbor_inds].transpose(0, 1)
        dists = ((x - neighbors) ** 2).sum(dim=-1).sqrt().T

        topk_inds = torch.topk(dists, k + 1, dim=1, largest=False, sorted=True)[1][:, 1:].contiguous()

        # `topk_inds` contains indices within the potential neighbors tensor, so convert these
        # to the original indices. This should be done with `torch.scatter` but that produces
        # CUDA errors on snellius.
        row_inds = torch.arange(topk_inds.size(0), device=topk_inds.device).unsqueeze(1).expand_as(topk_inds)
        topk_inds = potential_neighbor_inds[row_inds, topk_inds]

        topk_diffs = get_coord_differences(
            x, x[topk_inds].transpose(0, 1), self.coordinate_system, theta_normalized=True
        )
        topk_diffs = topk_diffs.transpose(0, 1)

        return topk_inds, topk_diffs

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

        initial_pred_cart = convert_to_cartesian(initial_pred, self.coordinate_system, theta_normalized=True)
        gt_cart = convert_to_cartesian(gt, self.coordinate_system, theta_normalized=True)

        pairs, num_pairs_per_batch = self.pairing_strategy.create_pairs(
            initial_pred_cart, gt_cart, pred_ind, gt_ind, paired=paired
        )
        diffs = get_coord_differences(
            gt[pairs[:, 1]], initial_pred[pairs[:, 0]], self.coordinate_system, theta_normalized=True
        )
        input_dim = initial_pred.size(1)

        # Forward process: create noisy steps. The first latent is the least noisy,
        # and the last latent is the most noisy.
        latents = [self._add_noise(diffs, 0)]
        for i in range(1, self.num_steps):
            latents.append(self._add_noise(latents[-1], i))

        # predict mus and log_vars with denoising networks
        mus = [None for _ in range(self.num_steps - 1)]
        log_vars = [None for _ in range(self.num_steps - 1)]
        zs = torch.cat([z[i].repeat(num_pairs_per_batch[i], 1) for i in range(z.size(0))], dim=0)

        neighbor_inds, neighbor_diffs = None, None
        if self.gnn_processor_used:
            processor: LocalGNNProcessor = self.processors[0]
            potential_neighbor_inds, potential_neighbor_diffs = self.get_gnn_potential_neighbors(
                initial_pred, pred_ind, processor
            )
            neighbor_inds = potential_neighbor_inds[:, : processor.k]
            neighbor_diffs = potential_neighbor_diffs[:, : processor.k]

        # Reverse process: denoise latent steps. Remove noise from the last latent to the first latent.
        for i in range(self.num_steps - 1, 0, -1):
            out = self.processors[i](
                latents[i],
                pred_ind,
                encodings_for_ddpm=zs,
                neighbor_inds=neighbor_inds,
                neighbor_diffs=neighbor_diffs,
            )
            mus[i - 1], log_vars[i - 1] = out[:, :input_dim], out[:, input_dim:]

            if self.gnn_processor_used:
                neighbor_inds, neighbor_diffs = self.pick_k_nearest(
                    initial_pred[pairs[:, 0]] + mus[i - 1], potential_neighbor_inds, processor.k
                )

        # Denoise the first latent to predict the noise added in the first step
        pred_diffs = self.processors[0](
            latents[0], pred_ind, encodings_for_ddpm=zs, neighbor_inds=neighbor_inds, neighbor_diffs=neighbor_diffs
        )[:, :input_dim]

        # calculate ELBO
        re_term = (log_standard_normal(diffs - pred_diffs)).sum(dim=1)

        kl_term = 0
        for i in range(self.num_steps - 2, -1, -1):
            kl_term_i = self._log_posterior(latents, diffs, i) - log_normal_diag(latents[i], mus[i], log_vars[i])
            kl_term = kl_term + kl_term_i.sum(dim=1)

        loss = -(re_term - kl_term).mean()

        return initial_pred[pairs[:, 0]] + pred_diffs, loss

    def generate(self, z: Tensor, size: Tensor, initial_pred: Tensor = torch.tensor([])) -> Tensor:
        hits_per_batch = size if size.dim() == 1 else size.sum(dim=1)
        pred_ind = torch.repeat_interleave(torch.arange(size.size(0), device=self.device), hits_per_batch)
        initial_points = super().generate(z, size, initial_pred)
        input_dim = initial_points.size(1)

        # Put noise around initial points
        diffs = torch.randn_like(initial_points, device=self.device)

        # Calculate potential neighbors for GNN processor
        neighbor_inds, neighbor_diffs = None, None
        if self.gnn_processor_used:
            processor: LocalGNNProcessor = self.processors[0]
            potential_neighbor_inds, potential_neighbor_diffs = self.get_gnn_potential_neighbors(
                initial_points, pred_ind, processor
            )
            neighbor_inds = potential_neighbor_inds[:, : processor.k]
            neighbor_diffs = potential_neighbor_diffs[:, : processor.k]

        # Reverse process: denoise latent steps. Remove noise first from pure noise, then from the
        # first output and so on.
        zs = torch.cat([z[i].repeat(hits_per_batch[i], 1) for i in range(z.size(0))], dim=0)
        for i in range(self.num_steps - 1, 0, -1):
            out = self.processors[i](
                diffs, pred_ind, encodings_for_ddpm=zs, neighbor_inds=neighbor_inds, neighbor_diffs=neighbor_diffs
            )
            mu, log_var = out[:, :input_dim], out[:, input_dim:]
            diffs = mu + torch.randn_like(diffs, device=self.device) * torch.exp(0.5 * log_var)

            if self.gnn_processor_used:
                neighbor_inds, neighbor_diffs = self.pick_k_nearest(
                    initial_points + diffs, potential_neighbor_inds, processor.k
                )

        diffs = self.processors[i](
            diffs, pred_ind, encodings_for_ddpm=zs, neighbor_inds=neighbor_inds, neighbor_diffs=neighbor_diffs
        )[:, :input_dim]

        # move points according to sample from denoised movement distributions
        return initial_points + diffs
