from dataclasses import dataclass
from typing import Iterator
import torch
from torch import nn, Tensor

from src.Modules.HitSetEncoder import HitSetEncoderEnum, GlobalPoolingEncoder
from src.Modules.HitSetGenerator.DDPM import DDPMSetGenerator
from src.Modules.HitSetGenerator.DDPM.BetaSchedules import CosineBetaSchedule, LinearBetaSchedule, BetaScheduleEnum
from src.Modules.HitSetProcessor import HitSetProcessorEnum, LocalGNNProcessor, PointNetProcessor
from src.Modules.HitSetSizeGenerator import GaussianSizeGenerator, HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import AdjustingSetGenerator, HitSetGeneratorEnum
from src.Pairing import PairingStrategyEnum
from src.TimeStep import ITimeStep
from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import normalize_theta


@dataclass
class HSGMLosses:
    """
    Dataclass for the losses of the hit-set generative model.
    """

    encoder_loss: Tensor
    size_loss: Tensor
    set_loss: Tensor
    total_loss: Tensor

    def __iter__(self) -> Iterator[float]:
        """
        Iterator for the losses.
        """

        yield self.encoder_loss.item()
        yield self.size_loss.item()
        yield self.set_loss.item()
        yield self.total_loss.item()

    def set_to_zeros(self, device: str = "cpu"):
        """
        Set all losses to zero.

        :param str device: Device to set the losses on.
        """

        self.encoder_loss = torch.tensor(0.0, device=device)
        self.size_loss = torch.tensor(0.0, device=device)
        self.set_loss = torch.tensor(0.0, device=device)
        self.total_loss = torch.tensor(0.0, device=device)


class HitSetGenerativeModel(nn.Module):
    """
    Model for generating hit sets at time t+1 given the hit set at time t. The model consists of three parts:
    1. An encoder that encodes the input hit set.
    2. A size generator that generates the size of the hit set at time t+1, based on the encoded hit-set.
       At training time it may also optionally make use of the ground-truth point-cloud at the next time-step.
    3. A set generator that generates the hit set at time t+1, based on the encoded hit-set and the generated size.
       At training time it may also optionally make use of the ground-truth point-cloud at the next time-step.
    """

    VARIATIONAL_ENCODER = "variational_encoder"
    POOLING_LEVELS = "pooling_levels"
    ENCODER_LAYERS = "encoder_layers"
    DDPM_PROCESSOR = "ddpm_processor"
    DDPM_NUM_STEPS = "ddpm_num_steps"
    DDPM_PROCESSOR_LAYERS = "ddpm_processor_layers"
    DDPM_BETA_SCHEDULE = "ddpm_beta_schedule"
    DDPM_USE_REVERSE_POSTERIOR = "ddpm_use_reverse_posterior"

    DEFAULT_VARIATIONAL_ENCODER = False
    DEFAULT_POOLING_LEVELS = 1
    DEFAULT_ENCODER_LAYERS = 3
    DDPM_DEFAULT_PROCESSOR = HitSetProcessorEnum.POINT_NET
    DDPM_DEFUALT_NUM_STEPS = 100
    DDPM_DEFAULT_PROCESSOR_LAYERS = 2
    DDPM_DEFAULT_BETA_SCHEDULE = BetaScheduleEnum.COSINE
    DDPM_DEFAULT_USE_REVERSE_POSTERIOR = False

    def __init__(
        self,
        encoder_type: HitSetEncoderEnum,
        size_generator_type: HitSetSizeGeneratorEnum,
        set_generator_type: HitSetGeneratorEnum,
        time_step: ITimeStep,
        pairing_strategy_type: PairingStrategyEnum,
        coordinate_system: CoordinateSystemEnum,
        use_shell_part_sizes: bool,
        input_dim: int = 3,
        encoding_dim: int = 16,
        device: str = "cpu",
        **kwargs,
    ):
        """
        :param HitSetEncoderEnum encoder_type: Type of encoder to use.
        :param HitSetSizeGeneratorEnum size_generator_type: Type of size generator to use.
        :param HitSetGeneratorEnum set_generator_type: Type of set generator to use.
        :param ITimeStep time_step: Time step object used to create the pseudo-time-step
            and to pass to the set generator.
        :param PairingStrategyEnum pairing_strategy_type: Pairing strategy to use for reconstruction loss.
        :param CoordinateSystemEnum coordinate_system: Coordinate system to use for input.
        :param bool use_shell_part_sizes: Whether to use shell part sizes for the loss calculation.
        :param int input_dim: Dimension of the input hits.
        :param int encoding_dim: Dimension of the encoded hits.
        :param str device: Device to run the model on.
        :param kwargs: Additional arguments:
        - `variational_encoder`: bool. Whether to use a variational encoder.
        - `pooling_levels`: int. Number of levels to use in the global pooling encoder.
        - `encoder_layers`: int. Number of layers in the encoder.
        - `ddpm_processor`: HitSetProcessorEnum. Processor to use in the denoising step of DDPM.
        - `ddpm_num_steps`: int. Number of steps in the diffusion process for DDPM.
        - `ddpm_processor_layers`: int. Number of layers in each processor for DDPM.
        - `ddpm_beta_schedule`: BetaScheduleEnum. Schedule for the beta parameter in DDPM.
        - `ddpm_use_reverse_posterior`: bool. Whether to use the reverse or forward posterior in the KL of DDPM.
        """

        super().__init__()

        self.time_step = time_step
        self.coordinate_system = coordinate_system
        self.pairing_strategy_type = pairing_strategy_type
        self.use_shell_part_sizes = use_shell_part_sizes
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

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

        self.min_size_to_generate = 0
        self.max_batch_size_to_generate = 50000

        if coordinate_system == CoordinateSystemEnum.CARTESIAN:
            input_channels = [0, 1, 2]
        else:  # if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
            input_channels = [0, 2]

        for t in range(time_step.get_num_time_steps() - 1):
            if encoder_type in [HitSetEncoderEnum.POINT_NET, HitSetEncoderEnum.LOCAL_GNN]:
                variational_encoder = kwargs.get(self.VARIATIONAL_ENCODER, False)
                pooling_levels = kwargs.get(self.POOLING_LEVELS, self.DEFAULT_POOLING_LEVELS)
                encoder_layers = kwargs.get(self.ENCODER_LAYERS, self.DEFAULT_ENCODER_LAYERS)

                self.information["encoder"] = {
                    "variational": variational_encoder,
                    "pooling_levels": pooling_levels,
                }

                encoder_out_dim = encoding_dim * 2 if variational_encoder else encoding_dim
                if encoder_type == HitSetEncoderEnum.POINT_NET:
                    processor = PointNetProcessor(
                        input_channels=input_channels,
                        hidden_dim=encoding_dim,
                        output_dim=encoder_out_dim,
                        num_layers=encoder_layers,
                        device=device,
                    )
                else:  # encoder_type == HitSetEncoderEnum.LOCAL_GNN
                    k = 5
                    self.min_size_to_generate = max(self.min_size_to_generate, k + 1)
                    processor = LocalGNNProcessor(
                        coordinate_system,
                        k=k,
                        input_channels=input_channels,
                        hidden_dim=encoding_dim,
                        output_dim=encoder_out_dim,
                        num_layers=encoder_layers,
                        device=device,
                    )

                self.encoding_dim = encoding_dim * pooling_levels
                self.encoders.append(
                    GlobalPoolingEncoder(
                        processor, num_levels=pooling_levels, variational=variational_encoder, device=device
                    )
                )

            num_sizes = 1 if not use_shell_part_sizes else time_step.get_num_shell_parts(t + 1)
            if size_generator_type == HitSetSizeGeneratorEnum.GAUSSIAN:
                self.size_generators.append(
                    GaussianSizeGenerator(num_size_samples=num_sizes, input_dim=self.encoding_dim, device=device)
                )

            if set_generator_type == HitSetGeneratorEnum.ADJUSTING:
                self.set_generators.append(
                    AdjustingSetGenerator(t, time_step, pairing_strategy_type, coordinate_system)
                )
            elif set_generator_type == HitSetGeneratorEnum.NONE:
                self.set_generators.append(None)
            elif set_generator_type == HitSetGeneratorEnum.DDPM:
                num_steps = kwargs.get(self.DDPM_NUM_STEPS, self.DDPM_DEFUALT_NUM_STEPS)
                processor = kwargs.get(self.DDPM_PROCESSOR, self.DDPM_DEFAULT_PROCESSOR)
                num_layers = kwargs.get(self.DDPM_PROCESSOR_LAYERS, self.DDPM_DEFAULT_PROCESSOR_LAYERS)
                beta_schedule = kwargs.get(self.DDPM_BETA_SCHEDULE, self.DDPM_DEFAULT_BETA_SCHEDULE)
                use_reverse_posterior = kwargs.get(
                    self.DDPM_USE_REVERSE_POSTERIOR, self.DDPM_DEFAULT_USE_REVERSE_POSTERIOR
                )

                self.information["ddpm"] = {
                    "num_steps": num_steps,
                    "processor": processor.value,
                    "processor_layers": num_layers,
                    "beta_schedule": beta_schedule.value,
                    "use_reverse_posterior": use_reverse_posterior,
                }

                self.set_generators.append(
                    self._create_ddpm_set_generator(
                        t, num_steps, beta_schedule, processor, input_dim, num_layers, use_reverse_posterior, device
                    )
                )

        self.device = device
        self.to(device)

    def _create_ddpm_set_generator(
        self,
        t: int,
        num_steps: int,
        beta_schedule: BetaScheduleEnum,
        processor: HitSetProcessorEnum,
        proc_in_dim: int,
        proc_num_layers: int,
        use_reverse_posterior: bool,
        device: str = "cpu",
    ) -> DDPMSetGenerator:
        """
        Create a DDPM set generator.

        :param int t: Time step to create the set generator for.
        :param int num_steps: Number of steps in the diffusion process.
        :param BetaScheduleEnum beta_schedule: Schedule for the beta parameter of the diffusion process.
        :param HitSetProcessorEnum processor: Processor to use in the denoising step of DDPM.
        :param int proc_in_dim: Input dimension of the processor.
        :param int proc_num_layers: Number of layers in the processor.
        :param bool use_reverse_posterior: Whether to use the reverse or forward posterior in the KL of DDPM.
        :param str device: Device to run the model on.
        :return DDPMSetGenerator: The created DDPM set generator.
        """

        if beta_schedule == BetaScheduleEnum.COSINE:
            beta_schedule = CosineBetaSchedule(offset=0.001, num_steps=num_steps)
        else:  # if beta_schedule == BetaScheduleEnum.LINEAR
            beta_schedule = LinearBetaSchedule(beta_start=0.01, beta_end=0.9, num_steps=num_steps)

        gnn_processor_used = False
        if processor == HitSetProcessorEnum.POINT_NET:
            denoising_processors = [
                PointNetProcessor(
                    input_dim=proc_in_dim,
                    extra_input_dim=self.encoding_dim,
                    output_dim=proc_in_dim * mult,
                    num_layers=proc_num_layers,
                    device=device,
                )
                for mult in [1] + [2] * (num_steps - 1)
            ]
        else:  # if processor == HitSetEncoderEnum.LOCAL_GNN
            k = 3
            gnn_processor_used = True
            self.min_size_to_generate = max(self.min_size_to_generate, k + 1)

            denoising_processors = [
                LocalGNNProcessor(
                    self.coordinate_system,
                    k=k,
                    input_dim=proc_in_dim,
                    extra_input_dim=self.encoding_dim,
                    output_dim=proc_in_dim * mult,
                    num_layers=proc_num_layers,
                    device=device,
                )
                for mult in [1] + [2] * (num_steps - 1)
            ]

        return DDPMSetGenerator(
            t,
            self.time_step,
            self.pairing_strategy_type,
            self.coordinate_system,
            num_steps,
            beta_schedule,
            denoising_processors,
            gnn_processor_used,
            use_reverse_posterior,
            device=device,
        )

    def forward(
        self,
        x: Tensor,
        gt: Tensor,
        x_ind: Tensor,
        gt_ind: Tensor,
        gt_size: Tensor,
        t: int,
        initial_pred: Tensor = torch.tensor([]),
    ) -> tuple[Tensor, Tensor, HSGMLosses]:
        """
        Forward pass of the hit-set generative model.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor gt: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
            or `[num_hits_next, 2 * hit_dim]` if using precomputed data, in which case
            the second half of each row in `gt` is the initial position of the pair
            of the corresponding hit in the first half.
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor gt_size: Ground truth hit set size tensor. Shape `[num_batches]`
            or `[num_batches, num_parts_next]`
        :param int t: Time step to generate the hit set for.
        :param Tensor initial_pred: Initial prediction for the hit set at time t+1.
            Shape `[num_hits_next, hit_dim]` or empty tensor if no initial prediction is available.
        :return tuple[Tensor, Tensor, HSGMLosses]: Tuple containing the generated
            hit set size, the generated hit set and losses of the model.
        """

        x_, gt_, initial_pred_ = x, gt, initial_pred
        if self.coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
            x_, gt_, initial_pred_ = normalize_theta(x), normalize_theta(gt), normalize_theta(initial_pred)

        z, encoder_loss = self.encoders[t - 1](x_, x_ind, gt_size.size(0))

        pred_size = self.size_generators[t - 1](z, gt_, gt_ind)
        size_loss = self.size_generators[t - 1].calc_loss(pred_size, gt_size)

        if self.set_generators[t - 1] is not None:
            # Use the ground truth sizes (rounded to integers) for set generation
            with torch.no_grad():
                used_size = torch.clamp(gt_size.round().int(), min=self.min_size_to_generate)

            # Calculate the indices for the pairing strategy
            flat_size = used_size if used_size.dim() == 1 else used_size.sum(dim=1)
            pred_ind = torch.repeat_interleave(torch.arange(len(flat_size), device=self.device), flat_size)

            pred_hits, set_loss = self.set_generators[t - 1](z, gt_, pred_ind, gt_ind, used_size, initial_pred_)
        else:
            pred_hits, set_loss = torch.tensor([]), torch.tensor(0.0, device=self.device)

        return pred_size, pred_hits, HSGMLosses(encoder_loss, size_loss, set_loss, 0.0)

    def generate(
        self, x: Tensor, x_ind: Tensor, t: int, initial_pred: Tensor = torch.tensor([]), batch_size: int = 0
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Generate the hit set at time t+1 given the hit set at time t.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param int t: Time step to generate the hit set for.
        :param Tensor initial_pred: Initial prediction for the hit set at time t+1.
            Shape `[num_hits_next, hit_dim]` or empty tensor if no initial prediction is available.
        :param int batch_size: Batch size to use for the set generation. If 0, try to determine the batch size
            from `x_ind`.
        :return tuple[Tensor, Tensor, Tensor]: Tuple containing the generated hit set size, the generated hit set
            and the hit set size used for generation. If self.set_generators[t - 1] is None,
            the hit set is an empty tensor.
        """

        x_ = normalize_theta(x) if self.coordinate_system == CoordinateSystemEnum.CYLINDRICAL else x

        z, _ = self.encoders[t - 1](x_, x_ind, batch_size)
        size = self.size_generators[t - 1].generate(z)

        used_size = torch.clamp(size.round().int(), min=self.min_size_to_generate)

        if self.set_generators[t - 1] is None:
            return size, torch.tensor([], device=self.device), used_size

        for b in range(used_size.size(0)):
            batch_item_size = used_size[b].sum().item()
            if batch_item_size > self.max_batch_size_to_generate:
                used_size[b] = used_size[b].float() / batch_item_size * self.max_batch_size_to_generate
                used_size[b] = torch.clamp(used_size[b].round().int(), min=self.min_size_to_generate)

        hits = (
            self.set_generators[t - 1].generate(z, used_size, initial_pred)
            if self.set_generators[t - 1] is not None
            else torch.tensor([], device=self.device)
        )

        return size, hits, used_size

    def to(self, device, *args, **kwargs):
        self.device = device
        for t in range(self.time_step.get_num_time_steps() - 1):
            self.encoders[t].to(device)
            self.size_generators[t].to(device)
            if self.set_generators[t] is not None:
                self.set_generators[t].to(device)
        return super().to(device, *args, **kwargs)
