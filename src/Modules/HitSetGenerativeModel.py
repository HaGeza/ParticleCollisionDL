import torch
from torch import nn, Tensor
import torch.nn.functional as F

from src.Modules.HitSetEncoder import HitSetEncoderEnum, GlobalPoolingEncoder
from src.Modules.HitSetGenerator.DDPM import DDPMSetGenerator
from src.Modules.HitSetGenerator.DDPM.BetaSchedules import CosineBetaSchedule
from src.Modules.HitSetProcessor import HitSetProcessorEnum, LocalGNNProcessor, PointNetProcessor
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

    DDPM_PROCESSOR = "ddpm_processor"
    DDPM_NUM_STEPS = "ddpm_num_steps"

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
        - `ddpm_processor`: HitSetProcessorEnum. Processor to use in the denoising step of DDPM.
        - `ddpm_num_steps`: int. Number of steps in the diffusion process for DDPM.
        """

        super().__init__()

        self.time_step = time_step
        self.coordinate_system = coordinate_system
        self.use_shell_part_sizes = use_shell_part_sizes
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

        if coordinate_system == CoordinateSystemEnum.CARTESIAN:
            input_channels = [0, 1, 2]
        else:  # if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
            input_channels = [0, 2]

        for t in range(time_step.get_num_time_steps() - 1):
            if encoder_type == HitSetEncoderEnum.POINT_NET:
                processor = PointNetProcessor(input_channels=input_channels, device=device)
                self.encoders.append(GlobalPoolingEncoder(processor, device=device))
            elif encoder_type == HitSetEncoderEnum.LOCAL_GNN:
                processor = LocalGNNProcessor(coordinate_system, k=5, input_channels=input_channels, device=device)
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
                num_steps = kwargs.get(self.DDPM_NUM_STEPS, 100)
                beta_schedule = CosineBetaSchedule(offset=0.002, num_steps=num_steps)

                processor = kwargs.get(self.DDPM_PROCESSOR, HitSetProcessorEnum.POINT_NET)

                if processor == HitSetProcessorEnum.POINT_NET:
                    denoising_processors = [
                        PointNetProcessor(
                            input_dim=input_dim, extra_input_dim=encoding_dim, output_dim=2 * input_dim, device=device
                        )
                        for _ in range(num_steps)
                    ]
                    decoder = PointNetProcessor(
                        input_dim=input_dim, extra_input_dim=encoding_dim, output_dim=input_dim, device=device
                    )
                else:  # if processor == HitSetEncoderEnum.LOCAL_GNN
                    denoising_processors = [
                        LocalGNNProcessor(
                            coordinate_system,
                            k=2,
                            input_dim=input_dim,
                            extra_input_dim=encoding_dim,
                            output_dim=2 * input_dim,
                            num_layers=2,
                            device=device,
                        )
                        for _ in range(num_steps)
                    ]
                    decoder = LocalGNNProcessor(
                        coordinate_system,
                        k=2,
                        input_dim=input_dim,
                        extra_input_dim=encoding_dim,
                        output_dim=input_dim,
                        num_layers=2,
                        device=device,
                    )

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
        self,
        x: Tensor,
        gt: Tensor,
        x_ind: Tensor,
        gt_ind: Tensor,
        gt_size: Tensor,
        t: int,
        initial_pred: Tensor = torch.tensor([]),
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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
        :return tuple[Tensor, Tensor, Tensor, Tensor]: Tuple containing the generated
            hit set size, the generated hit set, the size loss and the set loss.
        """

        z = self.encoders[t - 1](x, x_ind, gt_size.size(0))
        pred_size = self.size_generators[t - 1](z, gt, gt_ind)
        size_loss = F.mse_loss(pred_size, gt_size)

        if self.set_generators[t - 1] is not None:
            # Use the ground truth sizes (rounded to integers) for set generation
            with torch.no_grad():
                used_size = torch.clamp(gt_size, min=0.0).round().int()

            # Calculate the indices for the pairing strategy
            flat_size = used_size if used_size.dim() == 1 else used_size.sum(dim=1)
            pred_ind = torch.repeat_interleave(torch.arange(len(flat_size), device=self.device), flat_size)

            pred_hits, set_loss = self.set_generators[t - 1](z, gt, pred_ind, gt_ind, used_size, initial_pred)
        else:
            pred_hits, set_loss = torch.tensor([]), torch.tensor(0.0, device=self.device)

        return pred_size, pred_hits, size_loss, set_loss

    def generate(
        self, x: Tensor, x_ind: Tensor, t: int, initial_pred: Tensor = torch.tensor([]), batch_size: int = 0
    ) -> tuple[Tensor, Tensor]:
        """
        Generate the hit set at time t+1 given the hit set at time t.

        :param Tensor x: Input hit tensor. Shape `[num_hits, hit_dim]`
        :param Tensor x_ind: Input hit batch index tensor. Shape `[num_hits]`
        :param int t: Time step to generate the hit set for.
        :param Tensor initial_pred: Initial prediction for the hit set at time t+1.
            Shape `[num_hits_next, hit_dim]` or empty tensor if no initial prediction is available.
        :param int batch_size: Batch size to use for the set generation. If 0, try to determine the batch size
            from `x_ind`.
        :return tuple[Tensor, Tensor]: Tuple containing the generated hit set size and the generated hit set.
            If self.set_generators[t - 1] is None, the hit set is an empty tensor.
        """

        z = self.encoders[t - 1](x, x_ind, batch_size)
        size = self.size_generators[t - 1].generate(z)
        hits = (
            self.set_generators[t - 1].generate(z, size.round().int().clamp(min=0), initial_pred)
            if self.set_generators[t - 1] is not None
            else torch.tensor([], device=self.device)
        )
        return size, hits
