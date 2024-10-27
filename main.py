import argparse
import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.multiprocessing import set_start_method

from src.Modules.HitSetGenerator.DDPM.BetaSchedules import BetaScheduleEnum
from src.Modules.HitSetProcessor import HitSetProcessorEnum
from src.TimeStep.ForAdjusting.PlacementStrategy import EquidistantStrategy, PlacementStrategyEnum, SinusoidStrategy
from src.Util import CoordinateSystemEnum
from src.TimeStep import TimeStepEnum
from src.TimeStep.ForAdjusting import PrecomputedTimeStep
from src.TimeStep.ForAdjusting.VolumeLayer import VLTimeStep
from src.Pairing import PairingStrategyEnum
from src.Trainer import Trainer, TrainingRunIO
from src.Data import CollisionEventLoader, PrecomputedDataLoader
from src.Modules.HitSetEncoder import HitSetEncoderEnum
from src.Modules.HitSetSizeGenerator import HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import HitSetGeneratorEnum
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Util.Paths import DATA_DIR, get_precomputed_data_path


def initialize_trainer_from_args(run_io: TrainingRunIO, args: argparse.Namespace, root_dir: str) -> Trainer:
    """
    Initialize a trainer from the arguments.

    :param TrainingRunIO run_io: Training run IO object
    :param argparse.Namespace args: Arguments
    :param str root_dir: Root directory
    :return Trainer: Trainer object
    """
    # Determine device: cuda > (mps >) cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Convert strings to enums (if used in multiple places)
    coordinate_system = CoordinateSystemEnum(args.coordinate_system)
    time_step_type = TimeStepEnum(args.time_step)
    pairing_strategy_type = PairingStrategyEnum(args.pairing_strategy)
    placement_strategy_type = PlacementStrategyEnum(args.placement_strategy)

    # Convert other arguments (if used in multiple places)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    encoder_loss_weight = float(args.encoder_loss_w)
    size_loss_weight = float(args.size_loss_w)
    lr = float(args.lr)
    min_lr = lr if args.min_lr is None else float(args.min_lr)
    use_shell_part_sizes = not args.no_shell_part_sizes
    dataset = args.dataset

    # Initialize time step
    if time_step_type == TimeStepEnum.VOLUME_LAYER:
        if placement_strategy_type == PlacementStrategyEnum.SINUSOIDAL:
            placement_strategy = SinusoidStrategy()
        else:  # if placement_strategy== PlacementStrategyEnum.EQUIDISTANT:
            placement_strategy = EquidistantStrategy()

        time_step = VLTimeStep(placement_strategy=placement_strategy, use_shell_part_sizes=use_shell_part_sizes)
    else:
        raise NotImplementedError(f"Time step {time_step_type} is not implemented")

    if args.no_precomputed:
        # Initialize data loader
        data_loader = CollisionEventLoader(
            os.path.join(root_dir, DATA_DIR, dataset),
            time_step,
            batch_size,
            coordinate_system=coordinate_system,
            device=device,
        )
    else:
        # Initialize data loader
        data_path = get_precomputed_data_path(
            root_dir,
            dataset,
            time_step_type,
            coordinate_system,
            placement_strategy_type,
            pairing_strategy_type,
            use_shell_part_sizes,
        )
        data_loader = PrecomputedDataLoader(data_path, batch_size=batch_size, device=device)
        # Initialize time step
        time_step = PrecomputedTimeStep(data_loader, time_step)

    # Initialize model
    model = HitSetGenerativeModel(
        HitSetEncoderEnum(args.encoder),
        HitSetSizeGeneratorEnum(args.size_generator),
        HitSetGeneratorEnum(args.set_generator),
        time_step,
        pairing_strategy_type,
        coordinate_system,
        use_shell_part_sizes,
        device=device,
        variational_encoder=args.var_enc,
        pooling_levels=int(args.pooling_levels),
        ddpm_processor=HitSetProcessorEnum(args.ddpm_processor),
        ddpm_num_steps=int(args.ddpm_num_steps),
        ddpm_processor_layers=int(args.ddpm_processor_layers),
        ddpm_beta_schedule=BetaScheduleEnum(args.ddpm_beta_schedule),
        ddpm_use_reverse_posterior=args.ddpm_use_reverse_posterior,
    )

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # Initialize scheduler
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, step_size_up=100)

    # Set up runs IO
    run_io.setup(model, optimizer, scheduler, data_loader, epochs, encoder_loss_weight, size_loss_weight)

    return Trainer(
        model, optimizer, scheduler, data_loader, run_io, 0, epochs, encoder_loss_weight, size_loss_weight, device
    )


def initialize_trainer_from_checkpoint(run_io: TrainingRunIO, checkpoint: str, load_min_loss: bool = False) -> Trainer:
    """
    Initialize a trainer from a checkpoint.

    :param TrainingRunIO run_io: Training run IO object
    :param str checkpoint: Checkpoint ID
    :param bool load_min_loss: Load the model with the minimum loss
    :return Trainer: Trainer object
    """

    checkpoint = run_io.load_checkpoint(load_min_loss)

    model = checkpoint[run_io.MODEL_FIELD]

    return Trainer(
        model,
        checkpoint[run_io.OPTIMIZER_FIELD],
        checkpoint[run_io.SCHEDULER_FIELD],
        checkpoint[run_io.DATA_LOADER_NAME],
        run_io,
        checkpoint[run_io.EPOCH_FIELD],
        run_io.get_total_num_epochs(),
        checkpoint[run_io.ENCODER_LOSS_WEIGHT_FIELD],
        checkpoint[run_io.SIZE_LOSS_WEIGHT_FIELD],
        model.device,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Model ID of the checkpoint to resume training for. If `None` a new model with a random "
        f"ID is created. If '{TrainingRunIO.LATEST_NAME}' then the model with the newest creation "
        f"date (based on {TrainingRunIO.INFO_FILE}) is loaded. Otherwise, the model with the specified "
        "ID is loaded. In these two latter cases, the rest of the arguments are ignored.",
    )
    ap.add_argument("-d", "--dataset", default="train_sample", help=f"path to input dataset, relative to {DATA_DIR}")
    ap.add_argument(
        "--no_precomputed",
        default=False,
        action="store_true",
        help="do not use precomputed dataset. Using a precomputed dataset is recommended.",
    )
    ap.add_argument("-e", "--epochs", default=100, help="number of epochs to train the model")
    ap.add_argument("-b", "--batch_size", default=2, help="batch size for training")
    ap.add_argument("-l", "--lr", "--learning_rate", default=1e-3, help="learning rate for training")
    ap.add_argument(
        "--min_lr",
        "--min_learning_rate",
        default=None,
        help="minimum learning rate for training; if not specified, equal to lr",
    )
    ap.add_argument("--encoder_loss_w", default=Trainer.DEFAULT_ENCODER_LOSS_W, help="weight for the encoder loss")
    ap.add_argument("--size_loss_w", default=Trainer.DEFAULT_SIZE_LOSS_W, help="weight for the size loss")
    ap.add_argument("-r", "--random_seed", default=42, help="random seed")
    ap.add_argument(
        "--coordinate_system",
        default=CoordinateSystemEnum.CYLINDRICAL.value,
        help="coordinate system",
        choices=[e.value for e in CoordinateSystemEnum],
    )
    ap.add_argument(
        "--no_shell_part_sizes",
        action="store_true",
        help="predict the size of the whole shell instead of its parts",
        default=False,
    )
    ap.add_argument(
        "-t",
        "--time_step",
        default=TimeStepEnum.VOLUME_LAYER.value,
        help="type of pseudo time step to use",
        choices=[e.value for e in TimeStepEnum],
    )
    ap.add_argument(
        "--pairing_strategy",
        default=PairingStrategyEnum.HUNGARIAN.value,
        help="type of pairing strategy to use",
        choices=[e.value for e in PairingStrategyEnum],
    )
    ap.add_argument(
        "--placement_strategy",
        default=PlacementStrategyEnum.EQUIDISTANT.value,
        help="type of placement strategy to use",
        choices=[e.value for e in PlacementStrategyEnum],
    )
    ap.add_argument(
        "--encoder",
        default=HitSetEncoderEnum.POINT_NET.value,
        help="type of encoder to use",
        choices=[e.value for e in HitSetEncoderEnum],
    )
    ap.add_argument(
        "--size_generator",
        default=HitSetSizeGeneratorEnum.GAUSSIAN.value,
        help="type of hit set size generator to use",
        choices=[e.value for e in HitSetSizeGeneratorEnum],
    )
    ap.add_argument(
        "--set_generator",
        default=HitSetGeneratorEnum.DDPM.value,
        help="type of hit set generator to use",
        choices=[e.value for e in HitSetGeneratorEnum],
    )
    ap.add_argument(
        "--var_enc", "--variational_encoder", default=False, action="store_true", help="use a variational encoder"
    )
    ap.add_argument(
        "--pooling_levels",
        default=HitSetGenerativeModel.DEFAULT_POOLING_LEVELS,
        help="number of pooling levels for the global pooling encoder",
    )
    ap.add_argument(
        "--ddpm_processor",
        default=HitSetProcessorEnum.POINT_NET.value,
        help="type of hit set processor to use for DDPM for denoising",
        choices=[e.value for e in HitSetProcessorEnum],
    )
    ap.add_argument(
        "--ddpm_processor_layers",
        default=HitSetGenerativeModel.DDPM_DEFAULT_PROCESSOR_LAYERS,
        help="number of layers in each processor for DDPM for denoising",
    )
    ap.add_argument("--ddpm_num_steps", default=100, help="number of steps in the DDPM diffusion process")
    ap.add_argument(
        "--ddpm_beta_schedule",
        default=BetaScheduleEnum.COSINE.value,
        help="beta schedule for DDPM",
        choices=[e.value for e in BetaScheduleEnum],
    )
    ap.add_argument(
        "--ddpm_use_reverse_posterior",
        default=HitSetGenerativeModel.DDPM_DEFAULT_USE_REVERSE_POSTERIOR,
        action="store_true",
        help="use forward posterior",
    )

    args = ap.parse_args()

    # Determine root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Set random seeds
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Set multiprocessing start method
    set_start_method("spawn", force=True)

    # Initialize run IO and trainer
    run_io = TrainingRunIO(args.checkpoint)
    if not run_io.resume_from_checkpoint:
        trainer = initialize_trainer_from_args(run_io, args, root_dir)
    else:
        trainer = initialize_trainer_from_checkpoint(run_io, args.checkpoint)

    # train and evaluate model
    trainer.train_and_eval()
