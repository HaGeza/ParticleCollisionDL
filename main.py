import argparse
import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.multiprocessing import set_start_method

from src.TimeStep.ForAdjusting.PlacementStrategy import EquidistantStrategy, PlacementStrategyEnum, SinusoidStrategy
from src.Util import CoordinateSystemEnum
from src.TimeStep import TimeStepEnum
from src.TimeStep.ForAdjusting import PrecomputedTimeStep
from src.TimeStep.ForAdjusting.VolumeLayer import VLTimeStep
from src.Pairing import PairingStrategyEnum
from src.Trainer import Trainer
from src.Data import CollisionEventLoader, PrecomputedDataLoader
from src.Modules.HitSetEncoder import HitSetEncoderEnum
from src.Modules.HitSetSizeGenerator import HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import HitSetGeneratorEnum
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Util.Paths import DATA_DIR, MODELS_DIR, RESULTS_DIR, PRECOMPUTED_DATA_DIR, get_precomputed_data_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

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
    ap.add_argument("-r", "--random_seed", default=42, help="random seed")
    ap.add_argument(
        "--min_lr",
        "--min_learning_rate",
        default=None,
        help="minimum learning rate for training; if not specified, equal to lr",
    )
    ap.add_argument(
        "-c",
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

    args = ap.parse_args()
    # Convert strings to enums
    args.coordinate_system = CoordinateSystemEnum(args.coordinate_system)
    args.time_step = TimeStepEnum(args.time_step)
    args.pairing_strategy = PairingStrategyEnum(args.pairing_strategy)
    args.placement_strategy = PlacementStrategyEnum(args.placement_strategy)
    args.encoder = HitSetEncoderEnum(args.encoder)
    args.size_generator = HitSetSizeGeneratorEnum(args.size_generator)
    args.set_generator = HitSetGeneratorEnum(args.set_generator)

    # Determine root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine device: cuda > (mps >) cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Set random seeds
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Set multiprocessing start method
    set_start_method("spawn", force=True)

    use_shell_part_sizes = not args.no_shell_part_sizes
    if args.no_precomputed:
        # Initialize time step
        if args.time_step == TimeStepEnum.VOLUME_LAYER:
            if args.placement_strategy == PlacementStrategyEnum.SINUSOIDAL:
                placement_strategy = SinusoidStrategy()
            else:  # if args.placement_strategy== PlacementStrategyEnum.EQUIDISTANT:
                placement_strategy = EquidistantStrategy()

            time_step = VLTimeStep(placement_strategy=placement_strategy, use_shell_part_sizes=use_shell_part_sizes)

        # Initialize data loader
        data_loader = CollisionEventLoader(
            os.path.join(root_dir, DATA_DIR, args.dataset),
            time_step,
            int(args.batch_size),
            coordinate_system=args.coordinate_system,
            device=device,
        )
    else:
        # Initialize data loader
        data_path = get_precomputed_data_path(
            root_dir,
            args.dataset,
            args.time_step,
            args.coordinate_system,
            args.placement_strategy,
            args.pairing_strategy,
            use_shell_part_sizes,
        )
        data_loader = PrecomputedDataLoader(data_path, batch_size=int(args.batch_size), device=device)
        # Initialize time step
        time_step = PrecomputedTimeStep(data_loader)

    # Initialize model
    model = HitSetGenerativeModel(
        args.encoder,
        args.size_generator,
        args.set_generator,
        time_step,
        args.pairing_strategy,
        args.coordinate_system,
        use_shell_part_sizes,
        device=device,
    )

    # Initialize optimizer
    lr = float(args.lr)
    optimizer = Adam(model.parameters(), lr=lr)

    # Initialize scheduler
    min_lr = lr if args.min_lr is None else float(args.min_lr)
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=lr, step_size_up=100)

    # Train model
    trainer = Trainer(model, optimizer, scheduler, device, models_path=MODELS_DIR, results_path=RESULTS_DIR)
    trainer.train_and_eval(int(args.epochs), data_loader)
