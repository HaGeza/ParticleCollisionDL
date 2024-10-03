import argparse
import datetime
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.Util import CoordinateSystemEnum
from src.TimeStep import TimeStepEnum, DistanceTimeStep
from src.TimeStep.ForAdjusting.VolumeLayer import VLTimeStep
from src.Pairing import PairingStrategyEnum
from src.Trainer import Trainer
from src.Data import CollisionEventLoader
from src.Modules.HitSetEncoder import HitSetEncoderEnum
from src.Modules.HitSetSizeGenerator import HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator import HitSetGeneratorEnum
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--dataset", default="train_sample", help="path to input dataset")
    ap.add_argument("-e", "--epochs", default=100, help="number of epochs to train the model")
    ap.add_argument("-b", "--batch_size", default=2, help="batch size for training")
    ap.add_argument(
        "-c",
        "--coordinate_system",
        default=CoordinateSystemEnum.CYLINDRICAL.value,
        help="coordinate system",
        choices=[e.value for e in CoordinateSystemEnum],
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
        default=PairingStrategyEnum.KD_TREE.value,
        help="type of pairing strategy to use",
        choices=[e.value for e in PairingStrategyEnum],
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
        default=HitSetGeneratorEnum.ADJUSTING.value,
        help="type of hit set generator to use",
        choices=[e.value for e in HitSetGeneratorEnum],
    )

    args = ap.parse_args()
    # Convert strings to enums
    args.coordinate_system = CoordinateSystemEnum(args.coordinate_system)
    args.time_step = TimeStepEnum(args.time_step)
    args.pairing_strategy = PairingStrategyEnum(args.pairing_strategy)
    args.encoder = HitSetEncoderEnum(args.encoder)
    args.size_generator = HitSetSizeGeneratorEnum(args.size_generator)
    args.set_generator = HitSetGeneratorEnum(args.set_generator)

    # Determine root directory
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize time step
    if args.time_step == TimeStepEnum.VOLUME_LAYER:
        time_step = VLTimeStep()

    # Determine device: cuda > (mps >) cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mps could be used with:
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # but torch_scatter does not support mps

    # Initialize data loader
    data_loader = CollisionEventLoader(
        os.path.join(root_dir, "data", args.dataset),
        time_step,
        args.batch_size,
        coordinate_system=args.coordinate_system,
        device=device,
    )

    # Initialize model
    model = HitSetGenerativeModel(
        args.encoder,
        args.size_generator,
        args.set_generator,
        time_step,
        args.pairing_strategy,
        args.coordinate_system,
        device=device,
    )

    # Determine model name
    date = datetime.datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
    model_name = (
        f"{args.encoder.value}-{args.size_generator.value}-{args.set_generator.value}-{args.time_step.value}-{date}"
    )

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=1e-5)

    # Initialize scheduler
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model
    trainer = Trainer(model, optimizer, scheduler, device)
    trainer.train(data_loader, args.epochs)

    # Save model
    torch.save(model.state_dict(), os.path.join(root_dir, "models", f"{model_name}.pth"))


if __name__ == "__main__":
    main()
