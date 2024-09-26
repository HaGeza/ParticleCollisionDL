from src.TimeStep.TimeStepEnum import TimeStepEnum
from src.Modules.HitSetEncoder.HitSetEncoderEnum import HitSetEncoderEnum
from src.Modules.HitSetSizeGenerator.HitSetSizeGeneratorEnum import HitSetSizeGeneratorEnum
from src.Modules.HitSetGenerator.HitSetGeneratorEnum import HitSetGeneratorEnum

from src.DataLoader.DataLoader import DataLoader
from src.TimeStep.LayerTimeStep import LayerTimeStep
from src.TimeStep.DistanceTimeStep import DistanceTimeStep
from src.Modules.HitSetEncoder.PointNetEncoder import PointNetEncoder
from src.Modules.HitSetSizeGenerator.GaussianSizeGenerator import GaussianSizeGenerator
from src.Modules.HitSetGenerator.EquidistantSetGenerator import EquidistantSetGenerator
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel

import argparse


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("-d", "--dataset", default="data/train_sample", help="path to input dataset")
    ap.add_argument("-e", "--epochs", default=100, help="number of epochs to train the model")
    ap.add_argument(
        "-t",
        "--time_step",
        default=TimeStepEnum.LAYER,
        help="type of pseudo time step to use",
        choices=[e.value for e in TimeStepEnum],
    )
    ap.add_argument(
        "--encoder",
        default=HitSetEncoderEnum.POINT_NET,
        help="type of encoder to use",
        choices=[e.value for e in HitSetEncoderEnum],
    )
    ap.add_argument(
        "--size_generator",
        default=HitSetSizeGeneratorEnum.GAUSSIAN,
        help="type of hit set size generator to use",
        choices=[e.value for e in HitSetSizeGeneratorEnum],
    )
    ap.add_argument(
        "--set_generator",
        default=HitSetGeneratorEnum.EQUIDISTANT,
        help="type of hit set generator to use",
        choices=[e.value for e in HitSetGeneratorEnum],
    )

    args = ap.parse_args()

    # Initialize time step
    if args.time_step == TimeStepEnum.LAYER:
        time_step = LayerTimeStep()
    elif args.time_step == TimeStepEnum.DISTANCE:
        time_step = DistanceTimeStep()

    # Initialize data loader
    data_loader = DataLoader(args.dataset, time_step)

    # Initialize model
    if args.encoder == HitSetEncoderEnum.POINT_NET:
        encoder = PointNetEncoder()

    if args.size_generator == HitSetSizeGeneratorEnum.GAUSSIAN:
        size_generator = GaussianSizeGenerator()

    if args.set_generator == HitSetGeneratorEnum.EQUIDISTANT:
        set_generator = EquidistantSetGenerator()

    model = HitSetGenerativeModel(encoder, size_generator, set_generator)

    # Train model
    model.train(data_loader, epochs=args.epochs)


if __name__ == "__main__":
    main()
