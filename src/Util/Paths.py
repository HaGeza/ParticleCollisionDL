import os

from src.Pairing import PairingStrategyEnum
from src.TimeStep import TimeStepEnum
from src.TimeStep.ForAdjusting.PlacementStrategy import PlacementStrategyEnum
from src.Util import CoordinateSystemEnum

DATA_DIR = "data"
PRECOMPUTED_DATA_DIR = os.path.join(DATA_DIR, "precomputed")
MODELS_DIR = "models"
RESULTS_DIR = "results"


def get_precomputed_data_path(
    root_dir: str,
    dataset: str,
    time_step: TimeStepEnum,
    coordinate_system: CoordinateSystemEnum,
    placement_strategy: PlacementStrategyEnum,
    pairing_strategy: PairingStrategyEnum,
    use_shell_parts: bool,
) -> str:
    """
    Get the path to the precomputed dataset.

    :param str root_dir: root directory
    :param str dataset: dataset name
    :param TimeStepEnum time_step: time step used for computation
    :param PlacementStrategyEnum placement_strategy: placement strategy used for computation
    :param PairingStrategyEnum pairing_strategy: pairing strategy used for computation
    :param bool use_shell_parts: whether shell parts were used for computation
    :return str: The path to the precomputed dataset
    """

    out_name = (
        f"{dataset}-{time_step.value}-{coordinate_system.value}-{placement_strategy.value}-{pairing_strategy.value}"
    )
    out_name += f"-parts" if use_shell_parts else ""
    return os.path.join(root_dir, PRECOMPUTED_DATA_DIR, out_name)
