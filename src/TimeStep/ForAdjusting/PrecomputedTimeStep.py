from torch import Tensor
from src.Data import PrecomputedDataLoader
from src.Util.CoordinateSystemEnum import CoordinateSystemEnum
from .ITimeStepForAdjusting import ITimeStepForAdjusting


class PrecomputedTimeStep(ITimeStepForAdjusting):
    """
    Class for a precomputed time step.
    """

    def __init__(self, data_loader: PrecomputedDataLoader, time_step_for_generating: ITimeStepForAdjusting):
        """
        :param PrecomputedDataLoader data_loader: The data loader for the precomputed time step.
        :param ITimeStepForAdjusting time_step_for_generating: The time step to use at inference time.
        """
        self.data_loader = data_loader
        self.place_hits = time_step_for_generating.place_hits

    def get_num_time_steps(self) -> int:
        return self.data_loader.num_t

    def get_num_shell_parts(self, t: int) -> int:
        return self.data_loader.num_parts[t]
