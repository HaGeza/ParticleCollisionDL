from src.Data import PrecomputedDataLoader
from .ITimeStepForAdjusting import ITimeStepForAdjusting


class PrecomputedTimeStep(ITimeStepForAdjusting):
    """
    Class for a precomputed time step.
    """

    def __init__(self, data_loader: PrecomputedDataLoader):
        """
        :param int num_time_steps: The number of time steps.
        :param int num_shell_parts: The number of shell parts.
        """
        self.data_loader = data_loader

    def get_num_time_steps(self) -> int:
        return self.data_loader.num_t

    def get_num_shell_parts(self, t: int) -> int:
        return self.data_loader.num_parts[t]
