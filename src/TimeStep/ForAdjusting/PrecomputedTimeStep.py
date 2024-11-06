from pandas import DataFrame
from torch import Tensor
from src.Data import PrecomputedDataLoader
from src.Util import CoordinateSystemEnum
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
        self.fallback_time_step = time_step_for_generating

    def get_num_time_steps(self) -> int:
        return self.data_loader.num_t

    def get_num_shell_parts(self, t: int) -> int:
        return self.data_loader.num_parts[t]

    def assign_to_shell_parts(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        return self.fallback_time_step.assign_to_shell_parts(hit_tensor, t, coordinate_system)

    def define_time_step(self, hits: DataFrame):
        self.fallback_time_step.define_time_step(hits)

    def normalize_hit_tensor(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        return self.fallback_time_step.normalize_hit_tensor(hit_tensor, t, coordinate_system)

    def unnormalize_hit_tensor(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        return self.fallback_time_step.unnormalize_hit_tensor(hit_tensor, t, coordinate_system)

    def place_hits(
        self,
        t: int,
        size: Tensor,
        coordinate_system: CoordinateSystemEnum,
        device: str = "cpu",
    ) -> Tensor:
        return self.fallback_time_step.place_hits(t, size, coordinate_system, device)
