from pandas import DataFrame
from torch import Tensor

from src.TimeStep.TimeStepEnum import TimeStepEnum


class ITimeStep:
    def define_time_step(self, hits: DataFrame):
        """
        Define the pseudo time step for the given event data.

        :param DataFrame hits: The hit data to define the pseudo time step for
        """

        raise NotImplementedError

    def get_num_time_steps(self) -> int:
        """
        Get the number of pseudo time steps.

        :return int: The number of pseudo time steps
        """

        raise NotImplementedError

    def get_enum(self) -> TimeStepEnum:
        """
        Get the enum value of the time step.

        :return TimeStepEnum: The enum value of the time step
        """

        raise NotImplementedError

    def normalize_hit_tensor(self, hit_tensor: Tensor, t: int) -> Tensor:
        """
        Normalize the hit tensor for the given time step.

        :param Tensor hit_tensor: The hit tensor to normalize
        :param int t: The time step to normalize the hit tensor for
        :return Tensor: The normalized hit tensor
        """

        raise NotImplementedError
