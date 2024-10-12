from pandas import DataFrame
from torch import Tensor

from src.TimeStep.TimeStepEnum import TimeStepEnum
from src.Util import CoordinateSystemEnum

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

    def normalize_hit_tensor(self, hit_tensor: Tensor, t: int) -> Tensor:
        """
        Normalize the hit tensor for the given time step.

        :param Tensor hit_tensor: The hit tensor to normalize
        :param int t: The time step to normalize the hit tensor for
        :return Tensor: The normalized hit tensor
        """

        raise NotImplementedError

    def get_num_shell_parts(self, t: int) -> int:
        """
        Get the number of shell parts for the given time step.

        :param int t: The time step to get the number of shell parts for
        :return int: The number of shell parts
        """

        raise NotImplementedError

    def assign_to_shell_parts(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        """
        Assign a hit set to shell parts for the given time step.

        :param Tensor hit_tensor: The hit tensor to assign to shell parts. Shape `[num_hits, hit_dim]`.
        :param int t: The time step to assign the hits to shell parts for
        :return Tensor: The tensor assigning each hit to a shell part. Shape `[num_hits]`.
        """

        raise NotImplementedError
