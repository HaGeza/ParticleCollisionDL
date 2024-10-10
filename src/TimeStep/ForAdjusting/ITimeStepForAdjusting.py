from torch import Tensor

from src.TimeStep import ITimeStep
from src.Util import CoordinateSystemEnum


class ITimeStepForAdjusting(ITimeStep):
    def place_hits(self, t: int, size: Tensor, coordinate_system: CoordinateSystemEnum, device: str = "cpu") -> Tensor:
        """
        Place hits on the hit surface to be adjusted later.

        :param int t: The pseudo time step to place the hits for
        :param int size: The number of hits to place
        :param CoordinateSystemEnum coordinate_system: The coordinate system to use
        :param str device: The device to load the data on
        :return: The placed hits. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError

    def get_max_squared_distance(self) -> float:
        """
        Returns the maximum possible squared distance between any two hits.

        :return float: The maximum possible squared distance between any two hits.
        """

        raise NotImplementedError
