from torch import Tensor

from src.TimeStep import ITimeStep


class ITimeStepForEquidistant(ITimeStep):
    def place_equidistant_hits(self, t: int, size: Tensor, device: str = "cpu") -> Tensor:
        """
        Place equidistant hits on the hit surface.

        :param int t: The pseudo time step to place the hits for
        :param int size: The number of hits to place
        :param str device: The device to use
        :return: The placed hits. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError
