from torch import Tensor

from src.TimeStep import ITimeStep


class ITimeStepForEquidistant(ITimeStep):
    def place_equidistant_hits(self, t: int, size: Tensor) -> Tensor:
        """
        Place equidistant hits on the hit surface.

        :param int t: The pseudo time step to place the hits for
        :param int size: The number of hits to place
        :return: The placed hits. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError
