from torch import Tensor
import torch

from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import convert_from_cylindrical
from .IPlacementStrategy import IPlacementStrategy


class SinusoidStrategy(IPlacementStrategy):
    """
    Places points using sinusoids. Points are placed at regular intervals around the z axis,
    with the radius and z position calculated using sinusoids. Additionally, some Gaussian
    noise is used to spread out the hits more.
    """

    def __init__(self, step_size_multiplier: float = 1567.321):
        """
        :param float step_size_multiplier: Multiplier for the step size. This influences how
        the generated points are placed. Generally large values with some decimal part should
        work well. Check out `notebooks/plot_hits.ipynb`.
        """

        self.step_size_multiplier = step_size_multiplier

    def place_points_in_rings(
        self, rings: Tensor, ring_capacities: Tensor, coordinate_system: CoordinateSystemEnum
    ) -> Tensor:
        """
        Place points within rings using a squeezed sinusoid strategy.

        :param Tensor rings: Tensor of shape `[num_rings, 4]` containing the rings.
        :param Tensor ring_capacities: Tensor of shape `[num_batches, num_rings]` containing
            the number of points to place in each ring.
        :param CoordinateSystemEnum coordinate_system: The coordinate system to use.
        :return: Tensor of shape `[sum(ring_capacities), 3]` containing the placed points.
        """

        # step sizes (above 2) change the ordering of the points around the circle
        step_sizes = self.step_size_multiplier * torch.pi / ring_capacities
        total_hits = ring_capacities.sum().item()
        angles = torch.zeros(total_hits, device=rings.device, dtype=torch.float32)
        indices = torch.zeros(total_hits, device=rings.device, dtype=torch.int32)
        # different frequencies are used for each ring, proportional to the number of
        # points in the ring. This is used to ensure that angles close to each other
        # produce majorly different points.
        frequencies = torch.zeros(total_hits, device=rings.device, dtype=torch.float32)

        start = 0
        for b in range(ring_capacities.size(0)):
            for r in range(ring_capacities.size(1)):
                end = start + ring_capacities[b, r].item()
                angles[start:end] = (
                    torch.arange(0, ring_capacities[b, r], device=rings.device, dtype=torch.float32) * step_sizes[b, r]
                )
                indices[start:end] = r
                frequencies[start:end] = end - start + 1
                start = end

        widths = (torch.sin((angles + angles % 2 - 1) * frequencies) + 1) / 2
        widths = widths * (rings[indices, 3] - rings[indices, 2]) + rings[indices, 2]

        radii = (torch.cos((angles + angles % 2 - 1) * frequencies) + 1) / 2
        radii = radii * (rings[indices, 1] - rings[indices, 0]) + rings[indices, 0]

        return convert_from_cylindrical(torch.stack((radii, angles, widths), dim=1), coordinate_system)
