from torch import Tensor
import torch
from .IPlacementStrategy import IPlacementStrategy


class SqueezedSinusoidStrategy(IPlacementStrategy):
    """
    Places points using sinusoids, and squeezes them towards the outer radius using another sinusoid.
    """

    def __init__(self, return_cartesian: bool = True, step_size_multiplier: float = 1567.321):
        """
        :param bool return_cartesian: Whether to return the points in cartesian coordinates.
        :param float step_size_multiplier: Multiplier for the step size. This influences how
        the generated points are placed. Generally large values with some decimal part should
        work well. Check out `notebooks/plot_hits.ipynb`.
        """

        self.return_cartesian = return_cartesian
        self.step_size_multiplier = step_size_multiplier

    def place_points_in_rings(self, rings: Tensor, ring_capacities: Tensor) -> Tensor:
        """
        Place points within rings using a squeezed sinusoid strategy.

        :param Tensor rings: Tensor of shape `[num_rings, 4]` containing the rings.
        :param Tensor ring_capacities: Tensor of shape `[num_batches, num_rings]` containing the number of points to place in each ring.
        :return: Tensor of shape `[sum(ring_capacities), 3]` containing the placed points.
        """

        step_sizes = self.step_size_multiplier * torch.pi / ring_capacities
        total_hits = ring_capacities.sum().item()
        angles = torch.zeros(total_hits, device=rings.device, dtype=torch.float32)
        indices = torch.zeros(total_hits, device=rings.device, dtype=torch.int32)
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

        widths = (torch.sin(angles * frequencies) + 1) / 2
        widths = widths * (rings[indices, 3] - rings[indices, 2]) + rings[indices, 2]

        radii = (torch.cos(angles * frequencies) + 1) / 2
        squeezes = (torch.sin((angles * frequencies) * 2) + 1) / 2
        radii = (radii * squeezes + (1 - squeezes)) * (rings[indices, 1] - rings[indices, 0]) + rings[indices, 0]

        if self.return_cartesian:
            return torch.stack([radii * torch.cos(angles), radii * torch.sin(angles), widths], dim=1)
        else:
            return torch.stack([radii, angles, widths], dim=1)
