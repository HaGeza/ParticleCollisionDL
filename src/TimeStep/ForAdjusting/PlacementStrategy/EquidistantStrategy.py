from torch import Tensor
import torch
from src.Util.CoordinateSystemEnum import CoordinateSystemEnum
from src.Util import PI
from src.Util.CoordinateSystemFuncs import convert_from_cylindrical
from .IPlacementStrategy import IPlacementStrategy


class EquidistantStrategy(IPlacementStrategy):
    def __init__(self, z_to_r_normalizer: float = 10.0):
        """
        Initialize the equidistant placement strategy.

        :param float z_to_r_normalizer: The normalizing constant to use for calculating the disk to circle ratio.
        """
        self.z_to_r_normalizer = z_to_r_normalizer

    def place_points_in_rings(
        self, rings: Tensor, ring_capacities: Tensor, coordinate_system: CoordinateSystemEnum
    ) -> Tensor:
        """
        Place points within a ring, approximately equidistantly.

        :param Tensor rings: Tensor of shape `[num_rings, 4]` containing the rings. Each row contains the
            inner radius, outer radius, minimum z and maximum z of a ring.
        :param Tensor ring_capacities: Tensor of shape `[num_batches, num_rings]` containing the number of points
            to place in each ring.
        :param CoordinateSystemEnum coordinate_system: The coordinate system to use.
        :return: Tensor of shape `[sum(ring_capacities), 3]` containing the placed points.
        """

        r_sums = rings[:, 0] + rings[:, 1]
        r_ranges = rings[:, 1] - rings[:, 0]
        z_ranges = rings[:, 3] - rings[:, 2]
        z_r_ratios = (z_ranges + self.z_to_r_normalizer) / (r_ranges + self.z_to_r_normalizer)

        Rs = (r_ranges / (r_sums * z_r_ratios * PI**2)).unsqueeze(0) * ring_capacities
        Zs = (z_ranges / (r_sums * z_r_ratios * PI**2)).unsqueeze(0) * ring_capacities

        circles_on_r = ((Zs - Rs) / (z_r_ratios - 1)) ** (1 / 3)
        disks_on_z = circles_on_r * z_r_ratios

        circles_on_r = circles_on_r.ceil().int()
        disks_on_z = disks_on_z.ceil().int()

        hits_per_disk = (ring_capacities / disks_on_z).floor().int()
        disk_remainder = ring_capacities - hits_per_disk * disks_on_z

        radii = torch.tensor([], dtype=torch.float, device=rings.device)
        angles = torch.tensor([], dtype=torch.float, device=rings.device)
        widths = torch.tensor([], dtype=torch.float, device=rings.device)

        for b in range(ring_capacities.size(0)):
            for r in range(ring_capacities.size(1)):
                if circles_on_r[b, r] > 1:
                    radii_in_disk = torch.linspace(rings[r, 0], rings[r, 1], circles_on_r[b, r], device=rings.device)
                else:
                    radii_in_disk = torch.tensor([(rings[r, 0] + rings[r, 1]) / 2], device=rings.device)

                if disks_on_z[b, r] > 1:
                    widths_in_ring = torch.linspace(rings[r, 2], rings[r, 3], disks_on_z[b, r], device=rings.device)
                else:
                    widths_in_ring = torch.tensor([(rings[r, 2] + rings[r, 3]) / 2], device=rings.device)

                hits_per_circle = (hits_per_disk[b, r] * radii_in_disk / radii_in_disk.sum()).floor().int()
                hits_per_circle[-1] = hits_per_disk[b, r] - hits_per_circle[:-1].sum()

                for d in range(disks_on_z[b, r]):
                    for c in range(circles_on_r[b, r]):
                        num_hits = hits_per_circle[c].item()
                        if c == circles_on_r[b, r] - 1 and d < disk_remainder[b, r]:
                            num_hits += 1

                        radii = torch.cat((radii, radii_in_disk[c].repeat(num_hits)))
                        angles = torch.cat((angles, torch.linspace(0, 2 * PI, num_hits, device=rings.device)))
                        widths = torch.cat((widths, widths_in_ring[d].repeat(num_hits)))

        return convert_from_cylindrical(torch.stack((radii, angles, widths), dim=1), coordinate_system)
