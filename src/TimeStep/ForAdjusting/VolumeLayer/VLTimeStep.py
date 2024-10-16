from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.TimeStep.ForAdjusting.PlacementStrategy import EquidistantStrategy, IPlacementStrategy
from src.Util import CoordinateSystemEnum
from .VLRings import VL_TO_RING
from .VLMaps import get_volume_layer_to_t, get_t_to_volume_layers

import torch
from torch import Tensor
from pandas import DataFrame, Series


class VLTimeStep(ITimeStepForAdjusting):
    """
    Uses the volume-layer to time-step mapping to determine the time-step.
    """

    def __init__(
        self,
        map_index: int = 0,
        placement_strategy: IPlacementStrategy = EquidistantStrategy(),
        use_shell_part_sizes: bool = True,
        normalize_hits: bool = True,
    ):
        """
        :param int map_index: The index of the volume layer to time-step mapping to use.
        :param IPlacementStrategy placement_strategy: The placement strategy to use.
        :param bool use_shell_part_sizes: Whether to use the shell part sizes for hit placement, or
            calculate the part sizes based on the volume layer sizes.
        :param bool normalize_hits: Whether to normalize the hits.
        """

        self.vl_to_t, self.num_t = get_volume_layer_to_t(map_index)
        self.t_to_vls = get_t_to_volume_layers(self.vl_to_t, self.num_t)
        self.use_shell_part_sizes = use_shell_part_sizes
        self.normalize_hits = normalize_hits
        self.placement_strategy = placement_strategy

        self.vl_scales = [0] * self.num_t
        for t in range(self.num_t):
            vls = self.t_to_vls[t]
            max_r_max, min_z_min, max_z_max = float("-inf"), float("inf"), float("-inf")
            for volume_id, layer_id in vls:
                ring = VL_TO_RING[volume_id][layer_id]
                max_r_max = max(max_r_max, ring[1])
                min_z_min = min(min_z_min, ring[2])
                max_z_max = max(max_z_max, ring[3])
            self.vl_scales[t] = max(max_r_max, (max_z_max - min_z_min) / 2)

    def map_vl_to_t(self, row: Series) -> int:
        """
        Maps a volume-layer pair to a time-step.

        :param pandas.Series row: The row containing the volume_id and layer_id.
        :return: The time-step.
        """

        return self.vl_to_t[row["volume_id"]][row["layer_id"]]

    def define_time_step(self, hits: DataFrame):
        """
        Defines the time-step for each hit in the DataFrame.

        :param pandas.DataFrame hits: The hits DataFrame.
        """

        hits["t"] = hits.apply(self.map_vl_to_t, axis=1)

    def get_num_time_steps(self) -> int:
        """
        Returns the number of time-steps.

        :return: The number of time-steps.
        """

        return self.num_t

    def _get_rings(self, t: int, device: str = "cpu") -> Tensor:
        """
        Get the rings in the t'th shell

        :param int t: The time-step
        :param str device: The device to place the rings on
        :return Tensor: The rings in the t'th shell. Shape `[num_rings, 4]`, each row contains:
            `(r_min, r_max, z_min, z_max)`.
        """

        # List of R (volume_id, layer_id) tuples
        vls = self.t_to_vls[t]
        # Tensor of Rx4 with (r1, r2, z1, z2) for each of the R rings
        rings = torch.zeros((len(vls), 4), device=device, dtype=torch.float32)
        for i, (volume_id, layer_id) in enumerate(vls):
            rings[i] = torch.tensor(VL_TO_RING[volume_id][layer_id], device=device, dtype=torch.float32)

        return rings

    def place_hits(
        self,
        t: int,
        size: Tensor,
        coordinate_system: CoordinateSystemEnum,
        device: str = "cpu",
    ) -> Tensor:
        """
        Places hits in the detector for a given time-step.

        :param int t: The time-step.
        :param torch.Tensor size: The number of hits to place in each ring.
        :param str device: The device to place the hits on.
        :return: The placed hits.
        """

        rings = self._get_rings(t, device)

        if not self.use_shell_part_sizes:
            # Tensor of BxR with the number of hits to place in each of the R rings, within each of the B batches
            ring_capacities = (rings[:, 3] - rings[:, 2] + 1) * torch.pi * (rings[:, 1] ** 2 - rings[:, 0] ** 2 + 1)
            ring_capacities = torch.outer(size, ring_capacities / ring_capacities.sum())
            ring_capacities = ring_capacities.floor().int()
            remaining = size - ring_capacities.sum(dim=1, dtype=torch.int32)

            assert torch.all(remaining >= 0), "Negative remaining hits"

            for i in range(remaining.size(0)):
                ring_capacities[i, : remaining[i].item()] += 1
        else:
            ring_capacities = size

        # Generate and return the hit set using the placement strategy, as a tensor of shape `[sum(size), 3]`
        hits = self.placement_strategy.place_points_in_rings(rings, ring_capacities, coordinate_system)
        return self.normalize_hit_tensor(hits, t, coordinate_system) if self.normalize_hits else hits

    def normalize_hit_tensor(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        """
        Normalizes the hit tensor for the given time-step

        :param torch.Tensor hit_tensor: The hit tensor to normalize
        :param int t: The time-step to normalize the hit tensor for
        :param CoordinateSystemEnum coordinate_system: The coordinate system used
        :return: The normalized hit tensor.
        """

        if coordinate_system == CoordinateSystemEnum.CARTESIAN:
            return hit_tensor / self.vl_scales[t]
        else:  # coordinate_system == CoordinateSystemEnum.CYLINDRICAL
            normalizer = torch.tensor([self.vl_scales[t], 1.0, self.vl_scales[t]], device=hit_tensor.device)
            return hit_tensor / normalizer

    def get_max_squared_distance(self) -> float:
        """
        Returns the maximum possible squared distance between any two hits. This assumes
        that the hits have been normalized using the `normalize_hit_tensor` method. If this
        is not the case the max pairwise squared distance is potentially as large as
        `3 * max(self.vl_scales) ** 2`, which would likely lead to numerical instability.

        :return float: The maximum possible squared distance between any two hits.
        """

        return 3.0

    def get_num_shell_parts(self, t: int) -> int:
        """
        Get the number of shell parts for the given time step.

        :param int t: The time step to get the number of shell parts for
        :return int: The number of shell parts
        """

        return len(self.t_to_vls[t])

    def assign_to_shell_parts(self, hit_tensor: Tensor, t: int, coordinate_system: CoordinateSystemEnum) -> Tensor:
        """
        Assign a hit set to shell parts for the given time step.

        :param Tensor hit_tensor: The hit tensor to assign to shell parts. Shape `[num_hits, hit_dim]`.
        :param int t: The time step to assign the hits to shell parts for
        :return Tensor: The tensor assigning each hit to a shell part. Shape `[num_hits]`.
        """

        if coordinate_system == CoordinateSystemEnum.CARTESIAN:
            rs = torch.sqrt(hit_tensor[:, 0] ** 2 + hit_tensor[:, 1] ** 2)
            zs = hit_tensor[:, 2]
        else:  # coordinate_system == CoordinateSystemEnum.CYLINDRICAL
            rs = hit_tensor[:, 0]
            zs = hit_tensor[:, 2]

        rs = rs.unsqueeze(1)
        zs = zs.unsqueeze(1)

        rings = self._get_rings(t, hit_tensor.device).unsqueeze(0)
        if self.normalize_hits:
            rings = rings / self.vl_scales[t]

        # ring_dists has shape `[num_hits, num_rings]`, where the item `(i,j)`
        # contains the manhattan distance of the i'th hit to the j'th ring if
        # the hit is outside the ring, and 0 otherwise.
        ring_dists = ((rs < rings[:, :, 0]).bool() | (rs > rings[:, :, 1]).bool()) * torch.max(
            torch.abs(rs - rings[:, :, 0]), torch.abs(rs - rings[:, :, 1])
        )
        ring_dists += ((zs < rings[:, :, 2]).bool() | (zs > rings[:, :, 3]).bool()) * torch.max(
            torch.abs(zs - rings[:, :, 2]), torch.abs(zs - rings[:, :, 3])
        )

        return torch.argmin(ring_dists, dim=1)
