from src.TimeStep.ForEquidistant import ITimeStepForEquidistant
from src.TimeStep.ForEquidistant.PlacementStrategy import IPlacementStrategy, SqueezedSinusoidStrategy
from .VLRings import VL_TO_RING
from .VLMaps import get_volume_layer_to_t, get_t_to_volume_layers

import torch
from torch import Tensor
from pandas import DataFrame, Series


class VLTimeStep(ITimeStepForEquidistant):
    """
    Uses the volume-layer to time-step mapping to determine the time-step.
    """

    def __init__(self, map_index: int = 0, placement_strategy: IPlacementStrategy = SqueezedSinusoidStrategy()):
        """
        :param int map_index: The index of the volume layer to time-step mapping to use.
        :param IPlacementStrategy placement_strategy: The placement strategy to use.
        """

        self.vl_to_t, self.num_t = get_volume_layer_to_t(map_index)
        self.t_to_vls = get_t_to_volume_layers(self.vl_to_t, self.num_t)
        self.placement_strategy = placement_strategy

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

    def place_equidistant_hits(self, t: int, size: Tensor, device: str = "cpu") -> Tensor:
        """
        Places equidistant hits in the detector for a given time-step.

        :param int t: The time-step.
        :param torch.Tensor size: The number of hits to place in each ring.
        :param str device: The device to use.
        :return: The placed hits.
        """

        # List of R (volume_id, layer_id) tuples
        vls = self.t_to_vls[t]
        # Tensor of Rx4 with (r1, r2, z1, z2) for each of the R rings
        rings = torch.zeros((len(vls), 4), device=device, dtype=torch.float32)
        for i, (volume_id, layer_id) in enumerate(vls):
            rings[i] = torch.tensor(VL_TO_RING[volume_id][layer_id], device=device, dtype=torch.float32)

        # Tensor of BxR with the number of hits to place in each of the R rings, within each of the B batches
        ring_capacities = (rings[:, 3] - rings[:, 2] + 1) * torch.pi * (rings[:, 1] ** 2 - rings[:, 0] ** 2 + 1)
        ring_capacities = torch.outer(size, ring_capacities / ring_capacities.sum())
        ring_capacities = ring_capacities.round().int()
        remaining = size.sum(dtype=torch.int32) - ring_capacities.sum(dtype=torch.int32)
        ring_capacities[: remaining.item()] += 1

        # Generate and return the hit set using the placement strategy, as a tensor of shape `[sum(size), 3]`
        return self.placement_strategy.place_points_in_rings(rings, ring_capacities)
