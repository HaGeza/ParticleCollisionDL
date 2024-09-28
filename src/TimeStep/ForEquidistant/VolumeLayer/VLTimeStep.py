from src.TimeStep.ForEquidistant import ITimeStepForEquidistant
from src.TimeStep.ForEquidistant.PlacementStrategy import IPlacementStrategy, SqueezedSinusoidStrategy
from .VLRings import VL_TO_RING
from .VLMaps import get_volume_layer_to_t, get_t_to_volume_layers

import torch
from torch import Tensor
from pandas import DataFrame


class VLTimeStep(ITimeStepForEquidistant):
    def __init__(self, map_index: int = 0, placement_strategy: IPlacementStrategy = SqueezedSinusoidStrategy()):
        self.vl_to_t, self.num_t = get_volume_layer_to_t(map_index)
        self.t_to_vls = get_t_to_volume_layers(self.vl_to_t, self.num_t)
        self.placement_strategy = placement_strategy

    def map_vl_to_t(self, row) -> int:
        return self.vl_to_t[row["volume_id"]][row["layer_id"]]

    def define_time_step(self, hits: DataFrame):
        hits["t"] = hits.apply(self.map_vl_to_t, axis=1)

    def get_num_time_steps(self) -> int:
        return self.num_t

    def place_equidistant_hits(self, t: int, size: Tensor, device: str = "cpu") -> Tensor:
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
