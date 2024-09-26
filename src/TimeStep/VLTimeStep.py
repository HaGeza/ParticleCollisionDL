from src.TimeStep.ITimeStep import ITimeStep
from src.TimeStep.VLMaps import get_volume_layer_mapper

from pandas import DataFrame


class LayerTimeStep(ITimeStep):
    def __init__(self, map_index: int = 0):
        self.map_to_t = get_volume_layer_mapper(map_index)

    def define_time_step(self, hits: DataFrame):
        hits["t"] = hits.apply(self.map_to_t, axis=1)