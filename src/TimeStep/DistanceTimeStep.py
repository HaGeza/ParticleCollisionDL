import numpy as np
from src.TimeStep.ITimeStep import ITimeStep

from pandas import DataFrame


class DistanceTimeStep(ITimeStep):
    def __init__(self, z_scale: float = 1 / 3, steps: int = 11):
        self.z_scale = z_scale
        self.steps = steps

    def define_time_step(self, hits: DataFrame):
        hits["d"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + (hits["z"] * self.z_scale) ** 2)
        hits["t"] = (hits["d"] / hits["d"].max() * self.steps).astype(int)
        hits[hits["t"] >= self.steps - 1] = self.steps - 2
        hits.drop(columns=["d"], inplace=True)

    def get_num_time_steps(self) -> int:
        return self.steps
