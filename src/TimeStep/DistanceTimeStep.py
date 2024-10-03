import numpy as np
from src.TimeStep.ITimeStep import ITimeStep

from pandas import DataFrame


class DistanceTimeStep(ITimeStep):
    """
    Uses the distance of the hit from the origin to determine the time-step.
    """

    def __init__(self, z_scale: float = 1 / 3, steps: int = 11):
        """
        :param float z_scale: The scale factor for the z axis.
        :param int steps: The number of time steps.
        """

        self.z_scale = z_scale
        self.steps = steps

    def define_time_step(self, hits: DataFrame):
        """
        Defines the time-step for each hit in the DataFrame.

        :param pandas.DataFrame hits: The hits DataFrame.
        """

        hits["d"] = np.sqrt(hits["x"] ** 2 + hits["y"] ** 2 + (hits["z"] * self.z_scale) ** 2)
        hits["t"] = (hits["d"] / hits["d"].max() * self.steps).astype(int)
        hits[hits["t"] >= self.steps - 1] = self.steps - 2
        hits.drop(columns=["d"], inplace=True)

    def get_num_time_steps(self) -> int:
        """
        Returns the number of time-steps.

        :return: The number of time-steps.
        """

        return self.steps
