import os
from typing import Iterator
from src.TimeStep import ITimeStep
import trackml
from trackml.dataset import load_event
from pandas import DataFrame


class CollisionEventLoader:
    """
    DataLoader class for loading `CollisionEvent` data
    """

    def __init__(self, dataset_path: str, time_step: ITimeStep):
        self.dataset_path = dataset_path
        self.time_step = time_step

    def __iter__(self) -> Iterator[tuple[DataFrame, DataFrame, DataFrame]]:
        for root, _, files in os.walk(self.dataset_path):
            event_ids = {f.split("-")[0] for f in files}

            for event_id in event_ids:
                hits, _cells, particles, truth = load_event(os.path.join(root, event_id))
                self.time_step.define_time_step(hits)
                yield hits, particles, truth
