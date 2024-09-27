import os
from typing import Iterator
from src.TimeStep import ITimeStep
import trackml
from trackml.dataset import load_event
from pandas import DataFrame
from pandas.api.typing import DataFrameGroupBy
import torch


class CollisionEventLoader:
    """
    DataLoader class for loading `CollisionEvent` data
    """

    def __init__(
        self,
        dataset_path: str,
        time_step: ITimeStep,
        batch_size: int,
        hits_cols: list[str] = ["x", "y", "z"],
        device: str = "cpu",
    ):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.hits_cols = hits_cols
        self.hits_size = len(hits_cols)
        self.time_step = time_step
        self.num_t = time_step.get_num_time_steps()
        self.device = device

    def _get_t_hit_tensor(self, grouped_hits: DataFrameGroupBy, t: int) -> torch.Tensor:
        return torch.tensor(
            grouped_hits.get_group(t)[self.hits_cols].reset_index(drop=True).values,
            dtype=torch.float32,
            device=self.device,
        )

    def _reset_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return (
            [torch.empty(0, self.hits_size, device=self.device) for _ in range(self.num_t)],
            [torch.empty(0, device=self.device, dtype=torch.long, requires_grad=False) for _ in range(self.num_t)],
        )

    def __iter__(self) -> Iterator[tuple[DataFrame, DataFrame, DataFrame]]:
        hits_tensor_list, batch_index_list = self._reset_batch()
        index_in_batch = 0

        for root, _, files in os.walk(self.dataset_path):
            event_ids = {f.split("-")[0] for f in files}

            for event_id in event_ids:
                hits, _cells, _particles, _truth = load_event(os.path.join(root, event_id))
                self.time_step.define_time_step(hits)
                grouped_hits = hits.groupby("t")

                for t in range(self.num_t):
                    hits_tensor_t = self._get_t_hit_tensor(grouped_hits, t)
                    hits_tensor_list[t] = torch.cat([hits_tensor_list[t], hits_tensor_t], dim=0)
                    batch_index_t = torch.full(
                        (hits_tensor_t.shape[0],), index_in_batch, device=self.device, dtype=torch.long
                    )
                    batch_index_list[t] = torch.cat([batch_index_list[t], batch_index_t])

                index_in_batch += 1
                if index_in_batch >= self.batch_size:
                    yield hits_tensor_list, batch_index_list
                    hits_tensor_list, batch_index_list = self._reset_batch()
                    index_in_batch = 0

        if index_in_batch > 0:
            yield hits_tensor_list, batch_index_list
