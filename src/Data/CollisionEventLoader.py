import os
from typing import Iterator
from trackml.dataset import load_event
from pandas import DataFrame
from pandas.api.typing import DataFrameGroupBy
import torch

from src.TimeStep import ITimeStep
from src.Util import CoordinateSystemEnum


class CollisionEventLoader:
    """
    DataLoader class for loading hit-data from collision events.

    Yields two tensor lists, both containing one tensor per time step:
    1. List of hit tensors for each time step - i.e. the actual hit data to be used by a model
    2. List of batch indices for each time step - i.e. the batch index (which batch it belongs to)
       for each hit in the hit tensor
    """

    def __init__(
        self,
        dataset_path: str,
        time_step: ITimeStep,
        batch_size: int,
        hits_cols: list[str] = ["x", "y", "z"],
        coordinate_system: CoordinateSystemEnum = CoordinateSystemEnum.CYLINDRICAL,
        normalize_hits: bool = True,
        device: str = "cpu",
    ):
        """
        :param str dataset_path: Path to the dataset
        :param ITimeStep time_step: Time step object used for creating the pseudo-time step column
        :param int batch_size: Batch size
        :param list[str] hits_cols: List of columns to be returned from the hits `DataFrame`
        :param CoordinateSystemEnum coordinate_system: Coordinate system to use
        :param bool normalize_hits: Whether to normalize the hit tensor
        :param str device: Device to load the data on
        """

        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.hits_cols = hits_cols
        self.hits_size = len(hits_cols)
        self.time_step = time_step
        self.num_t = time_step.get_num_time_steps()
        self.coordinate_system = coordinate_system
        self.normalize_hits = normalize_hits
        self.device = device

    def _get_t_hit_tensor(self, grouped_hits: DataFrameGroupBy, t: int) -> torch.Tensor:
        """
        Get the hit tensor for a given time step `t`.

        :param DataFrameGroupBy grouped_hits: data frame grouped by time step
        :param int t: time step
        :return: Hit tensor for the given time step
        """

        hit_tensor = torch.tensor(
            grouped_hits.get_group(t)[self.hits_cols].reset_index(drop=True).values,
            dtype=torch.float32,
            device=self.device,
        )
        if self.normalize_hits:
            hit_tensor = self.time_step.normalize_hit_tensor(hit_tensor, t)

        if self.coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
            angles = torch.atan2(hit_tensor[:, 1], hit_tensor[:, 0])
            hit_tensor[:, 0] = torch.sqrt(hit_tensor[:, 0] ** 2 + hit_tensor[:, 1] ** 2)
            hit_tensor[:, 1] = angles

        return hit_tensor

    def _reset_batch(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Get the zero tensor lists for hits and batch indices.

        :return: Tuple of zero tensor lists for hits and batch indices
        """

        return (
            [torch.empty(0, self.hits_size, device=self.device) for _ in range(self.num_t)],
            [torch.empty(0, device=self.device, dtype=torch.long, requires_grad=False) for _ in range(self.num_t)],
        )

    def __iter__(self) -> Iterator[tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """
        Iterate over the collision events and yield the hit and batch indices tensor lists.

        :return: Iterator of tuple of hit and batch indices tensor lists. If `self.coordinate_system`
            is `CoordinateSystemEnum.CARTESIAN` each row contains the x, y, z coordinates of the hit.
            If `self.coordinate_system` is `CoordinateSystemEnum.CYLINDRICAL` each row contains r, phi, z.
        """

        hits_tensor_list, batch_index_list = self._reset_batch()
        index_in_batch = 0

        # Iterate over files in the dataset
        for root, _, files in os.walk(self.dataset_path):
            event_ids = {f.split("-")[0] for f in files}

            # Iterate over event files
            for event_id in event_ids:
                hits, _cells, _particles, _truth = load_event(os.path.join(root, event_id))
                self.time_step.define_time_step(hits)
                grouped_hits = hits.groupby("t")

                # Append hits and batch indices to each time step tensor
                for t in range(self.num_t):
                    hits_tensor_t = self._get_t_hit_tensor(grouped_hits, t)
                    hits_tensor_list[t] = torch.cat([hits_tensor_list[t], hits_tensor_t], dim=0)
                    batch_index_t = torch.full(
                        (hits_tensor_t.shape[0],), index_in_batch, device=self.device, dtype=torch.long
                    )
                    batch_index_list[t] = torch.cat([batch_index_list[t], batch_index_t])

                index_in_batch += 1
                # Yield the batch when it is full
                if index_in_batch >= self.batch_size:
                    yield hits_tensor_list, batch_index_list, event_id
                    hits_tensor_list, batch_index_list = self._reset_batch()
                    index_in_batch = 0

        # Yield the last batch if it is not empty
        if index_in_batch > 0:
            yield hits_tensor_list, batch_index_list, event_id
