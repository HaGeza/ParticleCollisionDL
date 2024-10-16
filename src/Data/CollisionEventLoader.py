import os
import random
from typing import Iterator

from trackml.dataset import load_event
from pandas.api.typing import DataFrameGroupBy
import torch
from torch import Tensor
from torch.nn import functional as F

from .IDataLoader import IDataLoader
from src.TimeStep import ITimeStep
from src.Util import CoordinateSystemEnum


class CollisionEventLoader(IDataLoader):
    """
    DataLoader class for loading hit-data from collision events.

    Yields two tensor lists and a list of event ids. Both tensor lists contain one tensor per time step:
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
        val_ratio: float = 0.1,
        shuffle: bool = True,
        device: str = "cpu",
    ):
        """
        :param str dataset_path: Path to the dataset
        :param ITimeStep time_step: Time step object used for creating the pseudo-time step column
        :param int batch_size: Batch size
        :param list[str] hits_cols: List of columns to be returned from the hits `DataFrame`
        :param CoordinateSystemEnum coordinate_system: Coordinate system to use
        :param bool normalize_hits: Whether to normalize the hit tensor
        :param float val_ratio: Ratio of the dataset to use for validation
        :param bool shuffle: Whether to shuffle the dataset
        :param str device: Device to load the data on
        """

        self.batch_size = batch_size
        self.hits_cols = hits_cols
        self.hits_size = len(hits_cols)
        self.time_step = time_step
        self.num_t = time_step.get_num_time_steps()
        self.coordinate_system = coordinate_system
        self.normalize_hits = normalize_hits
        self.shuffle = shuffle
        self.device = device

        events = []
        for root, _, files in os.walk(dataset_path):
            events += [os.path.join(root, f.split("-")[0]) for f in files if f.endswith("-hits.csv")]

        train_cutoff = int((1 - val_ratio) * len(events))
        self.train_events = events[:train_cutoff]
        self.val_events = events[train_cutoff:]

    def _get_t_hit_tensor(self, grouped_hits: DataFrameGroupBy, t: int) -> Tensor:
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

        if self.coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
            angles = torch.atan2(hit_tensor[:, 1], hit_tensor[:, 0])
            hit_tensor[:, 0] = torch.sqrt(hit_tensor[:, 0] ** 2 + hit_tensor[:, 1] ** 2)
            hit_tensor[:, 1] = angles

        if self.normalize_hits:
            hit_tensor = self.time_step.normalize_hit_tensor(hit_tensor, t, self.coordinate_system)

        return hit_tensor

    def _reset_batch(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the zero tensor lists for hits and batch indices.

        :return: Tuple of zero tensor lists for hits and batch indices
        """

        return (
            [torch.empty(0, self.hits_size, device=self.device) for _ in range(self.num_t)],
            [torch.empty(0, device=self.device, dtype=torch.long, requires_grad=False) for _ in range(self.num_t)],
        )

    def get_gt_size(
        self, gt_tensor: Tensor, gt_batch_index: Tensor, t: int, use_shell_parts: bool = True, events: list[str] = []
    ) -> tuple[Tensor, Tensor]:
        if use_shell_parts:
            part_ids = self.time_step.assign_to_shell_parts(gt_tensor, t, self.coordinate_system)
            num_parts = self.time_step.get_num_shell_parts(t)
            batch_size = torch.max(gt_batch_index) + 1

            batch_part_ids = gt_batch_index * num_parts + part_ids
            _, gt_size = torch.unique(batch_part_ids, return_counts=True)

            gt_size = F.pad(gt_size, (0, num_parts * batch_size - gt_size.size(0)), value=0)
            gt_size = gt_size.view(batch_size, num_parts)

            return gt_size.float().to(self.device), part_ids
        else:
            _, gt_size = torch.unique(gt_batch_index, return_counts=True)
            return gt_size.float().to(self.device), torch.tensor([])

    def iter_events(self, events: list[str]) -> Iterator[tuple[list[Tensor], list[Tensor], list[str]]]:
        hits_tensor_list, batch_index_list = self._reset_batch()
        event_ids = []
        index_in_batch = 0

        # Iterate over event files
        event_list = random.sample(events, len(events)) if self.shuffle else events
        print(len(event_list))
        for event in event_list:
            hits, _cells, _particles, _truth = load_event(event)
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
            event_ids.append(os.path.basename(event))
            # Yield the batch when it is full
            if index_in_batch >= self.batch_size:
                yield hits_tensor_list, batch_index_list, event_ids
                hits_tensor_list, batch_index_list = self._reset_batch()
                event_ids = []
                index_in_batch = 0

        # Yield the last batch if it is not empty
        if index_in_batch > 0:
            yield hits_tensor_list, batch_index_list, event_ids

    def __iter__(self) -> Iterator[tuple[list[Tensor], list[Tensor], list[str]]]:
        return self.iter_events(self.train_events)

    def get_batch_size(self) -> int:
        return self.batch_size
