import json
import os
import random
from typing import Iterator
from torch import Tensor
import torch
from .IDataLoader import IDataLoader


class PrecomputedDataLoader(IDataLoader):
    """
    DataLoader class for loading precomputed hit-set data
    """

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        val_ratio: float = 0.1,
        shuffle: bool = True,
        use_shell_parts: bool = True,
        device: str = "cpu",
    ):
        """
        :param str dataset_path: The path to the precomputed dataset
        :param int batch_size: The batch size to use
        :param float val_ratio: The ratio of the dataset to use for validation
        :param bool shuffle: Whether to shuffle the dataset
        :param bool use_shell_parts: Whether shell parts were used for computation
        :param str device: The device to use
        """

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_shell_parts = use_shell_parts
        self.device = device

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        events = []
        for root, _, files in os.walk(dataset_path):
            events += [os.path.join(root, f.rsplit(".")[0]) for f in files if f.endswith(".json")]

        train_cutoff = int((1 - val_ratio) * len(events))
        self.train_events = events[:train_cutoff]
        self.val_events = events[train_cutoff:]

        # Calculate number of time steps and shell parts
        with open(f"{events[0]}.json", "r") as f:
            data = json.load(f)
            self.num_t = len(data)
            if use_shell_parts:
                self.num_parts = [len(part_data) for part_data in data.values()]
                self.hits_size = len(list(data.values())[0][0][0])
            else:
                self.hits_size = len(list(data.values())[0][0])

        # Read ground truth sizes
        with open(os.path.join(dataset_path, "gt_sizes.json"), "r") as f:
            self.gt_sizes = json.load(f)

    def _reset_batch(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Get the zero tensor lists for hits and batch indices.

        :return: Tuple of zero tensor lists for hits and batch indices
        """

        return (
            [torch.empty(0, self.hits_size * (1 if t == 0 else 2), device=self.device) for t in range(self.num_t)],
            [torch.empty(0, device=self.device, dtype=torch.long, requires_grad=False) for _ in range(self.num_t)],
        )

    def get_gt_size(
        self, gt_tensor: Tensor, gt_batch_index: Tensor, t: int, use_shell_parts: bool = True, events: list[str] = []
    ) -> tuple[Tensor, Tensor]:
        gt_size = torch.stack(
            [self.gt_sizes[event_id][t] for event_id in events], dim=1, device=self.device, dtype=torch.long
        )
        part_ids = torch.tensor([], device=self.device, dtype=torch.long)
        if not use_shell_parts:
            part_ids = torch.cat(
                [torch.repeat_interleave(torch.arange(len(row), device=self.device), row) for row in gt_size], dim=0
            )
        return gt_size, part_ids

    def get_batch_size(self) -> int:
        return self.batch_size

    def iter_events(self, events: list[str]) -> Iterator[tuple[list[Tensor], list[Tensor], list[str]]]:
        hits_tensor_list, batch_index_list = self._reset_batch()
        event_ids = []
        index_in_batch = 0

        event_list = random.sample(events, len(events)) if self.shuffle else events
        for event in event_list:
            with open(f"{event}.json", "r") as f:
                event_data = json.load(f)

                for t_str, data in event_data.items():
                    t = int(t_str)
                    if self.use_shell_parts:
                        hit_tensor = torch.cat(
                            [torch.tensor(part_data, device=self.device, dtype=torch.float32) for part_data in data],
                            dim=0,
                        )
                    else:
                        hit_tensor = torch.tensor(data, device=self.device, dtype=torch.float32)

                    hits_tensor_list[t] = torch.cat([hits_tensor_list[t], hit_tensor], dim=0)
                    batch_index = torch.full(
                        (hit_tensor.size(0),), index_in_batch, device=self.device, dtype=torch.long
                    )
                    batch_index_list[t] = torch.cat([batch_index_list[t], batch_index], dim=0)

                event_ids.append(os.path.basename(event))
                index_in_batch += 1
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
