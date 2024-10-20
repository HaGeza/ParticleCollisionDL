from typing import Iterator
from torch import Tensor


class IDataLoader:
    """
    Interface for DataLoader classes
    """

    def get_gt_size(
        self, gt_tensor: Tensor, gt_batch_index: Tensor, t: int, use_shell_parts: bool = True, events: list[str] = []
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate ground truth sizes for shells / shell-parts

        :param Tensor gt_tensor: The ground truth hit tensor
        :param Tensor gt_batch_index: The ground truth batch index tensor
        :param int t: The time
        :param bool use_shell_part: Whether to use shell parts
        :param str event_id: The event id, no event id is used if empty
        :param list[str] events: The list of event ids
        :return Tensor: The ground truth sizes with shape `[num_batches]` and an empty tensor
            if `use_shell_parts` is `True`, or a tuple of the ground truth sizes with shape
            `[num_batches, num_parts]` and the part ids tensor with shape `[num_hits]` otherwise.
        """
        raise NotImplementedError

    def get_batch_size(self) -> int:
        """
        :return int: The batch size of the data loader
        """
        raise NotImplementedError

    def iter_events(self, events: list[str]) -> Iterator[tuple[list[Tensor], list[Tensor], list[str]]]:
        """
        Iterate over the collision events and yield the hit and batch indices tensor lists.

        :return: Iterator of tuple of hit and batch indices tensor lists and event ids.
            Each tensor in the hit tensor list has shape `[num_hits, hit_dim]` if initial placements
            are not precomputed. If initial placements are precomputed, then every tensor except the
            first one has shape `[num_hits, 2 * hit_dim]`, where the second half of each row
            contains the initial placement of the corresponding hit in the first half. Each tensor in
            the batch index tensor list has shape `[num_hits]`. Each list has the same length corresponding
            to the number of time steps.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[tuple[list[Tensor], list[Tensor], list[str]]]:
        """
        Alias for `CollisionEventLoader.iter_train`
        """
        raise NotImplementedError
