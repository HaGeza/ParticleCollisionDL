from torch import Tensor


class IPlacementStrategy:
    """
    Interface for the placement strategy of points within rings
    """

    def place_points_in_rings(self, rings: Tensor, ring_capacities: Tensor) -> Tensor:
        """
        Place points within a ring.

        :param Tensor rings: Tensor of shape `[num_rings, 4]` containing the rings.
        :param Tensor ring_capacities: Tensor of shape `[num_batches, num_rings]` containing the number of points to place in each ring.
        :return: Tensor of shape `[sum(ring_capacities), 3]` containing the placed points.
        """

        raise NotImplementedError
