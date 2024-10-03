import torch
from torch import Tensor

from .CoordinateSystemEnum import CoordinateSystemEnum


def cartesian_squared_euclidean(x, y):
    """
    Calculate the squared distance between two points in Cartesian coordinates.
    Each point in the input tensor is expected to have the format (x, y, z).

    :param Tensor x: First point tensor with shape `[N, 3]` or first point with shape `[3]`
    :param Tensor y: Second point tensor. Shape `[N, 3]`
    :return Tensor: Squared distances between the two point tensors.
    """

    return torch.sum((x - y) ** 2, dim=1)


def convert_to_cartesian(x: Tensor, coordinate_system: CoordinateSystemEnum) -> Tensor:
    """
    Convert a tensor of points from the given coordinate system to Cartesian coordinates.

    :param Tensor x: The tensor of points to convert. Shape `[N, 3]`
    :return Tensor: The tensor of points in Cartesian coordinates. Shape `[N, 3]`
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return x
    elif coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        return torch.stack([x[:, 0] * torch.cos(x[:, 1]), x[:, 0] * torch.sin(x[:, 1]), x[:, 2]], dim=1)
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
