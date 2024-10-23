import torch
from torch import Tensor

from .CoordinateSystemEnum import CoordinateSystemEnum


def convert_to_cartesian(x: Tensor, coordinate_system: CoordinateSystemEnum) -> Tensor:
    """
    Convert a tensor of points from the given coordinate system to Cartesian coordinates.

    :param Tensor x: The tensor of points to convert. Shape `[N, 3]`
    :return Tensor: The tensor of points in Cartesian coordinates. Shape `[N, 3]`
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return x
    if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        return torch.stack([x[:, 0] * torch.cos(x[:, 1]), x[:, 0] * torch.sin(x[:, 1]), x[:, 2]], dim=1)
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def convert_from_cylindrical(x: Tensor, coordinate_system: CoordinateSystemEnum) -> Tensor:
    """
    Convert a tensor of points from cylindrical coordinates to the given coordinate system.

    :param Tensor x: The tensor of points to convert. Shape `[N, 3]`
    :return Tensor: The tensor of points in the given coordinate system. Shape `[N, 3]`
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return torch.stack([x[:, 0] * torch.cos(x[:, 1]), x[:, 0] * torch.sin(x[:, 1]), x[:, 2]], dim=1)
    if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        return x
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def get_coord_differences(x: Tensor, y: Tensor, coordinate_system: CoordinateSystemEnum) -> Tensor:
    """
    Get the coordinate differences between x and y in the given coordinate system.

    :param Tensor x: The first tensor of points. Shape `[..., 3]`.
    :param Tensor y: The second tensor of points. Shape `[..., 3]`.
        Must be broadcastable with x.
    :param CoordinateSystemEnum coordinate_system: The coordinate system.
    :return Tensor: The coordinate differences. Shape `[..., 3]`.
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return x - y
    # if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
    diffs = x - y
    diffs[:, 1] = (diffs[:, 1] + 2 * torch.pi) % (2 * torch.pi)
    return diffs
