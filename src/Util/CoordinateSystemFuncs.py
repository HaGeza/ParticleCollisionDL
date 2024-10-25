import torch
from torch import Tensor

from .CoordinateSystemEnum import CoordinateSystemEnum


def convert_to_cartesian(x: Tensor, coordinate_system: CoordinateSystemEnum, theta_normalized: bool = False) -> Tensor:
    """
    Convert a tensor of points from the given coordinate system to Cartesian coordinates.

    :param Tensor x: The tensor of points to convert. Shape `[N, 3]`
    :param CoordinateSystemEnum coordinate_system: The coordinate system.
    :param bool theta_normalized: Whether the theta coordinate is normalized to [-1, 1].
    :return Tensor: The tensor of points in Cartesian coordinates. Shape `[N, 3]`
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return x
    if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        theta = x[:, 1] if not theta_normalized else (x[:, 1] + 1) * torch.pi
        return torch.stack([x[:, 0] * torch.cos(theta), x[:, 0] * torch.sin(theta), x[:, 2]], dim=1)
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def convert_from_cylindrical(
    x: Tensor, coordinate_system: CoordinateSystemEnum, theta_normalized: bool = False
) -> Tensor:
    """
    Convert a tensor of points from cylindrical coordinates to the given coordinate system.

    :param Tensor x: The tensor of points to convert. Shape `[N, 3]`
    :param CoordinateSystemEnum coordinate_system: The coordinate system.
    :param bool theta_normalized: Whether the theta coordinate is normalized to [-1, 1].
    :return Tensor: The tensor of points in the given coordinate system. Shape `[N, 3]`
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        theta = x[:, 1] if not theta_normalized else (x[:, 1] + 1) * torch.pi
        return torch.stack([x[:, 0] * torch.cos(theta), x[:, 0] * torch.sin(theta), x[:, 2]], dim=1)
    if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        return x
    raise ValueError(f"Unknown coordinate system: {coordinate_system}")


def normalize_theta(x: Tensor) -> Tensor:
    """
    Normalize a tensor of points in cylindrical coordinates to have theta in [-1, 1].

    :param Tensor x: The tensor of points to normalize. Shape `[..., 3]`
    :return Tensor: The tensor of points with theta normalized. Shape `[..., 3]`
    """
    x_ = x.clone()
    x_[:, 1] = (x_[:, 1] / torch.pi + 1) % 2 - 1
    return x_


def get_coord_differences(
    x: Tensor, y: Tensor, coordinate_system: CoordinateSystemEnum, theta_normalized: bool = False
) -> Tensor:
    """
    Get the coordinate differences between x and y in the given coordinate system.

    :param Tensor x: The first tensor of points. Shape `[..., 3]`.
    :param Tensor y: The second tensor of points. Shape `[..., 3]`.
        Must be broadcastable with x.
    :param CoordinateSystemEnum coordinate_system: The coordinate system.
    :param bool theta_normalized: Whether the theta coordinate is normalized to [-1, 1].
    :return Tensor: The coordinate differences. Shape `[..., 3]`.
    """

    if coordinate_system == CoordinateSystemEnum.CARTESIAN:
        return x - y
    # if coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
    diffs = x - y
    if theta_normalized:
        diffs[..., 1] = (diffs[..., 1] + 1) % 2 - 1
    else:
        diffs[..., 1] = (diffs[..., 1] + 2 * torch.pi) % (2 * torch.pi)
    return diffs
