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
    elif coordinate_system == CoordinateSystemEnum.CYLINDRICAL:
        return torch.stack([x[:, 0] * torch.cos(x[:, 1]), x[:, 0] * torch.sin(x[:, 1]), x[:, 2]], dim=1)
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")
