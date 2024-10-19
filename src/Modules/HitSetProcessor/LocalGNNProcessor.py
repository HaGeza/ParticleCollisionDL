from torch import Tensor
from .IHitSetProcessor import IHitSetProcessor


class LocalGNNProcessor(IHitSetProcessor):
    """
    Class to process a hit set using a GNN with local connections.
    """

    def __init__(self):
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
