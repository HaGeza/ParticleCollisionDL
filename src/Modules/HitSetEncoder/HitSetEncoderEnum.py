from enum import Enum


class HitSetEncoderEnum(Enum):
    """
    Enum class for the different encoders
    """

    POINT_NET = "point_net"
    LOCAL_GNN = "local_gnn"
