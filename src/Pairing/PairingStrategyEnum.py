from enum import Enum


class PairingStrategyEnum(Enum):
    """
    Enum for the different pairing strategies. See `IPairingStrategy`.
    """

    GREEDY = "greedy"
    VEC_GREEDY = "vec_greedy"
    KD_TREE = "kd_tree"
