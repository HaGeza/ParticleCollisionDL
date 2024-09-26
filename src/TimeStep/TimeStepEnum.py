from enum import Enum


class TimeStepEnum(Enum):
    """
    Enum class for different types of pseudo time steps
    """

    LAYER = "layer"
    DISTANCE = "distance"
