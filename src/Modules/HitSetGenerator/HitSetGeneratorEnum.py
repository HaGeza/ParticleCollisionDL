from enum import Enum


class HitSetGeneratorEnum(Enum):
    """
    Enum for the different hit set generators.
    """

    ADJUSTING = "adjusting"
    NONE = "none"
    DDPM = "ddpm"
