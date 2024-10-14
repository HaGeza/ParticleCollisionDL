import numpy as np


class IBetaSchedule:
    """
    Interface for the beta schedule.
    """

    BETA_MIN = 0.0
    BETA_MAX = 0.999

    def get_betas(self) -> list[float]:
        """
        Get the beta values for the diffusion process.

        :return list[float]: Beta values
        """
        raise NotImplementedError


class LinearBetaSchedule(IBetaSchedule):
    """
    Linear beta schedule.
    """

    def __init__(self, beta_start: float, beta_end: float, num_steps: int):
        """
        :param float beta_start: Starting beta value.
        :param float beta_end: Ending beta value.
        :param int num_steps: Number of steps in the diffusion process.
        """

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps

    def get_betas(self) -> list[float]:
        """
        Get the beta values for the diffusion process.

        :return list[float]: Beta values
        """

        return [
            np.clip(
                self.beta_start + (self.beta_end - self.beta_start) * i / (self.num_steps - 1),
                self.BETA_MIN,
                self.BETA_MAX,
            )
            for i in range(self.num_steps)
        ]


class CosineBetaSchedule(IBetaSchedule):
    """
    Cosine beta schedule. Based on:
    "Nichol, A. Q., & Dhariwal, P. (2021, July). Improved denoising diffusion probabilistic models.
    In International conference on machine learning (pp. 8162-8171). PMLR."
    """

    def __init__(self, offset: float, num_steps: int):
        """
        :param float offset: Offset for the cosine function.
        """

        self.offset = offset
        self.num_steps = num_steps
        self.f0 = self._f(0)

    def _f(self, step: int) -> float:
        """
        Helper function for the cosine beta schedule, according to Nichol et al.

        :param int step: Step of the diffusion process.
        :return float: Value of the function.
        """

        return np.cos((step / self.num_steps + self.offset) / (1 + self.offset) * np.pi / 2) ** 2

    def get_betas(self) -> list[float]:
        """
        Get the beta values for the diffusion process.

        :param int num_steps: Number of steps in the diffusion process.
        :return list[float]: Beta values
        """

        alphas = [self._f(i) / self.f0 for i in range(self.num_steps + 1)]
        return [np.clip(1 - alphas[i + 1] / alphas[i], self.BETA_MIN, self.BETA_MAX) for i in range(self.num_steps)]
