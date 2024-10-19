import torch
from torch import Tensor


PI = torch.tensor(torch.pi)


def log_standard_normal(x) -> torch.Tensor:
    """
    Compute log probability of standard normal distribution.

    :param Tensor x: Input tensor.
    :return: Log probability of standard normal distribution
    """

    return -0.5 * torch.log(2.0 * PI) - 0.5 * x**2.0


def log_normal_diag(x, mu, log_var) -> Tensor:
    """
    Compute log probability of normal distribution with diagonal covariance matrix.

    :param Tensor x: Input tensor.
    :param Tensor mu: Mean of the normal distribution.
    :param Tensor log_var: Logarithm of the variance of the normal distribution.
    :return: Log probability of normal distribution
    """

    return -0.5 * torch.log(2.0 * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu) ** 2.0
