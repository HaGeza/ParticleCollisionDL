from torch import nn, Tensor
import torch
from torch.nn import functional as F

from .IHitSetSizeGenerator import IHitSetSizeGenerator


class GaussianSizeGenerator(IHitSetSizeGenerator):
    """
    Hit set size generator that generates sizes from a Gaussian distribution, with parameters learned
    by a simple neural network.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 16,
        num_layers: int = 2,
        num_size_samples: int = 1,
        size_scaler: int = 10000,
        device: str = "cpu",
    ):
        """
        Constructor for the Gaussian size generator.

        :param int input_dim: Input dimension of the NN.
        :param int hidden_dim: Hidden dimension of the NN.
        :param int num_layers: Number of layers in the NN.
        :param int num_size_samples: Number of size samples to generate.
        :param int size_scaler: Multiplier for the predicted size.
        :param str device: Device to run the NN on.
        """

        super().__init__()

        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_size_samples = num_size_samples
        self.size_scaler = size_scaler

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim, device=device))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
        self.layers.append(nn.Linear(hidden_dim, 2 * num_size_samples, device=device))

    def forward(self, z: Tensor, _gt: Tensor, _gt_ind: Tensor) -> Tensor:
        return self.generate(z)

    def generate(self, z: Tensor) -> int:
        x = z
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        mus, log_vars = x[:, : self.num_size_samples], x[:, self.num_size_samples :]

        samples = mus + torch.exp(0.5 * log_vars) * torch.randn_like(log_vars, device=self.device)
        return samples * self.size_scaler

    def calc_loss(self, pred_size: Tensor, gt_size: Tensor) -> Tensor:
        return F.mse_loss(pred_size / self.size_scaler, gt_size / self.size_scaler)
