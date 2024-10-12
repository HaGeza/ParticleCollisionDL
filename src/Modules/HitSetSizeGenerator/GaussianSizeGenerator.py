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
        hidden_dim: int = 2,
        num_layers: int = 2,
        num_size_samples: int = 1,
        device: str = "cpu",
        size_scaler: int = 10000,
    ):
        """
        Constructor for the Gaussian size generator.

        :param int input_dim: Input dimension of the of the NN.
        :param int hidden_dim: Hidden dimension of the NN.
        :param int num_layers: Number of layers in the NN.
        :param int num_size_samples: Number of size samples to generate.
        :param str device: Device to run the NN on.
        :param int size_scaler: Multiplier for the predicted size.
        """

        super().__init__()

        self.device = device
        self.to(device)

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
        """
        Forward pass of the Gaussian size generator.

        :param Tensor z: Input tensor encoding the input hit-set. Shape `[batch_num, input_dim]`
        :param Tensor _gt: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor _gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :return: Generated hit set size tensor. Shape `[batch_num]`
        """

        return self.generate(z)

    def generate(self, z: Tensor) -> int:
        """
        Generate a size for the hit set.

        :param Tensor z: Input tensor encoding the input hit-set. Shape `[batch_num, input_dim]`
        :return: Generated hit set size tensor. Shape `[batch_num]`
        """

        x = z
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        mus, log_vars = x[:, : self.num_size_samples], x[:, self.num_size_samples :]

        samples = mus + torch.exp(0.5 * log_vars) * torch.randn_like(log_vars, device=self.device)
        return samples * self.size_scaler
