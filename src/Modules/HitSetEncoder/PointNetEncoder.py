import torch
from torch import nn
import torch.nn.functional as F

from .IHitSetEncoder import IHitSetEncoder


class PointNetEncoder(IHitSetEncoder):
    """
    PointNet hit set encoder.
    """

    def __init__(
        self, input_dim: int = 3, hidden_dim: int = 16, output_dim: int = 16, num_layers: int = 3, device: str = "cpu"
    ):
        """
        Constructor for the PointNet encoder.
        :param input_dim: The dimension of the input.
        :param hidden_dim: The dimension of the hidden layer.
        :param output_dim: The dimension of the output.
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim, device=device))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
        self.layers.append(nn.Linear(hidden_dim, output_dim, device=device))

    def forward(self, x):
        """
        Forward pass of the encoder.

        :param Tensor x: The input tensor containing the hit point-cloud. Shape: `[num_hits, input_dim]`.
        :return Tensor: The output tensor encoding information about the hit point-cloud.
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        x, _ = torch.max(x, dim=0)

        return x
