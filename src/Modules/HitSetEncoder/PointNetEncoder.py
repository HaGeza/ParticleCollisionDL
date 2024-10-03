import torch
from torch import nn, Tensor
from torch_scatter import scatter_max
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
        :param num_layers: The number of layers in the encoder.
        :param device: The device to run the encoder on.
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

    def forward(self, x: Tensor, x_ind: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        :param Tensor x: The input tensor containing the hit point-cloud. Shape: `[num_hits, input_dim]`.
        :param Tensor x_ind: The batch index tensor. Shape: `[num_hits]`.
        :return Tensor: The output tensor encoding information about the hit point-cloud.
        """
        for layer in self.layers:
            x = F.relu(layer(x))

        out = torch.zeros(x_ind.max().item() + 1, x.size(1), device=x.device)
        out, _ = scatter_max(x, x_ind, dim=0, out=out)

        return out
