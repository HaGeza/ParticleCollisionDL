from torch import Tensor, nn
from torch.nn import functional as F

from .IHitSetProcessor import IHitSetProcessor


class PointNetProcessor(IHitSetProcessor):
    """
    Class to process a hit set using a PointNet.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 16,
        output_dim: int = 16,
        num_layers: int = 3,
        activation: callable = F.relu,
        device: str = "cpu",
    ):
        """
        Constructor for the PointNet encoder.

        :param int input_dim: The dimension of the input.
        :param int hidden_dim: The dimension of the hidden layer.
        :param int output_dim: The dimension of the output.
        :param int num_layers: The number of layers in the encoder.
        :param str device: The device to run the encoder on.
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.device = device

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim, device=device))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))
        self.layers.append(nn.Linear(hidden_dim, output_dim, device=device))

    def forward(self, x: Tensor) -> Tensor:
        """
        Process a hit set.

        :param Tensor int: The hit set to process. Shape `[num_hits, hit_dim]`.
        """

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
