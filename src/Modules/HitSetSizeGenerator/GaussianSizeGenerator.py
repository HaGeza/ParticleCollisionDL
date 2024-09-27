from torch import nn, Tensor, randn

from .IHitSetSizeGenerator import IHitSetSizeGenerator


class GaussianSizeGenerator(IHitSetSizeGenerator):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 2, num_layers: int = 2, device: str = "cpu"):
        super().__init__()

        self.device = device
        self.to(device)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim, device=device))
        for _ in range(1, num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, device=device))

        self.layers.append(nn.Linear(hidden_dim, 2, device=device))

    def forward(self, x: Tensor, _gt: Tensor):
        """
        Forward pass of the Gaussian size generator.

        :param Tensor x: Input tensor. Shape `[input_dim]`
        """

        for layer in self.layers:
            x = layer(x)

        mu, sigma = x

        return mu + sigma * randn(1, device=self.device)
