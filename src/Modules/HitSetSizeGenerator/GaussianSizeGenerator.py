from torch import nn, Tensor, randn
from torch.nn import functional as F

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

    def forward(self, z: Tensor, _gt: Tensor, _gt_ind: Tensor) -> Tensor:
        """
        Forward pass of the Gaussian size generator.

        :param Tensor x: Input tensor. Shape `[input_dim]`
        :param Tensor gt: UNUSED - Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor gt_ind: UNUSED - Ground truth hit batch index tensor. Shape `[num_hits_next]`
        """
        return self.generate(z)

    def generate(self, z: Tensor) -> int:
        """
        Generate a size for the hit set.

        :param Tensor z: Input tensor. Shape `[input_dim]`
        """

        x = z
        for layer in self.layers:
            x = F.relu(layer(x))

        mus, sigmas = x[:, 0], x[:, 1]

        samples = mus + sigmas * randn(x.size(0), device=self.device)
        return F.relu(samples)
