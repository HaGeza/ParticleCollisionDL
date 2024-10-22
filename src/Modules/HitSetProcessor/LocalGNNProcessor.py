import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import convert_to_cartesian
from .IHitSetProcessor import IHitSetProcessor


class LocalGNNProcessor(IHitSetProcessor):
    """
    Class to process a hit set using a GNN with local connections.
    """

    def __init__(
        self,
        input_coord_system: CoordinateSystemEnum,
        k: int = 5,
        input_channels: list[int] = [0, 2],
        hidden_dim=16,
        output_dim=16,
        num_layers: int = 3,
        cdist_chunk_max_size: int = 2048,
        activation: callable = F.relu,
        device: str = "cpu",
    ):
        """
        :param CoordinateSystemEnum coordinate_system: The coordinate system of the inputs
        :param int k: The number of neighbors to make connections to for each node
        :param list[int] input_channels: The input channels to use
        :param int hidden_dim: The number of hidden dimensions
        :param int output_dim: The number of output dimensions
        :param int num_layers: The number of layers in the encoder.
        :param int cdist_chunk_max_size: The maximum size of the chunks to use for computing the cdist
        :param callable activation: The activation to apply
        :param str device: The device to use
        """
        super().__init__()

        self.input_coord_system = input_coord_system
        self.k = k

        self.layers = nn.ModuleList()

        self.input_channels = input_channels
        self.input_dim = len(input_channels)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.cdist_chunk_max_size = cdist_chunk_max_size
        self.activation = activation
        self.device = device

        self.layers.append(nn.Linear(self.input_dim * k + k, hidden_dim, device=device))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim * k + k, hidden_dim, device=device))
        self.layers.append(nn.Linear(hidden_dim * k + k, self.output_dim, device=device))

    def _create_local_graph_tensor(
        self, x: Tensor, neighbor_inds: Tensor, neighbor_dists: Tensor, channels: list[int] = None
    ) -> Tensor:
        """
        Create a tensor representing a local graph, with the node information first then the edge information

        :param Tensor x: The input tensor. Shape `[num_hits, input_dim]`.
        :param Tensor neighbor_inds: The indices of the K closest neighbors of each of the nodes in `x`.
            Shape `[num_hits, input_dim * k]`.
        :param Tensor neighbor_dists: The distances to each of the K closest neighbors of each of the nodes in `x`.
            Shape `[num_hits, k]`.
        :param list[int] channels: The list of channels to select from the input tensor
        :return Tensor: The tensor representing the local graph
        """
        if channels is not None:
            node_part = x[neighbor_inds][:, :, channels].view(x.shape[0], -1)
        else:
            node_part = x[neighbor_inds].view(x.shape[0], -1)

        return torch.cat([node_part, neighbor_dists], dim=1)

    def forward(self, x: Tensor, x_ind: Tensor) -> Tensor:
        x = convert_to_cartesian(x, self.input_coord_system)

        neighbor_inds = torch.empty([0, self.k], device=self.device, dtype=torch.long)
        neighbor_dists = torch.empty([0, self.k], device=self.device, dtype=torch.float32)
        x_padded = torch.empty([0, x.shape[1]], device=self.device, dtype=torch.float32)
        padding_inds = torch.tensor([], device=self.device, dtype=torch.long)

        if x_ind.size(0) == 0:
            return torch.empty([0, self.output_dim], device=self.device, dtype=torch.float32)

        B = x_ind.max().item() + 1
        b_start = 0
        for b in range(B):
            x_b = x[x_ind == b]
            if x_b.shape[0] < self.k + 1:
                padding_size = self.k + 1 - x_b.shape[0]
                padding_start = b_start + x_b.shape[0]
                padding_inds = torch.cat(
                    [padding_inds, padding_start + torch.arange(padding_size, device=self.device)], dim=0
                )
                x_b = torch.cat([x_b, torch.zeros([padding_size, x_b.shape[1]], device=self.device)], dim=0)
            x_padded = torch.cat([x_padded, x_b], dim=0)

            c_start = b_start
            cdist_chunks = x_b.shape[0] // self.cdist_chunk_max_size + 1
            c_size = x_b.shape[0] // cdist_chunks + 1

            for _ in range(cdist_chunks):
                c_end = min(c_start + c_size, b_start + x_b.shape[0])
                dists = torch.cdist(x_b[c_start - b_start : c_end - b_start], x_b, p=2)

                topk_dists, topk_inds = torch.topk(dists, self.k + 1, dim=1, largest=False, sorted=True)
                neighbor_inds = torch.cat([neighbor_inds, topk_inds[:, 1:] + b_start], dim=0)
                neighbor_dists = torch.cat([neighbor_dists, topk_dists[:, 1:]], dim=0)
                c_start = c_end
            b_start = c_end

        x = self._create_local_graph_tensor(x_padded, neighbor_inds, neighbor_dists, self.input_channels)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self._create_local_graph_tensor(x, neighbor_inds, neighbor_dists)
        x = self.layers[-1](x)

        mask = torch.ones(x.shape[0], device=self.device, dtype=torch.bool)
        mask[padding_inds] = False
        return x[mask]
