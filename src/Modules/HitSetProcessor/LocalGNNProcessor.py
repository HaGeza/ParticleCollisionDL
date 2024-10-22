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
        input_dim: int | None = None,
        input_channels: list[int] | None = None,
        extra_input_dim: int = 0,
        hidden_dim: int = 16,
        output_dim: int | None = None,
        num_layers: int = 3,
        cdist_chunk_max_size: int = 2048,
        activation: callable = F.relu,
        device: str = "cpu",
    ):
        """
        :param CoordinateSystemEnum coordinate_system: The coordinate system of the inputs
        :param int k: The number of neighbors to make connections to for each node
        :param list[int] input_channels: The input channels to use
        :param int input_dim: The dimension of the extra input to add to the input tensor,
            that does not get used for message passing
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

        self.input_dim, self.input_channels = self._get_input_dim_and_channels(input_dim, input_channels)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.extra_input_dim = extra_input_dim

        self.cdist_chunk_max_size = cdist_chunk_max_size
        self.activation = activation
        self.device = device

        def get_linear_layer_dim(in_dim: int):
            return in_dim * (k + 1) + k + self.extra_input_dim

        if num_layers > 1:
            self.layers.append(nn.Linear(get_linear_layer_dim(self.input_dim), hidden_dim, device=device))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(get_linear_layer_dim(hidden_dim), hidden_dim, device=device))
            self.layers.append(nn.Linear(get_linear_layer_dim(hidden_dim), self.output_dim, device=device))
        elif num_layers == 1:
            self.layers.append(nn.Linear(get_linear_layer_dim(self.input_dim), self.output_dim, device=device))
        else:
            raise ValueError("Number of layers must be at least 1")

    def _create_local_graph_tensor(
        self,
        x: Tensor,
        neighbor_inds: Tensor,
        neighbor_dists: Tensor,
        encodings: Tensor | None,
        channels: list[int] = None,
    ) -> Tensor:
        """
        Create a tensor representing a local graph, with the node information first then the edge information

        :param Tensor x: The input tensor. Shape `[num_hits, input_dim]`.
        :param Tensor neighbor_inds: The indices of the K closest neighbors of each of the nodes in `x`.
            Shape `[num_hits, input_dim * k]`.
        :param Tensor neighbor_dists: The distances to each of the K closest neighbors of each of the nodes in `x`.
            Shape `[num_hits, k]`.
        :param Tensor encodings: The encodings to add to the input tensor. Shape `[num_hits, encoding_dim]`.
            If `None`, do not add encodings.
        :param list[int] channels: The list of channels to select from the input tensor
        :return Tensor: The tensor representing the local graph
        """
        # Add node and neighbor information
        if channels is not None:
            graph_tensor = x[neighbor_inds][:, :, channels].view(x.shape[0], -1)
            graph_tensor = torch.cat([x[:, channels], graph_tensor], dim=1)
        else:
            graph_tensor = x[neighbor_inds].view(x.shape[0], -1)
            graph_tensor = torch.cat([x, graph_tensor], dim=1)

        # Add encodings
        if encodings is not None:
            graph_tensor = torch.cat([graph_tensor, encodings], dim=1)

        # Add distances and return
        return torch.cat([graph_tensor, neighbor_dists], dim=1)

    def forward(self, x: Tensor, x_ind: Tensor, **kwargs) -> Tensor:
        x = convert_to_cartesian(x, self.input_coord_system)

        neighbor_inds = torch.empty([0, self.k], device=self.device, dtype=torch.long)
        neighbor_dists = torch.empty([0, self.k], device=self.device, dtype=torch.float32)
        x_padded = torch.empty([0, x.shape[1]], device=self.device, dtype=torch.float32)
        padding_inds = torch.tensor([], device=self.device, dtype=torch.long)

        if x_ind.size(0) == 0:
            return torch.empty([0, self.output_dim], device=self.device, dtype=torch.float32)

        encodings = kwargs.get(self.ENCODING_FOR_DDPM)
        if encodings is not None:
            encodings_padded = torch.empty([0, encodings.shape[1]], device=self.device, dtype=torch.float32)
        else:
            encodings_padded = None

        B = x_ind.max().item() + 1
        b_start = 0
        for b in range(B):
            mask = x_ind == b
            x_b = x[mask]
            if encodings is not None:
                encodings_b = encodings[mask]

            if x_b.shape[0] < self.k + 1:
                padding_size = self.k + 1 - x_b.shape[0]
                padding_start = b_start + x_b.shape[0]
                padding_inds = torch.cat(
                    [padding_inds, padding_start + torch.arange(padding_size, device=self.device)], dim=0
                )
                x_b = torch.cat([x_b, torch.zeros([padding_size, x_b.shape[1]], device=self.device)], dim=0)
                if encodings is not None:
                    if encodings_b.shape[0] == 0:
                        encodings_b = torch.zeros([padding_size, encodings.shape[1]], device=self.device)
                    else:
                        encodings_b = torch.cat([encodings_b, encodings_b[-1].repeat(padding_size, 1)], dim=0)

            x_padded = torch.cat([x_padded, x_b], dim=0)
            if encodings is not None:
                encodings_padded = torch.cat([encodings_padded, encodings_b], dim=0)

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

        x = self._create_local_graph_tensor(
            x_padded, neighbor_inds, neighbor_dists, encodings_padded, self.input_channels
        )
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self._create_local_graph_tensor(x, neighbor_inds, neighbor_dists, encodings_padded)
        x = self.layers[-1](x)

        mask = torch.ones(x.shape[0], device=self.device, dtype=torch.bool)
        mask[padding_inds] = False
        return x[mask]
