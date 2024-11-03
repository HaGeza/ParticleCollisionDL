import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.Util import CoordinateSystemEnum, convert_to_cartesian, get_coord_differences
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
        :param extra_input_dim: The dimension of the extra input to add to the input tensor.
            The extra input is not used for message passing.
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
        self.extra_input_dim = extra_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim

        self.cdist_chunk_max_size = cdist_chunk_max_size
        self.activation = activation
        self.device = device

        def get_linear_layer_dim(in_dim: int, extra_input: bool = False) -> int:
            return in_dim * (k + 1) + k * 3 + self.extra_input_dim * extra_input

        if num_layers > 1:
            self.layers.append(nn.Linear(get_linear_layer_dim(self.input_dim, True), hidden_dim, device=device))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(get_linear_layer_dim(hidden_dim), hidden_dim, device=device))
            self.layers.append(nn.Linear(get_linear_layer_dim(hidden_dim), self.output_dim, device=device))
        elif num_layers == 1:
            self.layers.append(nn.Linear(get_linear_layer_dim(self.input_dim, True), self.output_dim, device=device))
        else:
            raise ValueError("Number of layers must be at least 1")

    def _create_local_graph_tensor(
        self,
        x: Tensor,
        neighbor_inds: Tensor,
        neighbor_diffs: Tensor,
        encodings: Tensor | None = None,
        channels: list[int] = None,
    ) -> Tensor:
        """
        Create a tensor representing a local graph, with the node information first then the edge information

        :param Tensor x: The input tensor. Shape `[num_hits, input_dim]`.
        :param Tensor neighbor_inds: The indices of the K closest neighbors of each of the nodes in `x`.
            Shape `[num_hits, input_dim * k]`.
        :param Tensor neighbor_diffs: The attributes of the edges connecting the nodes to their neighbors.
            Shape `[num_hits, k]` or `[num_hits, k, num_neighbor_diffs]`.
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

        # Add attributes
        return torch.cat([graph_tensor, neighbor_diffs.view(neighbor_diffs.size(0), -1)], dim=1)

    NEIGHBOR_INDS = "neighbor_inds"
    neighbor_diffS = "neighbor_diffs"

    def get_top_k_neighbors(self, x: Tensor, k: int = 0) -> tuple[Tensor, Tensor]:
        """
        Get the indices and coordinate differences of the K nearest neighbors
        of each point in the input tensor.

        :param Tensor x: The input tensor. Shape `[num_hits, 3]`.
        :param int k: The number of neighbors to get. If 0 or less, use `self.k`.
        :return tuple[Tensor, Tensor]: The indices of the neighbors and the coordinate differences.
        """

        k = k if k > 0 else self.k
        x_cart = convert_to_cartesian(x, self.input_coord_system, theta_normalized=True)

        c_start = 0
        cdist_chunks = x.shape[0] // self.cdist_chunk_max_size + 1
        c_size = x.shape[0] // cdist_chunks + 1

        neighbor_inds = torch.empty([0, k], device=self.device, dtype=torch.long)
        neighbor_diffs = torch.empty([0, k, 3], device=self.device, dtype=torch.float32)

        for _ in range(cdist_chunks):
            c_end = min(c_start + c_size, x.shape[0])

            with torch.no_grad():
                dists = torch.cdist(x_cart[c_start:c_end], x_cart, p=2)
                topk_inds = torch.topk(dists, k + 1, dim=1, largest=False, sorted=True)[1][:, 1:]
                neighbor_inds = torch.cat([neighbor_inds, topk_inds], dim=0)

            diffs = get_coord_differences(
                x[c_start:c_end], x[topk_inds].transpose(0, 1), self.input_coord_system, theta_normalized=True
            ).transpose(0, 1)
            neighbor_diffs = torch.cat([neighbor_diffs, diffs], dim=0)

            c_start = c_end

        return neighbor_inds, neighbor_diffs

    def forward(self, x: Tensor, x_ind: Tensor, **kwargs) -> Tensor:
        """
        This method overrides :meth:`IHitSetProcessor.forward`.

        Additional keyword arguments:
        - `encodings_for_ddpm`: The output of the encoder. Used for implementing
            specialized logic in the denoising processors of the DDPM.
        - `neighbor_inds`: Optional precomputed graph connections. Shape `[num_hits, k]`.
            If not provided, the connections are computed using k-nearest neighbors.
        - `neighbor_diffs`: Optional precomputed graph attributes. Shape `[num_hits, k, num_neighbor_diffs]`.
            If not provided, the coordinate differences are used as edge attributes.
        """
        # Get nearest neighbors in case they have been precomputed
        neighbor_inds = kwargs.get(self.NEIGHBOR_INDS)
        neighbor_diffs = kwargs.get(self.neighbor_diffS)

        compute_neighbors = False
        if neighbor_inds is None or neighbor_diffs is None:
            neighbor_inds = torch.empty([0, self.k], device=self.device, dtype=torch.long)
            neighbor_diffs = torch.empty([0, self.k, 3], device=self.device, dtype=torch.float32)
            compute_neighbors = True

        if x_ind.size(0) == 0:
            return torch.empty([0, self.output_dim], device=self.device, dtype=torch.float32)

        # Get encodings in case of DDPM denoising
        encodings = kwargs.get(self.ENCODING_FOR_DDPM)

        # Nearest neighbors have not been precomputed => calculate distance matrix
        if compute_neighbors:
            b_start = 0
            for b in range(x_ind.max().item() + 1):
                x_b = x[x_ind == b]
                neighbor_inds_b, neighbor_diffs_b = self.get_top_k_neighbors(x_b)
                neighbor_inds = torch.cat([neighbor_inds, neighbor_inds_b + b_start], dim=0)
                neighbor_diffs = torch.cat([neighbor_diffs, neighbor_diffs_b], dim=0)
                b_start = neighbor_inds.size(0)

        # Neural network forward pass
        x = self._create_local_graph_tensor(x, neighbor_inds, neighbor_diffs, encodings, self.input_channels)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self._create_local_graph_tensor(x, neighbor_inds, neighbor_diffs)
        return self.layers[-1](x)

    def to(self, device, *args, **kwargs):
        self.device = device
        self.layers.to(device)
        return super().to(device, *args, **kwargs)
