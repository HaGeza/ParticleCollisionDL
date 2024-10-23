from torch import Tensor, nn
import torch
from torch.nn import functional as F

from .IHitSetProcessor import IHitSetProcessor


class PointNetProcessor(IHitSetProcessor):
    """
    Class to process a hit set using a PointNet.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        input_channels: list[int] | None = None,
        extra_input_dim: int = 0,
        hidden_dim: int = 16,
        output_dim: int | None = None,
        num_layers: int = 3,
        activation: callable = F.relu,
        device: str = "cpu",
    ):
        """
        Constructor for the PointNet encoder.

        :param int input_dim: The dimension of the input. if `None`, set equal to `len(input_channels)`.
            If `input_channels` is None or empty and `input_dim` is None, raise an error.
        :param list[int] input_channels: The channels to use from the input.
        :param int extra_input_dim: The dimension of the extra input.
        :param int hidden_dim: The dimension of the hidden layer.
        :param int output_dim: The dimension of the output. if `None`, set equal to `hidden_dim`.
        :param int num_layers: The number of layers in the encoder.
        :param callable activation: The activation to apply
        :param str device: The device to run the encoder on.
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_dim, self.input_channels = self._get_input_dim_and_channels(input_dim, input_channels)
        self.extra_input_dim = extra_input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.activation = activation
        self.device = device

        self.layers = nn.ModuleList()

        if num_layers > 1:
            self.layers.append(nn.Linear(self.input_dim + extra_input_dim, hidden_dim, device=device))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim + extra_input_dim, hidden_dim, device=device))
            self.layers.append(nn.Linear(hidden_dim + extra_input_dim, self.output_dim, device=device))
        elif num_layers == 1:
            self.layers.append(nn.Linear(self.input_dim + extra_input_dim, self.output_dim, device=device))
        else:
            raise ValueError("Number of layers must be at least 1.")

    def forward(self, x: Tensor, x_ind: Tensor, **kwargs) -> Tensor:
        """
        This method overrides :meth:`IHitSetProcessor.forward`.

        Additional keyword arguments:
        - `encodings_for_ddpm`: The output of the encoder. If provided, the
            encodings are added to the input and intermediate tensors.
        """

        x = x[:, self.input_channels]

        encodings = kwargs.get(self.ENCODING_FOR_DDPM)
        if encodings is not None:
            x = torch.cat([x, encodings], dim=1)

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            if encodings is not None:
                x = torch.cat([x, encodings], dim=1)
        return self.layers[-1](x)
