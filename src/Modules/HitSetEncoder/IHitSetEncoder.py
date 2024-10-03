from torch import nn, Tensor


class IHitSetEncoder(nn.Module):
    """
    Interface for hit set encoders.
    """

    def forward(self, x: Tensor, x_ind: Tensor) -> Tensor:
        """
        Forward pass of the encoder.

        :param Tensor x: The input tensor containing the hit point-cloud. Shape: `[num_hits, input_dim]`.
        :param Tensor x_ind: The batch index tensor. Shape: `[num_hits]`. See `CollisionEventLoader` for more information.
        :return Tensor: The output tensor encoding information about the hit point-cloud.
        """
        raise NotImplementedError
