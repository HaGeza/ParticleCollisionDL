from torch import nn, Tensor


class IHitSetEncoder(nn.Module):
    """
    Interface for hit set encoders.
    """

    def forward(self, x: Tensor, x_ind: Tensor, batch_size: int = 0) -> Tensor:
        """
        Forward pass of the encoder.

        :param Tensor x: The input tensor containing the hit point-cloud. Shape: `[num_hits, input_dim]`.
        :param Tensor x_ind: The batch index tensor. Shape: `[num_hits]`.
            See `CollisionEventLoader` for more information.
        :param int batch_size: The batch size. If 0, the batch size is determined by the maximum value in `x_ind`.
        :return Tensor: The output tensor encoding information about the hit point-cloud.
        """
        raise NotImplementedError

    def get_loss(self) -> Tensor:
        """
        Get the loss of the encoder. The default implementation returns 0.

        :return Tensor: The loss of the encoder.
        """
        return 0.0
