from torch import nn, Tensor


class IHitSetSizeGenerator(nn.Module):
    """
    Interface for hit set size generators.
    """

    def forward(self, z: Tensor, gt: Tensor, gt_ind: Tensor) -> Tensor:
        """
        Forward pass of the hit set size generator.

        :param Tensor z: Input tensor encoding the input hit-set. Shape `[batch_num, input_dim]`
        :param Tensor gt: Ground truth hit tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :return: Generated hit set size tensor. Shape `[batch_num]`
        """

        raise NotImplementedError

    def generate(self, z: Tensor) -> Tensor:
        """
        Generate a size for the hit set.

        :param Tensor z: Input tensor encoding the input hit-set. Shape `[batch_num, input_dim]`
        :return: Generated hit set size tensor. Shape `[batch_num]` or `[batch_num, num_parts]`.
        """

        raise NotImplementedError

    def calc_loss(self, pred_size: Tensor, gt_size: Tensor) -> Tensor:
        """
        Calculate the loss of the predicted hit set size.

        :param Tensor pred_size: Predicted hit set size tensor. Shape `[batch_num]` or `[batch_num, num_parts]`.
        :param Tensor gt_size: Ground truth hit set size tensor. Shape `[batch_num]` or `[batch_num, num_parts]`.
        :return Tensor: The loss value.
        """

        raise NotImplementedError
