from torch import nn, Tensor


class IHitSetGenerator(nn.Module):
    """
    Interface for hit set generators.
    """

    def forward(self, x: Tensor, gt: Tensor, gt_ind: Tensor, size: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass of the hit set generator.

        :param Tensor _x: Input tensor. Shape `[encoding_dim]`
        :param Tensor _gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor _gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set (Shape `[sum(size), hit_dim]`), or a tuple of the generated hit set and the loss
        """

        raise NotImplementedError

    def calc_loss(self, pred_tensor: Tensor, gt_tensor: Tensor, pred_ind: Tensor, gt_ind: Tensor) -> Tensor:
        """
        Forward pass of the adjusting set generator.

        :param Tensor _x: Input tensor. Shape `[encoding_dim]`
        :param Tensor _gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor _gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError

    def generate(self, x: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor _x: (UNUSED) Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError
