from torch import nn, Tensor
import torch


class IHitSetGenerator(nn.Module):
    """
    Interface for hit set generators.
    """

    def forward(
        self,
        z: Tensor,
        gt: Tensor,
        pred_ind: Tensor,
        gt_ind: Tensor,
        size: Tensor,
        initial_pred: Tensor = torch.tensor([]),
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the adjusting set generator.

        :param Tensor z: Encoded input hit set. Shape `[encoding_dim]`
        :param Tensor gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits_pred]`.
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`.
        :param Tensor size: Size of the generated hit point-cloud. Shape `[num_batches]`
            or `[num_batches, num_parts_next]`.
        :param Tensor initial_pred: Initial prediction. Shape `[num_hits_next, hit_dim]`
            or empty tensor if no initial prediction is available.
        :return: Generated hit set (Shape `[sum(size), hit_dim]`), and the loss
        """

        raise NotImplementedError

    def generate(self, z: Tensor, size: Tensor, initial_pred: Tensor = torch.tensor([])) -> Tensor:
        """
        Generate a hit set.

        :param Tensor z: Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :param Tensor initial_pred: Initial prediction. Shape `[num_hits_next, hit_dim]`
            or empty tensor if no initial prediction is available.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        raise NotImplementedError
