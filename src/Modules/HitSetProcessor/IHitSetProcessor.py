from torch import nn, Tensor


class IHitSetProcessor(nn.Module):
    """
    Interface for a class that processes a hit set, encoding some information
    into each hit in the set.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Process a hit set.

        :param Tensor int: The hit set to process. Shape `[num_hits, hit_dim]`.
        """

        raise NotImplementedError
