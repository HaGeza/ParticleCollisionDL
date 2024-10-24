from torch import nn, Tensor


class IHitSetProcessor(nn.Module):
    """
    Interface for a class that processes a hit set, encoding some information
    into each hit in the set.
    """

    @staticmethod
    def _get_input_dim_and_channels(input_dim: int | None, input_channels: list[int] | None) -> tuple[int, list[int]]:
        """
        Get the input dimension and channels. If `input_channels` is `None` or empty, set
        use all channels. If `input_dim` is `None` or 0, but `input_channels` is given
        set `input_dim` to the length of `input_channels`. Otherwise raise an error.

        :param int | None input_dim: The input dimension.
        :param list[int] | None input_channels: The input channels.
        :return tuple[int, list[int]]: The actual input dimension and channels.
        """

        act_input_channels = input_channels
        if input_channels is None or len(input_channels) == 0:
            if input_dim is None or input_dim == 0:
                raise ValueError(
                    "input_dim cannot be None or 0 if input_channels is None or empty. "
                    f"Got input_dim = {input_dim} and input_channels = {input_channels}."
                )
            act_input_channels = list(range(input_dim))
        act_input_dim = len(act_input_channels)

        return act_input_dim, act_input_channels

    ENCODING_FOR_DDPM = "encodings_for_ddpm"
    EXTRA_INPUT_DIM = "extra_input_dim"

    def forward(self, x: Tensor, x_ind: Tensor, **kwargs) -> Tensor:
        """
        Process a hit set.

        :param Tensor x: The hit set to process. Shape `[num_hits, hit_dim]`.
        :param Tensor x_ind: The batch index of each hit. Shape `[num_hits]`.
        :param kwargs: Additional arguments.
        - `encodings_for_ddpm`: The output of the encoder. Used for implementing specialized
        logic in the denoising processors of the DDPM.
        """
        raise NotImplementedError
