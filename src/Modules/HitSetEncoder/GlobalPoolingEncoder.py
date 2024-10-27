import torch
from torch import Tensor

from src.Modules.HitSetProcessor import IHitSetProcessor
from src.Util.Distributions import log_normal_diag, log_standard_normal
from .IHitSetEncoder import IHitSetEncoder


class GlobalPoolingEncoder(IHitSetEncoder):
    """
    Global pooling encoder. Works by processing each point in the input and global max-pooling them.
    """

    def __init__(
        self, processor: IHitSetProcessor, num_levels: int = 1, variational: bool = False, device: str = "cpu"
    ):
        """
        Constructor for the global pooling encoder.

        :param IHitsSetProcessor processor: The processor to use before global pooling.
        :param num_levels: The number of levels to pool.
        :param bool variational: Whether the encoder is variational.
        :param str device: The device to run the encoder on.
        """
        super().__init__()

        self.processor = processor
        self.num_levels = num_levels
        self.variational = variational
        self.device = device

        self.kl_div = 0.0

    def forward(self, x: Tensor, x_ind: Tensor, batch_size: int = 0) -> Tensor:
        x = self.processor(x, x_ind)

        kl_div = torch.tensor(0.0, device=self.device)
        if self.variational:
            enc_dim = x.size(1) // 2
            mus, log_vars = x[:, :enc_dim], x[:, enc_dim:]
            x = mus + torch.exp(0.5 * log_vars) * torch.randn_like(log_vars, device=self.device)
            kl_div = log_standard_normal(x) - log_normal_diag(x, mus, log_vars).sum()

        out_size = x_ind.max().item() + 1 if batch_size == 0 else batch_size
        out = torch.full((out_size, x.size(1) * self.num_levels), -float("inf"), device=x.device)

        for i in range(out_size):
            batch = x[x_ind == i]
            if batch.size(0) > 0:
                out[i] = torch.topk(batch, self.num_levels, dim=0).values.view(-1)
            else:
                out[i] = torch.zeros_like(out[i])

        return out, kl_div.mean()
