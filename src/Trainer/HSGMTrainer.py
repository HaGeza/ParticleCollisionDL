import torch
from torch.optim import Adam, lr_scheduler, Optimizer
from pandas.api.typing import DataFrameGroupBy

from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Data.CollisionEventLoader import CollisionEventLoader


class Trainer:
    """
    Trainer class for training a `HitSetGenerativeModel`
    """

    def __init__(
        self,
        model: HitSetGenerativeModel,
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        device: str = "cpu",
        size_loss_weight: float = 0.25,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.size_loss_weight = size_loss_weight

        if model.device != device:
            model.to(device)

    def _get_t_hit_tensor(grouped_hits: DataFrameGroupBy, t: int, device: str) -> torch.Tensor:
        df = grouped_hits.get_group(t)[["x", "y", "z"]].reset_index(drop=True)
        return torch.tensor(df.values, dtype=torch.float32, device=device)

    def train(self, data_loader: CollisionEventLoader, epochs: int):
        self.model.train()

        if self.model.device != self.device:
            self.model.to(self.device)

        for _epoch in range(epochs):
            for hits, _, _ in data_loader:
                grouped_hits = hits.groupby("t")

                input_tensor = Trainer._get_t_hit_tensor(grouped_hits, 0, self.device)
                for t in range(1, self.model.time_step.get_num_time_steps()):
                    gt_tensor = Trainer._get_t_hit_tensor(grouped_hits, t, self.device)

                    self.optimizer.zero_grad()

                    pred_size, pred_tensor = self.model(input_tensor, gt_tensor, t)
                    loss = self.model.calc_loss(pred_size, pred_tensor, gt_tensor, t, self.size_loss_weight)

                    loss.backward()
                    self.optimizer.step()

                    input_tensor = gt_tensor

            self.scheduler.step()
