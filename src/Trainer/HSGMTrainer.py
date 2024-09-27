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

    def train(self, data_loader: CollisionEventLoader, epochs: int):
        self.model.train()

        if self.model.device != self.device:
            self.model.to(self.device)

        for _epoch in range(epochs):
            for hits_tensor_list, batch_index_list in data_loader:
                in_tensor = hits_tensor_list[0]
                in_batch_index = batch_index_list[0].detach()
                for t in range(1, self.model.time_step.get_num_time_steps()):
                    gt_tensor = hits_tensor_list[t]
                    gt_batch_index = batch_index_list[t].detach()
                    with torch.no_grad():
                        _, gt_size = torch.unique(gt_batch_index, return_counts=True)
                        gt_size = gt_size.float().to(self.device)

                    self.optimizer.zero_grad()

                    pred_size, pred_tensor = self.model(in_tensor, gt_tensor, in_batch_index, gt_batch_index, t)
                    loss = self.model.calc_loss(
                        pred_size, pred_tensor, gt_size, gt_tensor, gt_batch_index, t, self.size_loss_weight
                    )

                    loss.backward()
                    self.optimizer.step()

                    in_tensor = gt_tensor
                    in_batch_index = gt_batch_index

            self.scheduler.step()
