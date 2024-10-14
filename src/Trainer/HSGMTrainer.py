import csv
from datetime import datetime
import json
import os
import random
import re
import string

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler, Optimizer

from src.Util import MODELS_DIR, RESULTS_DIR
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Data.CollisionEventLoader import CollisionEventLoader


class Trainer:
    """
    Trainer class for training a `HitSetGenerativeModel`s
    """

    MODEL_KEY_LENGTH = 8

    @staticmethod
    def _generate_model_name() -> str:
        """
        Generate a random model name

        :return str: The model name
        """

        return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(Trainer.MODEL_KEY_LENGTH))

    def _find_existing_models(self) -> list[str]:
        """
        Find existing models in the models directory

        :return list[str]: The list of existing models
        """

        files = os.listdir(self.models_path)
        pattern = re.compile(f"^[a-z0-9]{{{Trainer.MODEL_KEY_LENGTH}}}$")
        return [f for f in files if pattern.match(f)]

    def __init__(
        self,
        model: HitSetGenerativeModel,
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        device: str = "cpu",
        size_loss_weight: float = 0.25,
        models_path: str = MODELS_DIR,
        results_path: str = RESULTS_DIR,
    ):
        """
        :param HitSetGenerativeModel model: The model to train
        :param Optimizer optimizer: The optimizer to use
        :param lr_scheduler._LRScheduler scheduler: The learning rate scheduler to use
        :param str device: The device to use
        :param float size_loss_weight: The weight to give to the size loss
        :param str models_path: The path to save the models
        :param str results_path: The path to save the results
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.size_loss_weight = size_loss_weight
        self.models_path = models_path
        self.results_path = results_path

        if model.device != device:
            model.to(device)

    def train(self, data_loader: CollisionEventLoader, epochs: int, no_log: bool = False):
        """
        Train the model using the given data loader.

        :param CollisionEventLoader data_loader: The data loader to use
        :param int epochs: The number of epochs to train for
        :param bool no_log: Whether to log the training progress
        """

        if self.model.device != self.device:
            self.model.to(self.device)

        os.makedirs(self.models_path, exist_ok=True)

        model_name = self._generate_model_name()
        while model_name in self._find_existing_models():
            model_name = self._generate_model_name()

        model_dir = os.path.join(self.models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        result_dir = os.path.join(self.results_path, model_name)
        os.makedirs(result_dir, exist_ok=True)

        info_file = os.path.join(result_dir, "info.json")
        with open(info_file, "w") as f:
            model_info = self.model.get_info()
            training_info = {
                "model_name": model_name,
                "epochs": epochs,
                "batch_size": data_loader.batch_size,
                "device": str(self.device),
                "size_loss_weight": self.size_loss_weight,
                "optimizer_type": self.optimizer.__class__.__name__,
                "scheduler_type": self.scheduler.__class__.__name__,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "date": datetime.today().strftime("%Y:%m:%d_%H:%M:%S"),
            }
            json.dump({"model": model_info, "training": training_info}, f, indent=4)

        if not no_log:
            log_file = os.path.join(result_dir, "log.csv")
            with open(log_file, "w") as f:
                row = ["epoch", "event", "t", "loss"]

                if not self.model.use_shell_part_sizes:
                    row += [f"pred_size_{i}" for i in range(data_loader.batch_size)]
                    row += [f"gt_size_{i}" for i in range(data_loader.batch_size)]
                    num_size_preds = data_loader.batch_size
                else:
                    max_num_parts = np.max(
                        [
                            self.model.time_step.get_num_shell_parts(t)
                            for t in range(1, self.model.time_step.get_num_time_steps())
                        ]
                    )
                    row += [f"pred_size_{i}_{j}" for i in range(data_loader.batch_size) for j in range(max_num_parts)]
                    row += [f"gt_size_{i}_{j}" for i in range(data_loader.batch_size) for j in range(max_num_parts)]
                    num_size_preds = data_loader.batch_size * max_num_parts
                csv.writer(f).writerow(row)
            data_loader.return_event_ids = True

        min_loss = float("inf")

        self.model.train()

        # Main training loop
        for epoch in range(epochs):
            for entry in data_loader:
                hits_tensor_list, batch_index_list, event_id = entry

                in_tensor = hits_tensor_list[0]
                in_batch_index = batch_index_list[0].detach()

                loss_sum = 0
                for t in range(1, self.model.time_step.get_num_time_steps()):
                    gt_tensor = hits_tensor_list[t]
                    gt_batch_index = batch_index_list[t].detach()

                    # Calculate ground truth sizes for shells / shell-parts
                    with torch.no_grad():
                        if self.model.use_shell_part_sizes:
                            part_ids = self.model.time_step.assign_to_shell_parts(
                                gt_tensor, t, self.model.coordinate_system
                            )
                            num_parts = self.model.time_step.get_num_shell_parts(t)
                            batch_size = data_loader.batch_size

                            batch_part_ids = gt_batch_index * num_parts + part_ids
                            _, gt_size = torch.unique(batch_part_ids, return_counts=True)

                            gt_size = F.pad(gt_size, (0, num_parts * batch_size - gt_size.size(0)), value=0)
                            gt_size = gt_size.view(batch_size, num_parts)
                        else:
                            _, gt_size = torch.unique(gt_batch_index, return_counts=True)

                        gt_size = gt_size.float().to(self.device)

                    self.optimizer.zero_grad()

                    pred_size, _pred_tensor, loss = self.model(
                        in_tensor, gt_tensor, in_batch_index, gt_batch_index, gt_size, t
                    )

                    loss.backward()
                    self.optimizer.step()

                    in_tensor = gt_tensor
                    in_batch_index = gt_batch_index

                    loss_sum += loss.item()

                    if not no_log:
                        with open(log_file, "a") as f:
                            row = [epoch, event_id, t, loss.item()]
                            pred_size_flat = pred_size.view(-1).tolist()
                            padding = [""] * (num_size_preds - len(pred_size_flat))
                            row += pred_size_flat + padding + gt_size.view(-1).tolist()
                            csv.writer(f).writerow(row)

                self.scheduler.step()

                torch.save(self.model.state_dict(), os.path.join(model_dir, f"latest.pth"))

                if loss_sum < min_loss:
                    min_loss = loss_sum
                    torch.save(self.model.state_dict(), os.path.join(model_dir, "min_loss.pth"))
