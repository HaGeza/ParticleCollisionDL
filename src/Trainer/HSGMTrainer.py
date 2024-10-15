import csv
from datetime import datetime
import json
import os
import random
import re
import string

import numpy as np
from scipy.spatial.distance import directed_hausdorff
import torch
from torch import Tensor
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

    def _get_gt_size(self, gt_tensor: Tensor, gt_batch_index: Tensor, t: int) -> Tensor:
        """
        Calculate ground truth sizes for shells / shell-parts

        :param Tensor gt_tensor: The ground truth hit tensor
        :param Tensor gt_batch_index: The ground truth batch index tensor
        :param int t: The time step
        :return Tensor: The ground truth sizes. Shape `[num_batches]` or `[num_batches, num_parts]`
            if `self.model.use_shell_part_sizes` is `True`
        """
        with torch.no_grad():
            if self.model.use_shell_part_sizes:
                part_ids = self.model.time_step.assign_to_shell_parts(gt_tensor, t, self.model.coordinate_system)
                num_parts = self.model.time_step.get_num_shell_parts(t)
                batch_size = torch.max(gt_batch_index) + 1

                batch_part_ids = gt_batch_index * num_parts + part_ids
                _, gt_size = torch.unique(batch_part_ids, return_counts=True)

                gt_size = F.pad(gt_size, (0, num_parts * batch_size - gt_size.size(0)), value=0)
                gt_size = gt_size.view(batch_size, num_parts)
            else:
                _, gt_size = torch.unique(gt_batch_index, return_counts=True)

            gt_size = gt_size.float().to(self.device)

        return gt_size

    def _get_hausdorff_distance(self, pred_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        """
        Calculate the Hausdorff distance between the predicted and ground truth hit sets.

        :param Tensor pred_tensor: The predicted hit tensor
        :param Tensor gt_tensor: The ground truth hit tensor
        :return Tensor: The Hausdorff distance
        """

        pred_hits = pred_tensor.cpu().detach().numpy()
        gt_hits = gt_tensor.cpu().detach().numpy()

        pred_to_gt = directed_hausdorff(pred_hits, gt_hits)[0]
        gt_to_pred = directed_hausdorff(gt_hits, pred_hits)[0]

        return torch.tensor(max(pred_to_gt, gt_to_pred), dtype=torch.float32, device=self.device)

    def evaluate(self, data_iter: callable) -> tuple[list[float], list[float]]:
        """
        Evaluate the model using the given data iterator.

        :param callable data_iter: The data iterator; Should be `CollisionEventLoader.iter_train`
            or `CollisionEventLoader.iter_val`
        :return tuple[list[float], list[float]]: The mean Hausdorff distances and mean size MSEs
            per time step
        """

        T = self.model.time_step.get_num_time_steps()
        hds = [0] * (T - 1)
        mses = [0] * (T - 1)
        num_entries = 0

        for entry in data_iter:
            hits_tensor_list, batch_index_list, _ = entry

            in_tensor = hits_tensor_list[0]
            in_batch_index = batch_index_list[0].detach()

            for t in range(1, T):
                gt_tensor = hits_tensor_list[t]
                gt_batch_index = batch_index_list[t].detach()
                gt_size = self._get_gt_size(gt_tensor, gt_batch_index, t)

                pred_size, pred_tensor = self.model.generate(in_tensor, in_batch_index, t)

                hds[t - 1] += self._get_hausdorff_distance(pred_tensor, gt_tensor).item()
                mses[t - 1] += F.mse_loss(pred_size, gt_size).item()
                num_entries += gt_size.size(0)

        return [hd / num_entries for hd in hds], [mse / num_entries for mse in mses]

    def train_and_eval(
        self,
        epochs: int,
        data_loader: CollisionEventLoader,
        no_log: bool = False,
    ):
        """
        Train and evaluate the model using the given data loader.

        :param int epochs: The number of epochs to train for
        :param CollisionEventLoader data_loader: The data loader to use
        :param bool no_log: Whether to log the training progress
        """

        if self.model.device != self.device:
            self.model.to(self.device)

        T = self.model.time_step.get_num_time_steps()

        # Set up directories
        os.makedirs(self.models_path, exist_ok=True)

        model_name = self._generate_model_name()
        while model_name in self._find_existing_models():
            model_name = self._generate_model_name()

        model_dir = os.path.join(self.models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        result_dir = os.path.join(self.results_path, model_name)
        os.makedirs(result_dir, exist_ok=True)

        # Set up info file
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

        # Create log files
        if not no_log:
            # Create log of training progress
            train_log = os.path.join(result_dir, "train_log.csv")
            with open(train_log, "w") as f:
                row = ["epoch", "t"]
                row += [f"event_{i}" for i in range(data_loader.batch_size)]
                row += ["size_loss", "set_loss", "loss"]

                if not self.model.use_shell_part_sizes:
                    row += [f"pred_size_{i}" for i in range(data_loader.batch_size)]
                    row += [f"gt_size_{i}" for i in range(data_loader.batch_size)]
                    num_size_preds = data_loader.batch_size
                else:
                    max_num_parts = np.max([self.model.time_step.get_num_shell_parts(t) for t in range(1, T)])
                    row += [f"pred_size_{i}_{j}" for i in range(data_loader.batch_size) for j in range(max_num_parts)]
                    row += [f"gt_size_{i}_{j}" for i in range(data_loader.batch_size) for j in range(max_num_parts)]
                    num_size_preds = data_loader.batch_size * max_num_parts
                csv.writer(f).writerow(row)
            data_loader.return_event_ids = True

            # Create log of evaluation metrics
            eval_log = os.path.join(result_dir, "eval_log.csv")
            with open(eval_log, "w") as f:
                row = ["epoch", "loss"] + [
                    f"{m}_{s}_{t}" for t in range(1, T) for m in ["mse", "hd"] for s in ["train", "val"]
                ]
                csv.writer(f).writerow(row)

        # Main training loop
        min_loss = float("inf")
        self.model.train()

        for epoch in range(epochs):
            loss_mean = 0
            num_entries = 0

            for entry in data_loader.iter_train():
                hits_tensor_list, batch_index_list, event_ids = entry

                in_tensor = hits_tensor_list[0]
                in_batch_index = batch_index_list[0].detach()

                for t in range(1, T):
                    gt_tensor = hits_tensor_list[t]
                    gt_batch_index = batch_index_list[t].detach()
                    gt_size = self._get_gt_size(gt_tensor, gt_batch_index, t)

                    self.optimizer.zero_grad()

                    pred_size, pred_tensor, size_loss, set_loss = self.model(
                        in_tensor, gt_tensor, in_batch_index, gt_batch_index, gt_size, t
                    )

                    loss = size_loss * self.size_loss_weight + set_loss
                    loss.backward()
                    self.optimizer.step()

                    in_tensor = gt_tensor
                    in_batch_index = gt_batch_index

                    loss_mean += loss.item()
                    num_entries += gt_size.size(0)

                    if not no_log:
                        with open(train_log, "a") as f:
                            row = [epoch, t]
                            row += event_ids
                            row += [size_loss.item(), set_loss.item(), loss.item()]
                            pred_size_flat = pred_size.view(-1).tolist()
                            padding = [""] * (num_size_preds - len(pred_size_flat))
                            row += pred_size_flat + padding + gt_size.view(-1).tolist()
                            csv.writer(f).writerow(row)

                # end of batch
                self.scheduler.step()

            # end of epoch
            torch.save(self.model.state_dict(), os.path.join(model_dir, f"latest.pth"))

            if num_entries > 0:
                loss_mean = loss_mean / num_entries

            if loss_mean < min_loss:
                min_loss = loss_mean
                torch.save(self.model.state_dict(), os.path.join(model_dir, "min_loss.pth"))

            if not no_log:
                self.model.eval()

                hd_train, mse_train = self.evaluate(data_loader.iter_train())
                hd_val, mse_val = self.evaluate(data_loader.iter_val())

                with open(eval_log, "a") as f:
                    row = [epoch, loss_mean]
                    for t in range(T - 1):
                        row += [mse_train[t], mse_val[t], hd_train[t], hd_val[t]]
                    csv.writer(f).writerow(row)
