import csv
from datetime import datetime
import json
import os
import random
import re
import string

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Util.Paths import RUNS_DIR


class TrainingRunIO:
    """
    Class for reading and writing training run artifacts, such as models, info dumps and logs
    """

    INFO_FILE = "info.json"
    TRAIN_LOG_FILE = "train.log"
    EVAL_LOG_FILE = "eval.log"
    LATEST_NAME = "latest"
    MIN_LOSS_NAME = "min_loss"

    TRAINING_SECTION = "training"
    MODEL_SECTION = "model"

    DATE_FIELD = "date"

    MODEL_FIELD = "model"
    OPTIMIZER_FIELD = "optimizer"
    SCHEDULER_FIELD = "scheduler"
    EPOCH_FIELD = "epoch"

    def __init__(
        self,
        model_id: str | None = None,
        no_train_log: bool = False,
        no_eval_log: bool = False,
        runs_dir: str = RUNS_DIR,
    ):
        """
        :param str model_id: The id of the model. If `None` or empty string, a random id will be generated.
        :param bool no_train_log: Whether to not create a training log
        :param bool no_eval_log: Whether to not create an evaluation log
        :param str runs_dir: The directory to save the results in
        """
        MODEL_KEY_LENGTH = 8

        self.runs_dir = runs_dir
        self.model_id = model_id
        self.no_train_log = no_train_log
        self.no_eval_log = no_eval_log

        if not os.path.exists(runs_dir):
            existing_runs = []
        else:
            files = os.listdir(runs_dir)
            pattern = re.compile(f"^[a-z0-9]{{{MODEL_KEY_LENGTH}}}$")
            existing_runs = [f for f in files if pattern.match(f)]

        if model_id is None or model_id == "":

            def _get_random_id() -> str:
                return "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(MODEL_KEY_LENGTH))

            self.model_id = _get_random_id()
            while self.model_id in existing_runs:
                self.model_id = _get_random_id()

            self.resume_from_checkpoint = False
        else:
            if model_id == self.LATEST_NAME:
                if not existing_runs:
                    raise ValueError("No existing runs to get the latest from")

                run_dates = []
                for run in existing_runs:
                    with open(os.path.join(runs_dir, run, self.INFO_FILE)) as f:
                        info = json.load(f)
                        run_dates.append((run, info[self.TRAINING_SECTION][self.DATE_FIELD]))

                self.model_id = max(run_dates, key=lambda x: x[1])[0]
            else:
                if model_id not in existing_runs:
                    raise ValueError(f"Model with id {model_id} does not exist")
                self.model_id = model_id

            self.resume_from_checkpoint = True

        self.dir = os.path.join(runs_dir, self.model_id)

    def setup(
        self,
        model: HitSetGenerativeModel,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        batch_size: int,
        epochs: int,
        size_loss_weight: float,
    ):
        """
        Set up training run directories and log files

        :param HitSetGenerativeModel model: The model used for training
        :param Optimizer optimizer: The optimizer used for training
        :param _LRScheduler scheduler: The scheduler used for training
        :param int batch_size: The batch size used for training
        :param int epochs: The number of training epochs
        :param float size_loss_weight: The weight to give to the size loss
        """

        B = batch_size
        T = model.time_step.get_num_time_steps()

        # Set up directories
        os.makedirs(self.dir, exist_ok=True)

        # Set up info file
        info_file = os.path.join(self.dir, TrainingRunIO.INFO_FILE)
        with open(info_file, "w") as f:
            model_info = model.information
            training_info = {
                "model_name": self.model_id,
                "epochs": epochs,
                "batch_size": B,
                "device": str(model.device),
                "size_loss_weight": size_loss_weight,
                "optimizer_type": optimizer.__class__.__name__,
                "scheduler_type": scheduler.__class__.__name__,
                "learning_rate": optimizer.param_groups[0]["lr"],
                self.DATE_FIELD: datetime.today().strftime("%Y:%m:%d_%H:%M:%S"),
            }
            json.dump({"model": model_info, "training": training_info}, f, indent=4)

        # Create log files
        if not self.no_train_log:
            # Create log of training progress
            self.train_log = os.path.join(self.dir, self.TRAIN_LOG_FILE)
            with open(self.train_log, "w") as f:
                row = ["epoch", "t"]
                row += [f"event_{i}" for i in range(B)]
                row += ["size_loss", "set_loss", "loss"]

                if not model.use_shell_part_sizes:
                    row += [f"pred_size_{i}" for i in range(B)]
                    row += [f"gt_size_{i}" for i in range(B)]
                    self.num_size_preds = B
                else:
                    max_num_parts = np.max([model.time_step.get_num_shell_parts(t) for t in range(1, T)])
                    row += [f"pred_size_{i}_{j}" for i in range(B) for j in range(max_num_parts)]
                    row += [f"gt_size_{i}_{j}" for i in range(B) for j in range(max_num_parts)]
                    self.num_size_preds = B * max_num_parts
                csv.writer(f).writerow(row)

        if not self.no_eval_log:
            # Create log of evaluation metrics
            self.eval_log = os.path.join(self.dir, self.EVAL_LOG_FILE)
            with open(self.eval_log, "w") as f:
                row = ["epoch", "loss"] + [
                    f"{m}_{s}_{t}" for t in range(1, T) for m in ["mse", "hd"] for s in ["train", "val"]
                ]
                csv.writer(f).writerow(row)

    def append_to_training_log(
        self,
        epoch: int,
        t: int,
        event_ids: list[str],
        size_loss: Tensor,
        set_loss: Tensor,
        loss: Tensor,
        pred_size: Tensor,
        gt_size: Tensor,
    ):
        """
        Append a new row to the training log

        :param int epoch: The current epoch
        :param int t: The current time step
        :param list[str] event_ids: The ids of the events in the batch
        :param Tensor size_loss: The size loss
        :param Tensor set_loss: The set loss
        :param Tensor loss: The total loss
        :param Tensor pred_size: The predicted sizes
        :param Tensor gt_size: The ground truth sizes
        """

        with open(self.train_log, "a") as f:
            row = [epoch, t]
            row += event_ids
            row += [size_loss.item(), set_loss.item(), loss.item()]
            pred_size_flat = pred_size.view(-1).tolist()
            padding = [""] * (self.num_size_preds - len(pred_size_flat))
            row += pred_size_flat + padding + gt_size.view(-1).tolist()
            csv.writer(f).writerow(row)

    def append_to_evaluation_log(self, epoch, loss_mean, mse_train, hd_train, mse_val, hd_val):
        """
        Append a new row to the evaluation log

        :param int epoch: The current epoch
        :param float loss_mean: The mean loss from the previous training epoch
        :param list[float] mse_train: The mean squared errors of size prediction for the training set
        :param list[float] hd_train: The Hausdorff distances of set prediction for the training set
        :param list[float] mse_val: The mean squared errors of size prediction for the validation set
        :param list[float] hd_val: The Hausdorff distances of set prediction for the validation set
        """

        with open(self.eval_log, "a") as f:
            row = [epoch, loss_mean]
            for t in range(len(mse_train)):
                row += [mse_train[t], mse_val[t], hd_train[t], hd_val[t]]
            csv.writer(f).writerow(row)

    def save_checkpoint(self, epoch, model, optimizer, scheduler, save_min_loss_model):
        """
        Save the model, optimizer and scheduler to a checkpoint file

        :param int epoch: The current epoch
        :param HitSetGenerativeModel model: The model to save
        :param Optimizer optimizer: The optimizer to save
        :param _LRScheduler scheduler: The scheduler to save
        :param bool save_min_loss_model: Whether to save the model twice (once for being
            the latest model, once for being the model with the lowest loss)
        """

        checkpoint = {
            self.MODEL_FIELD: model.state_dict(),
            self.OPTIMIZER_FIELD: optimizer.state_dict(),
            self.SCHEDULER_FIELD: scheduler.state_dict(),
            self.EPOCH_FIELD: epoch,
        }

        torch.save(checkpoint, os.path.join(self.dir, f"{self.LATEST_NAME}.pth"))
        if save_min_loss_model:
            torch.save(checkpoint, os.path.join(self.dir, f"{self.MIN_LOSS_NAME}.pth"))
