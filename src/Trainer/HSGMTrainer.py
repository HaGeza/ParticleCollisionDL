from tqdm import tqdm, trange

from scipy.spatial.distance import directed_hausdorff
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import lr_scheduler, Optimizer

from src.Data import IDataLoader
from src.Trainer.TrainingRunIO import TrainingRunIO
from src.Modules.HitSetGenerativeModel import HitSetGenerativeModel
from src.Util import CoordinateSystemEnum
from src.Util.CoordinateSystemFuncs import convert_to_cartesian


class Trainer:
    """
    Trainer class for training a `HitSetGenerativeModel`s
    """

    DEFAULT_SIZE_LOSS_W = 1000

    def __init__(
        self,
        model: HitSetGenerativeModel,
        optimizer: Optimizer,
        scheduler: lr_scheduler._LRScheduler,
        data_loader: IDataLoader,
        run_io: TrainingRunIO,
        start_epoch: int = 0,
        epochs: int = 100,
        size_loss_weight: float = DEFAULT_SIZE_LOSS_W,
        device: str = "cpu",
    ):
        """
        :param HitSetGenerativeModel model: The model to train
        :param Optimizer optimizer: The optimizer to use
        :param lr_scheduler._LRScheduler scheduler: The learning rate scheduler to use
        :param IDataLoader data_loader: The data loader to use
        :param TrainingRunIO run_io: The run IO to use for logging and saving models
        :param int start_epoch: The epoch to start training from
        :param int epochs: The number of epochs to train for
        :param float size_loss_weight: The weight to give to the size loss
        :param str device: The device to use
        :param str models_path: The path to save the models
        """

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.run_io = run_io
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.size_loss_weight = size_loss_weight
        self.device = device

        if model.device != device:
            model.to(device)

    def _get_hausdorff_distance(
        self, pred_tensor: Tensor, gt_tensor: Tensor, coordinate_system: CoordinateSystemEnum
    ) -> Tensor:
        """
        Calculate the Hausdorff distance between the predicted and ground truth hit sets.

        :param Tensor pred_tensor: The predicted hit tensor
        :param Tensor gt_tensor: The ground truth hit tensor
        :return Tensor: The Hausdorff distance
        """

        if pred_tensor.size(0) == 0 or gt_tensor.size(0) == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        pred_cart = convert_to_cartesian(pred_tensor, coordinate_system)
        gt_cart = convert_to_cartesian(gt_tensor, coordinate_system)

        if pred_cart.device.type != "cuda":
            pred_hits = pred_cart.cpu().numpy()
            gt_hits = gt_cart.cpu().numpy()

            pred_to_gt = directed_hausdorff(pred_hits, gt_hits)[0]
            gt_to_pred = directed_hausdorff(gt_hits, pred_hits)[0]
        else:
            dists = torch.cdist(pred_cart, gt_cart, p=2)
            pred_to_gt = torch.min(dists, dim=1)[0].max(dim=0)[0].item()
            gt_to_pred = torch.min(dists, dim=0)[0].max(dim=0)[0].item()

        return torch.tensor(max(pred_to_gt, gt_to_pred), dtype=torch.float32, device=self.device)

    def evaluate(self, data_loader: IDataLoader, events: list[str]) -> tuple[list[float], list[float]]:
        """
        Evaluate the model using the given data iterator.

        :param callable data_iter: The data loader
        :param list[str] events: The list of events to evaluate
        :return tuple[list[float], list[float]]: The mean Hausdorff distances and mean size MSEs
            per time step
        """

        T = self.model.time_step.get_num_time_steps()
        hds = [0] * (T - 1)
        mses = [0] * (T - 1)
        num_entries = 0

        for entry in tqdm(data_loader.iter_events(events), leave=False, desc="Evaluating..."):
            hits_tensor_list, batch_index_list, event_ids = entry

            in_tensor = hits_tensor_list[0]
            in_batch_index = batch_index_list[0].detach()
            in_dim = in_tensor.size(1)

            for t in trange(1, T, leave=False, desc="Time steps"):
                gt_tensor = hits_tensor_list[t][:, :in_dim]

                gt_batch_index = batch_index_list[t].detach()
                gt_size, _ = data_loader.get_gt_size(gt_tensor, gt_batch_index, t, events=event_ids)
                B = gt_size.size(0)

                pred_size, pred_tensor = self.model.generate(in_tensor, in_batch_index, t, batch_size=B)

                used_size = torch.clamp(pred_size, min=self.model.min_size_to_generate).round().int()
                if used_size.dim() > 1:
                    used_size = used_size.sum(dim=1)

                if used_size.sum() > 0:
                    pred_batch_index = torch.repeat_interleave(
                        torch.arange(len(used_size), device=self.device), used_size
                    )
                    for b in range(B):
                        hds[t - 1] += self._get_hausdorff_distance(
                            pred_tensor[pred_batch_index == b],
                            gt_tensor[gt_batch_index == b],
                            self.model.coordinate_system,
                        ).item()
                else:
                    pred_batch_index = torch.tensor([], device=self.device, dtype=torch.long)

                mses[t - 1] += self.model.size_generators[t - 1].calc_loss(pred_size, gt_size).item()
                num_entries += B

                in_tensor = pred_tensor
                in_batch_index = pred_batch_index

        num_entries = max(num_entries, 1)
        return [hd / num_entries for hd in hds], [mse / num_entries for mse in mses]

    def train_and_eval(self):
        """
        Train and evaluate the model using the given data loader.
        """

        if self.model.device != self.device:
            self.model.to(self.device)

        # Main training loop
        min_loss = float("inf")
        self.model.train()

        for epoch in trange(self.start_epoch, self.epochs):
            loss_mean = 0
            num_entries = 0

            for entry in tqdm(self.data_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False):
                hits_tensor_list, batch_index_list, event_ids = entry

                in_tensor = hits_tensor_list[0]
                in_batch_index = batch_index_list[0].detach()
                in_dim = in_tensor.size(1)

                for t in trange(1, self.model.time_step.get_num_time_steps(), leave=False, desc="Time steps"):
                    gt_tensor = hits_tensor_list[t][:, :in_dim]
                    initial_pred = hits_tensor_list[t][:, in_dim:]

                    gt_batch_index = batch_index_list[t].detach()
                    gt_size, _ = self.data_loader.get_gt_size(gt_tensor, gt_batch_index, t, events=event_ids)

                    self.optimizer.zero_grad()

                    pred_size, _pred_tensor, size_loss, set_loss = self.model(
                        in_tensor, gt_tensor, in_batch_index, gt_batch_index, gt_size, t, initial_pred
                    )

                    size_loss = size_loss * self.size_loss_weight
                    loss = size_loss + set_loss
                    loss.backward()
                    self.optimizer.step()

                    in_tensor = gt_tensor
                    in_batch_index = gt_batch_index

                    loss_mean += loss.item()
                    num_entries += gt_size.size(0)

                    if not self.run_io.no_log:
                        self.run_io.append_to_training_log(
                            epoch, t, event_ids, size_loss, set_loss, loss, pred_size, gt_size
                        )

                self.scheduler.step()

            # end of epoch
            if num_entries > 0:
                loss_mean = loss_mean / num_entries

            save_min_loss_model = False
            if loss_mean < min_loss:
                min_loss = loss_mean
                save_min_loss_model = True

            self.run_io.save_checkpoint(
                epoch + 1, self.model, self.optimizer, self.scheduler, self.size_loss_weight, save_min_loss_model
            )

            if not self.run_io.no_log:
                self.model.eval()
                with torch.no_grad():
                    hd_train, mse_train = self.evaluate(self.data_loader, self.data_loader.train_events)
                    hd_val, mse_val = self.evaluate(self.data_loader, self.data_loader.val_events)
                self.run_io.append_to_evaluation_log(epoch, loss_mean, hd_train, mse_train, hd_val, mse_val)
