from torch import Tensor
from src.Pairing import PairingStrategyEnum
from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Util import CoordinateSystemEnum
from .AdjustingSetGenerator import AdjustingSetGenerator


class DDPMSetGenerator(AdjustingSetGenerator):
    def __init__(
        self,
        t: int,
        time_step: ITimeStepForAdjusting,
        pairing_strategy_type: PairingStrategyEnum,
        coordinate_system: CoordinateSystemEnum,
        encoding_dim: int = 16,
        hit_dim: int = 3,
        device: str = "cpu",
    ):
        """
        Constructor for the adjusting set generator.

        :param int t: Time step to generate hits for.
        :param ITimeStepForAdjusting time_step: Time step object used for the original input hit set,
            and for generating the hit set.
        :param PairingStrategyEnum pairing_strategy_type: Pairing strategy to use.
        :param int encoding_dim: Dimension of the encoding.
        :param int hit_dim: Dimension of the hit.
        :param str device: Device to load the data on.
        """

        super().__init__(
            t,
            time_step,
            pairing_strategy_type,
            coordinate_system,
            encoding_dim,
            hit_dim,
            device,
        )

    def forward(self, _x: Tensor, _gt: Tensor, _gt_ind: Tensor, size: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the adjusting set generator.

        :param Tensor _x: Input tensor. Shape `[encoding_dim]`
        :param Tensor _gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor _gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set (Shape `[sum(size), hit_dim]`) and loss
        """

        initial_points = super().generate(_x, size)

        # create pairs

        # create noisy steps

        # predict mus and log_vars with denoising networks

        # predict final mu with final denoising network

        # calculate ELBO according to 2AMU20/A3

        # return the generated hit set and the loss

    def generate(self, _x: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor _x: (UNUSED) Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        initial_points = super().generate(_x, size)

        # create pairs

        # put standard normal around paired predicted points

        # denoise movement distributions

        # move points according to sample from denoised movement distributions

        # return the generated hit set

    def calc_loss(self, pred_tensor: Tensor, gt_tensor: Tensor, pred_ind: Tensor, gt_ind: Tensor) -> Tensor:
        """
        Not implemented, as calculating the KL terms of the loss requires the intermediate steps of
        the forward and reverse processes.

        :param Tensor pred_tensor: Predicted hit tensor. Shape `[num_hits_pred, hit_dim]`
        :param Tensor gt_tensor: Ground truth hit tensor. Shape `[num_hits_act, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits_pred]`
        "param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_act]`
        """

        raise NotImplementedError("Loss calculation for DDPM set generator is done in the forward pass.")
