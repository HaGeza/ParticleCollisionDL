import torch
from torch import Tensor

from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Pairing import PairingStrategyEnum, IPairingStrategy, GreedyStrategy, RepeatedKDTreeStrategy
from src.Util import CoordinateSystemEnum
from .IHitSetGenerator import IHitSetGenerator


class AdjustingSetGenerator(IHitSetGenerator):
    """
    Class for generating placing hits according to some strategy, and learning to adjust them.
    """

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

        super().__init__()

        self.device = device
        self.to(device)

        self.t = t
        self.time_step = time_step
        self.coordinate_system = coordinate_system
        self.encoding_dim = encoding_dim
        self.hit_dim = hit_dim

        self.pairing_strategy: IPairingStrategy = None
        if pairing_strategy_type == PairingStrategyEnum.GREEDY:
            self.pairing_strategy = GreedyStrategy()
        elif pairing_strategy_type == PairingStrategyEnum.KD_TREE:
            self.pairing_strategy = RepeatedKDTreeStrategy()
        else:
            raise ValueError(f"Unknown pairing strategy: {pairing_strategy_type}")

        self.max_pair_loss = time_step.get_max_squared_distance()

    def forward(self, z: Tensor, gt: Tensor, pred_ind: Tensor, gt_ind: Tensor, size: Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward pass of the adjusting set generator.

        :param Tensor z: Encoded input hit set. Shape `[encoding_dim]`
        :param Tensor gt: Ground truth tensor. Shape `[num_hits_next, hit_dim]`
        :param Tensor pred_ind: Predicted hit batch index tensor. Shape `[num_hits_pred]`.
        :param Tensor gt_ind: Ground truth hit batch index tensor. Shape `[num_hits_next]`.
        :param Tensor used_size: Size of the generated hit point-cloud.
            Shape `[num_batches]` or `[num_batches, num_parts_next]`.
        :return: Generated hit set (Shape `[sum(size), hit_dim]`), and the loss
        """

        pred = self.generate(z, size)
        return pred, self.pairing_strategy.calculate_loss(pred, gt, pred_ind, gt_ind, self.coordinate_system)

    def generate(self, z: Tensor, size: Tensor) -> Tensor:
        """
        Generate a hit set.

        :param Tensor z: Input tensor. Shape `[encoding_dim]`
        :param Tensor size: Size of the generated hit point-cloud.
        :return: Generated hit set. Shape `[sum(size), hit_dim]`
        """

        with torch.no_grad():
            return self.time_step.place_hits(self.t + 1, size, self.coordinate_system, device=self.device)
