import torch
from torch import Tensor

from src.TimeStep.ForAdjusting import ITimeStepForAdjusting
from src.Pairing import (
    HungarianAlgorithmStrategy,
    PairingStrategyEnum,
    IPairingStrategy,
    GreedyStrategy,
    RepeatedKDTreeStrategy,
    VectorizedGreedyStrategy,
)
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
        _device: str = "cpu",
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

        self.t = t
        self.time_step = time_step
        self.coordinate_system = coordinate_system

        self.pairing_strategy: IPairingStrategy = None
        if pairing_strategy_type == PairingStrategyEnum.GREEDY:
            self.pairing_strategy = GreedyStrategy()
        elif pairing_strategy_type == PairingStrategyEnum.KD_TREE:
            self.pairing_strategy = RepeatedKDTreeStrategy()
        elif pairing_strategy_type == PairingStrategyEnum.VEC_GREEDY:
            self.pairing_strategy = VectorizedGreedyStrategy()
        else:  # if args.pairing_strategy == PairingStrategyEnum.HUNGARIAN.value:
            self.pairing_strategy = HungarianAlgorithmStrategy()

    def forward(
        self,
        z: Tensor,
        gt: Tensor,
        pred_ind: Tensor,
        gt_ind: Tensor,
        size: Tensor,
        initial_pred: Tensor = torch.tensor([]),
    ) -> tuple[Tensor, Tensor]:
        pred = self.generate(z, size)
        return pred, self.pairing_strategy.calculate_loss(
            pred, gt, pred_ind, gt_ind, coordinate_system=self.coordinate_system
        )

    def generate(self, z: Tensor, size: Tensor, initial_pred: Tensor = torch.tensor([])) -> Tensor:
        if initial_pred.size(0) > 0:
            return initial_pred

        with torch.no_grad():
            return self.time_step.place_hits(self.t + 1, size, self.coordinate_system, device=self.device)
