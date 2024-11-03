import argparse
import json
import os
import sys

import torch
from tqdm import tqdm, trange

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.Data import CollisionEventLoader
from src.TimeStep import TimeStepEnum
from src.TimeStep.ForAdjusting.VolumeLayer import VLTimeStep
from src.TimeStep.ForAdjusting.PlacementStrategy import EquidistantStrategy, PlacementStrategyEnum, SinusoidStrategy
from src.Pairing import (
    HungarianAlgorithmStrategy,
    PairingStrategyEnum,
    RepeatedKDTreeStrategy,
    GreedyStrategy,
    VectorizedGreedyStrategy,
)
from src.Util import CoordinateSystemEnum
from src.Util.Paths import DATA_DIR, get_precomputed_data_path
from src.Util.CoordinateSystemFuncs import convert_to_cartesian


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="train_sample", help=f"path to the input dataset, relative to {DATA_DIR}")
    ap.add_argument("--no_shell_parts", default=False, action="store_true", help="do not use shell parts")
    ap.add_argument(
        "-t",
        "--time_step",
        default=TimeStepEnum.VOLUME_LAYER.value,
        choices=[e.value for e in TimeStepEnum],
        help="time step",
    )
    ap.add_argument(
        "--placement_strategy",
        default=PlacementStrategyEnum.EQUIDISTANT.value,
        choices=[e.value for e in PlacementStrategyEnum],
        help="placement strategy",
    )
    ap.add_argument(
        "--pairing_strategy",
        default=PairingStrategyEnum.HUNGARIAN.value,
        choices=[e.value for e in PairingStrategyEnum],
        help="pairing strategy",
    )

    args = ap.parse_args()
    use_shell_parts = not args.no_shell_parts

    # Create time step
    if args.time_step == TimeStepEnum.VOLUME_LAYER.value:
        if args.placement_strategy == PlacementStrategyEnum.EQUIDISTANT.value:
            placement_strategy = EquidistantStrategy()
        else:  # if args.placement_strategy == PlacementStrategyEnum.SINUSOID.value:
            placement_strategy = SinusoidStrategy()

        if args.pairing_strategy == PairingStrategyEnum.KD_TREE.value:
            pairing_strategy = RepeatedKDTreeStrategy(k=1)
        elif args.pairing_strategy == PairingStrategyEnum.GREEDY.value:
            pairing_strategy = GreedyStrategy()
        elif args.pairing_strategy == PairingStrategyEnum.VEC_GREEDY.value:
            pairing_strategy = VectorizedGreedyStrategy()
        else:  # if args.pairing_strategy == PairingStrategyEnum.HUNGARIAN.value:
            pairing_strategy = HungarianAlgorithmStrategy()

        time_step = VLTimeStep(placement_strategy=placement_strategy, use_shell_part_sizes=use_shell_parts)
    else:
        raise ValueError(f"Time step {args.time_step} not implemented")

    # Determine device: cuda > (mps >) cpu
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"  # "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Create data loader
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_loader = CollisionEventLoader(
        os.path.join(root_dir, DATA_DIR, args.input),
        time_step,
        batch_size=2,
        val_ratio=0.0,
        shuffle=False,
        device=device,
    )

    # Determine output path
    coord_system = CoordinateSystemEnum.CYLINDRICAL
    out_path = get_precomputed_data_path(
        root_dir,
        args.input,
        TimeStepEnum(args.time_step),
        coord_system,
        PlacementStrategyEnum(args.placement_strategy),
        PairingStrategyEnum(args.pairing_strategy),
        use_shell_parts,
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Create precomputed dataset
    T = time_step.get_num_time_steps()
    gt_sizes = {os.path.basename(event_id): [] for event_id in data_loader.train_events + data_loader.val_events}

    for entry in tqdm(data_loader, desc="Computing..."):
        hits_tensor_list, batch_index_list, event_ids = entry
        out_files = {event_id: os.path.join(out_path, f"{event_id}.json") for event_id in event_ids}
        if use_shell_parts:
            out_dicts = {
                event_id: {t: [[] for _ in range(time_step.get_num_shell_parts(t))] for t in range(T)}
                for event_id in event_ids
            }
        else:
            out_dicts = {event_id: {t: [] for t in range(T)} for event_id in event_ids}

        for t in trange(T, leave=False, desc="Time steps"):
            hits_tensor = hits_tensor_list[t]
            batch_index = batch_index_list[t]
            size_tensor, part_index = data_loader.get_gt_size(hits_tensor, batch_index, t, use_shell_parts)
            part_index = part_index if use_shell_parts else None

            if t > 0:
                start_hits = time_step.place_hits(t, size_tensor, coord_system, device=device)
                hits_cart = convert_to_cartesian(hits_tensor, coord_system)
                start_hits_cart = convert_to_cartesian(start_hits, coord_system)
                pairings, _ = pairing_strategy.create_pairs(
                    hits_cart, start_hits_cart, batch_index, batch_index, part_index, part_index
                )

            for b, event_id in enumerate(event_ids):
                gt_sizes[event_id].append(size_tensor[b].tolist())

                batch_mask = batch_index == b
                if use_shell_parts:
                    for p in range(time_step.get_num_shell_parts(t)):
                        part_mask = part_index == p
                        batch_part_hits = hits_tensor[batch_mask & part_mask]
                        if t > 0:
                            batch_part_starts = start_hits[pairings[batch_mask & part_mask, 1]]
                            out_dicts[event_id][t][p] = torch.cat([batch_part_hits, batch_part_starts], dim=1).tolist()
                        else:
                            out_dicts[event_id][t][p] = batch_part_hits.tolist()
                else:
                    batch_hits = hits_tensor[batch_mask]
                    if t > 0:
                        batch_starts = start_hits[pairings[batch_mask, 1]]
                        out_dicts[event_id][t] = torch.cat([batch_hits, batch_starts], dim=1).tolist()
                    else:
                        out_dicts[event_id][t] = batch_hits.tolist()

        for event_id in event_ids:
            with open(out_files[event_id], "w") as f:
                json.dump(out_dicts[event_id], f)

    with open(os.path.join(out_path, "gt_sizes.json"), "w") as f:
        json.dump(gt_sizes, f)
