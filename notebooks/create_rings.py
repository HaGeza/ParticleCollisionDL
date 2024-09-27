import json
import os
from trackml.dataset import load_event
from tqdm import tqdm


def find_root_directory(target_file="main.py"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if target_file in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise FileNotFoundError(f"{target_file} not found in any parent directories.")
        current_dir = parent_dir


def save_dict_as_python_file(dictionary, filename):
    with open(filename, "w") as f:
        f.write("VL_TO_RING = ")
        f.write(repr(dictionary))


if __name__ == "__main__":
    vl_to_ring = {}

    root_dir = find_root_directory()
    data_dir = os.path.join(root_dir, "data", "train_sample")

    for root, _, files in tqdm(os.walk(data_dir)):
        event_ids = {f.split("-")[0] for f in files}

        for event_id in tqdm(event_ids, leave=False):
            hits = load_event(os.path.join(root, event_id))[0]

            grouped_hits = hits.groupby(["volume_id", "layer_id"])

            for (volume_id, layer_id), group in grouped_hits:
                z1 = float(group["z"].min())
                z2 = float(group["z"].max())
                xy_dist2 = group["x"] ** 2 + group["y"] ** 2
                r1 = float(xy_dist2.min() ** 0.5)
                r2 = float(xy_dist2.max() ** 0.5)

                volume_id = int(volume_id)
                layer_id = int(layer_id)
                if volume_id not in vl_to_ring:
                    vl_to_ring[volume_id] = {layer_id: (r1, r2, z1, z2)}
                elif layer_id not in vl_to_ring[volume_id]:
                    vl_to_ring[volume_id][layer_id] = (r1, r2, z1, z2)
                else:
                    r1_old, r2_old, z1_old, z2_old = vl_to_ring[volume_id][layer_id]
                    vl_to_ring[volume_id][layer_id] = (
                        min(r1, r1_old),
                        max(r2, r2_old),
                        min(z1, z1_old),
                        max(z2, z2_old),
                    )

    vl_to_ring_path = os.path.join(root_dir, "src", "TimeStep", "VolumeLayer", "VLRings.py")
    save_dict_as_python_file(vl_to_ring, vl_to_ring_path)
