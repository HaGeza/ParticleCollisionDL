import argparse
import os
import shutil
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.Pairing import PairingStrategyEnum
from src.TimeStep.ForAdjusting.PlacementStrategy import PlacementStrategyEnum
from src.Util import CoordinateSystemEnum
from src.TimeStep import TimeStepEnum
from src.Util.Paths import DATA_DIR, PRECOMPUTED_DATA_DIR, get_precomputed_data_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=1, help="start_index of zips to process")
    parser.add_argument("--end_index", type=int, default=5, help="end_index of zips to process (inclusive)")
    parser.add_argument("--force", action="store_true", default=False, help="force overwrite existing files")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(args.start_index, args.end_index)

    for i in range(args.start_index, args.end_index + 1):
        # Check if file already exists
        train_file = f"train_{i}"
        out_path = get_precomputed_data_path(
            root_dir,
            train_file,
            TimeStepEnum.VOLUME_LAYER,
            CoordinateSystemEnum.CYLINDRICAL,
            PlacementStrategyEnum.EQUIDISTANT,
            PairingStrategyEnum.HUNGARIAN,
            True,
        )

        if os.path.exists(out_path) and not args.force:
            print(f"Skipping {out_path}")
            continue

        # Download the data if needed
        raw_data_path = os.path.join(root_dir, DATA_DIR, train_file)
        if not os.path.exists(raw_data_path):
            print(f"Downloading {train_file}.zip ...")
            try:
                subprocess.check_call(
                    [
                        "kaggle",
                        "competitions",
                        "download",
                        "trackml-particle-identification",
                        "-f",
                        f"{train_file}.zip",
                    ]
                )
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {train_file}.zip: {e}")
                print("Make sure Kaggle is configured properly!")
                sys.exit(1)

        # Unzip the data and remove zip
        subprocess.check_call(["unzip", f"{train_file}.zip", "-d", raw_data_path])
        os.remove(f"{train_file}.zip")

        # Preprocess the data
        subprocess.check_call([sys.executable, "scripts/create_precomputed_dataset.py", "--input", train_file])

        # Remove raw data
        shutil.rmtree(raw_data_path)

        # Zip the preprocessed data and remove the directory
        zip_path = os.path.join(os.path.dirname(out_path), f"{train_file}.zip")
        subprocess.check_call(["zip", "-r", zip_path, out_path])
        shutil.rmtree(out_path)
