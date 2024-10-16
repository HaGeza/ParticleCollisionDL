import os
import zipfile
import argparse

# os.chdir("..")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include_data", action="store_true", help="Include data directory in zip file", default=False
    )

    args = parser.parse_args()

    include_data = args.include_data

    # Create zip file containing the following:
    # - requirements.txt
    # - setup.py
    # - main.py
    # - src/* # except __pycache__ files
    # - data  # if `include_data=True`
    with zipfile.ZipFile("snellius.zip", "w") as zf:
        zf.write("requirements.txt")
        zf.write("setup.py")
        zf.write("main.py")

        for root, dirs, files in os.walk("src"):
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")  # Exclude __pycache__ directories
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, os.path.relpath(file_path, os.path.join("src", "..")))

        if include_data and os.path.exists("data"):
            for root, dirs, files in os.walk("data"):
                for file in files:
                    if not os.path.isdir(file):
                        continue

                    file_path = os.path.join(root, file)
                    zf.write(file_path, os.path.relpath(file_path, os.path.join("data", "..")))
