import os
import subprocess
import sys


def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def clone_and_install_trackml():
    if not os.path.exists("../trackml-library"):
        subprocess.check_call(["git", "clone", "https://github.com/LAL/trackml-library", "../trackml-library"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "../trackml-library"])


def clone_and_install_torch_scatter():
    if not os.path.exists("../pytorch_scatter"):
        subprocess.check_call(["git", "clone", "https://github.com/rusty1s/pytorch_scatter.git", "../pytorch_scatter"])
    subprocess.check_call([sys.executable, "setup.py", "install"], cwd="../pytorch_scatter")


if __name__ == "__main__":
    install_requirements()
    clone_and_install_trackml()
    clone_and_install_torch_scatter()
