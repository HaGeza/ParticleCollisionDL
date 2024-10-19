import os
import subprocess
import sys

if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    if not os.path.exists("../trackml-library"):
        subprocess.check_call(["git", "clone", "https://github.com/LAL/trackml-library", "../trackml-library"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "../trackml-library"])
