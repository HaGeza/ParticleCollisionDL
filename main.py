"""Trivial example of what your main.py should look like."""
import argparse
from src.example import example_train

def main():
    ap = argparse.ArgumentParser()
    # Make arguments for number of epochs, optimizer, model, batch size, loss function, and model name
    ap.add_argument("-n", "--n-epochs", required=False, help="Number of epochs to run for", type=int, default=25)
    ap.add_argument("-o", "--optimizer", required=False, help="Optimizer to use", default="Adam")
    ap.add_argument("-m", "--model", required=False, help="Model to use", default="UNet")
    ap.add_argument("-bs", "--batch-size", required=False, help="Batch size to use", type=int, default=32)
    ap.add_argument("-l", "--loss-fn", required=False, help="Loss function to use ", default="MSE")
    ap.add_argument("-mn", "--model-name", required=False, help="Name for model", default="AR_UNet.pt")

    args = vars(ap.parse_args())
    # Run example method
    example_train(args)

if __name__ == "__main__":
    main()