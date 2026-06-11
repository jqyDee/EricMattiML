import argparse

import logistic_regression.evaluate as lr_eval
import logistic_regression.train as lr_train
import neural_network.evaluate as nn_eval
import neural_network.train as nn_train


def main():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["lr", "nn"],
        default="lr",
        help="Specify the algorithm to train: 'lr' for Logistic Regression or 'nn' for Neural Network.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        default="train",
        help="Specify the execution mode: 'train' to train a new model, or 'eval' to test an existing one.",
    )
    args = parser.parse_args()

    print(
        f"--- Starting {args.mode.upper()} mode for algorithm: {args.algo.upper()} ---"
    )

    if args.mode == "train":
        if args.algo == "lr":
            lr_train.train()
        elif args.algo == "nn":
            nn_train.train()

    elif args.mode == "eval":
        if args.algo == "lr":
            lr_eval.evaluate()
        elif args.algo == "nn":
            nn_eval.evaluate()


if __name__ == "__main__":
    main()
