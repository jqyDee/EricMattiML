import argparse

import logistic_regression.evaluate as lr_eval
import logistic_regression.train as lr_train
import logistic_regression.visualize as lr_viz
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
        choices=["train", "eval", "visualize"],
        default="train",
        help="Specify the execution mode: 'train', 'eval', or 'visualize'.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Use GridSearchCV hyperparameter tuning (slower). Default: simple fixed-param training.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="average_precision",
        help="Scoring metric for GridSearchCV. Defaults to average_precision.",
    )
    args = parser.parse_args()

    print(
        f"--- Starting {args.mode.upper()} mode for algorithm: {args.algo.upper()} ---"
    )

    if args.mode == "train":
        if args.algo == "lr":
            lr_train.run(tuned=args.tune, create_cv_results=True, scoring=args.scoring)
        elif args.algo == "nn":
            nn_train.run()

    elif args.mode == "eval":
        if args.algo == "lr":
            lr_eval.evaluate(tuned=args.tune)
        elif args.algo == "nn":
            nn_eval.evaluate()

    elif args.mode == "visualize":
        if args.algo == "lr":
            lr_viz.visualize()


if __name__ == "__main__":
    main()
