import argparse

import compare as compare_viz
import logistic_regression.evaluate as lr_eval
import logistic_regression.train as lr_train
import logistic_regression.visualize as lr_viz
import random_forest.evaluate as rf_eval
import random_forest.train as rf_train
import random_forest.visualize as rf_viz


def separator():
    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["both", "lr", "rf"],
        default="rf",
        help="Specify the algorithm to train: 'lr' for Logistic Regression or 'nn' for Neural Network.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "eval", "viz", "no-train", "compare"],
        default="train",
        help="Specify the execution mode: 'train', 'eval', 'viz', 'compare' (LR vs RF), 'no-train', or 'all'.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_false",
        default=True,  # this is inverted, so default (True) means tuning is enabled
        help="Use GridSearchCV hyperparameter tuning (slower). Default: simple fixed-param training.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="average_precision",
        help='Scoring metric for GridSearchCV. Defaults to "average_precision".',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save model artifacts. Defaults to ./models",
    )
    args = parser.parse_args()

    print("=" * 100)
    print(
        f"--- Starting {args.mode.upper()} mode for algorithm: {args.algo.upper()} ---"
    )
    print("=" * 100)

    if args.mode == "train" or args.mode == "all":
        if args.algo == "lr" or args.algo == "both":
            lr_train.run(
                tuned=args.no_tune,
                create_cv_results=True,
                scoring=args.scoring,
                output_dir=args.output_dir,
            )

        if args.algo == "both" or args.mode == "all":
            separator()

        if args.algo == "rf" or args.algo == "both":
            rf_train.run(
                tuned=args.no_tune,
                create_cv_results=True,
                scoring=args.scoring,
                output_dir=args.output_dir,
            )

    if args.mode == "eval" or args.mode == "all" or args.mode == "no-train":
        if args.algo == "lr" or args.algo == "both":
            lr_eval.evaluate(tuned=args.no_tune)

        if args.algo == "both" or args.mode == "all":
            separator()

        if args.algo == "rf" or args.algo == "both":
            rf_eval.evaluate(tuned=args.no_tune)

    if args.mode == "viz" or args.mode == "all" or args.mode == "no-train":
        if args.algo == "lr" or args.algo == "both":
            lr_viz.visualize()

        if args.algo == "both" or args.mode == "all":
            separator()

        if args.algo == "rf" or args.algo == "both":
            rf_viz.visualize()

    if args.mode == "compare":
        compare_viz.compare()


if __name__ == "__main__":
    main()
