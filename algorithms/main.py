import argparse
import shutil
import zipfile

import compare as compare_viz
import logistic_regression.evaluate as lr_eval
import logistic_regression.train as lr_train
import logistic_regression.visualize as lr_viz
import random_forest.evaluate as rf_eval
import random_forest.train as rf_train
import random_forest.visualize as rf_viz
from config import (
    LR_TUNED_MODEL_SAVE_PATH,
    LR_TUNED_SCALER_SAVE_PATH,
    LR_TUNED_THRESHOLD_SAVE_PATH,
    MODEL_PATH,
    RF_TUNED_MODEL_SAVE_PATH,
    RF_TUNED_THRESHOLD_SAVE_PATH,
)

SIZE_LIMIT_MB = 50


def separator():
    print("\n\n" + "=" * 100)
    print("=" * 100 + "\n\n")


def check_and_package():
    print("=" * 100)
    print("SUBMISSION CHECK")
    print("=" * 100)

    artifacts = [
        ("LR tuned model", LR_TUNED_MODEL_SAVE_PATH, True),
        ("LR tuned scaler", LR_TUNED_SCALER_SAVE_PATH, False),
        ("LR tuned threshold", LR_TUNED_THRESHOLD_SAVE_PATH, False),
        ("RF tuned model", RF_TUNED_MODEL_SAVE_PATH, True),
        ("RF tuned threshold", RF_TUNED_THRESHOLD_SAVE_PATH, False),
    ]

    ok = True
    total_mb = 0.0
    print("\n--- Artifact check ---")
    for label, path, check_size in artifacts:
        if not path.exists():
            print(f"  MISSING  {label}: {path}")
            ok = False
            continue
        size_mb = path.stat().st_size / 1024 / 1024
        if check_size:
            total_mb += size_mb
        print(
            f"  OK       {label}: {f'{size_mb:.2f} MB' if check_size else 'n/a':>10}  {path.name}"
        )

    print(f"\n  Total model size: {total_mb:.2f} MB / {SIZE_LIMIT_MB} MB limit")
    if total_mb > SIZE_LIMIT_MB:
        print(f"  TOO BIG — combined models exceed {SIZE_LIMIT_MB} MB limit.")
        ok = False

    if not ok:
        print("\nFix issues above before packaging.")
        return

    print("\n--- Packaging submission ---")
    submission_dir = MODEL_PATH / "selected_models"
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True)

    copies = [
        (LR_TUNED_MODEL_SAVE_PATH, "chosen-model-1.pkl"),
        (LR_TUNED_SCALER_SAVE_PATH, "chosen-model-1-scaler.pkl"),
        (LR_TUNED_THRESHOLD_SAVE_PATH, "chosen-model-1-threshold.pkl"),
        (RF_TUNED_MODEL_SAVE_PATH, "chosen-model-2.pkl"),
        (RF_TUNED_THRESHOLD_SAVE_PATH, "chosen-model-2-threshold.pkl"),
    ]

    for src, dst_name in copies:
        dst = submission_dir / dst_name
        shutil.copy2(src, dst)
        print(f"  {src.name}  ->  {dst_name}")

    zip_path = MODEL_PATH / "model_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(submission_dir.iterdir()):
            zf.write(f, f.name)

    zip_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"\n  Zipped -> {zip_path}  ({zip_mb:.2f} MB)")
    print("\nAll checks passed. Submission ready.")


def main():
    parser = argparse.ArgumentParser(description="Train a fraud detection model.")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["both", "lr", "rf"],
        default="rf",
        help="Algorithm: 'lr', 'rf', or 'both'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "train", "eval", "viz", "no-train", "compare", "check"],
        default="train",
        help="'train', 'eval', 'viz', 'compare', 'check' (verify + package submission), 'no-train', or 'all'.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_false",
        default=True,
        help="Disable tuned training, use simple fixed params.",
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

    if args.mode == "check":
        check_and_package()
        return

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
