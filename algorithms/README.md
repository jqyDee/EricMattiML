# algorithms/

All commands must be run from inside this directory - imports are flat and only resolve here.
```bash
cd algorithms/
```

## Optional: Create virtual environment
```bash
python -m venv ../.venv

source ../.venv/bin/activate  # UNIX
..\.venv\Scripts\activate.bat  # Windows
```

## Dataset setup
Create a `./dataset/` folder and place the dataset file there:
```
.
└── dataset/
    └── transactions.csv.zip   <- drop it here (no need to unzip)
```
`config.py` reads the `.zip` directly via pandas.

## Dependencies
```bash
pip install -r requirements.txt
```

---

## main.py - convenience entry point

Dispatches to any algorithm's train/eval/visualize functions via flags.

```bash
# Logistic Regression
python main.py --algo lr --mode train              # train LR with GridSearchCV (default)
python main.py --algo lr --mode train --no-tune    # train LR simple, fixed params
python main.py --algo lr --mode eval               # evaluate tuned LR on val set
python main.py --algo lr --mode eval --no-tune     # evaluate simple LR on val set
python main.py --algo lr --mode viz                # generate LR plots -> plots/lr/

# Random Forest
python main.py --algo rf --mode train              # train RF with fixed best params (default)
python main.py --algo rf --mode train --no-tune    # train RF simple (100 trees)
python main.py --algo rf --mode eval               # evaluate tuned RF on val set
python main.py --algo rf --mode eval --no-tune     # evaluate simple RF on val set
python main.py --algo rf --mode viz                # generate RF plots -> plots/rf/

# Cross-model comparison
python main.py --mode compare                      # compare LR vs RF -> plots/compare/

# Train both
python main.py --algo both --mode train            # train both algorithms (slow)
```

Flags:
- `--algo` - `lr` (Logistic Regression), `rf` (Random Forest), or `both`
- `--mode` - `train`, `eval`, `viz`, `compare`, `no-train`, or `all`
- `--no-tune` - disable tuned training, use simple fixed params instead
- `--scoring` - scoring metric for GridSearch (default: `average_precision`)
- `--output-dir` - where to save artifacts (default: `./models`)

> Note: `--algo both --mode train` takes a long time. LR GridSearch is the slow part.

Help:
```bash
python main.py --help
python logistic_regression/train.py --help
python random_forest/train.py --help
```

---

## Individual scripts - run directly

Every train script can be run standalone with a custom dataset path and output directory.
Useful for running on JupyterHub without the full repo structure.

### logistic_regression/train.py

```bash
python logistic_regression/train.py \
    --dataset /path/to/transactions.csv \
    --output-dir ./models \
    --tune \
    --scoring average_precision \
    --create-cv-results
```

Flags:
- `--dataset` - path to CSV (defaults to config.py path if omitted)
- `--output-dir` - where to save `.pkl` artifacts (default: `./models`)
- `--tune` - use GridSearchCV; omit for simple fixed-param training
- `--scoring` - GridSearch metric (default: `average_precision`)
- `--create-cv-results` - save raw CV results pickle for visualization

Artifacts: `lr_simple_model.pkl` / `lr_tuned_model.pkl`, `lr_*_scaler.pkl`, `lr_*_threshold.pkl`

### random_forest/train.py

```bash
python random_forest/train.py \
    --dataset /path/to/transactions.csv \
    --output-dir ./models \
    --tune
```

Same flags as LR train script. No scaler artifact - RF does not require feature scaling.

Best params (fixed in `train_tuned`): `n_estimators=500`, `max_depth=None`, `max_features='log2'`,
`class_weight='balanced'`. Set `fixed_params_enabled=False` in source to run RandomizedSearchCV.

Artifacts: `rf_simple_model.pkl` / `rf_tuned_model.pkl`, `rf_*_threshold.pkl`

---

## config.py

Centralizes all paths and constants. Imported automatically when running from `algorithms/`.
When running scripts directly with `--dataset` / `--output-dir`, config is not required.

Add new artifact paths here - do not hardcode paths in individual scripts.

---

## models/

Created automatically on first import of `config.py`. Contains:
- `*.pkl` - trained model, scaler (LR only), and threshold artifacts

## plots/

Created automatically. Subdirectories per algorithm plus comparison:
- `plots/lr/` - LR PR curves, confusion matrices, GridSearch results, threshold analysis
- `plots/rf/` - RF PR curves, confusion matrices, feature importance, threshold analysis
- `plots/compare/` - cross-model PR curves, ROC curves, metrics bar chart, confusion matrices
