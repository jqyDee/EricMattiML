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

## Dependencies
```bash
pip install -r requirements.txt
```

---

## main.py - convenience entry point

Dispatches to any algorithm's train/eval/visualize functions via flags.

```bash
python main.py --algo lr --mode train              # train LR with GridSearchCV (default)
python main.py --algo lr --mode train --no-tune    # train LR simple, fixed params
python main.py --algo lr --mode eval               # evaluate tuned LR on dataset
python main.py --algo lr --mode eval --no-tune     # evaluate simple LR on dataset
python main.py --algo lr --mode viz                # generate LR plots -> data/plots/

python main.py --algo rf --mode train              # train RF with GridSearchCV (default)
python main.py --algo rf --mode train --no-tune    # train RF simple, fixed params
python main.py --algo rf --mode eval               # evaluate tuned RF on dataset
python main.py --algo rf --mode eval --no-tune     # evaluate simple RF on dataset
```

Flags:
- `--algo` - `lr` (Logistic Regression) or `rf` (Random Forest) or `both` (both algorithms)
- `--mode` - `train`, `eval`, or `viz` or `all` (train/eval/viz for both algorithms)
- `--no-tune` - disable GridSearchCV, use fixed params (GridSearch is on by default)
- `--scoring` - scoring metric for GridSearch (default: `average_precision`)
- `--output-dir` - where to save artifacts (default: `./models`)

> Important: When using `--algo both` with `--mode train` or `--mode all`, the
  script will take quite a while to train the models. Use with caution. 

Help:
```bash
python {main.py, logistic_regression/train.py, random_forest/train.py, ...} --help
```


---

## Individual scripts - run directly

Every train script can be run standalone with a custom dataset path and output directory.
Useful for running on JupyterHub or any environment without the full repo structure.
main.py is just a convenience wrapper - it adds nothing the individual scripts cannot do.

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
    --tune \
    --create-cv-results
```

Same flags as LR train script. No scaler artifact - RF does not require feature scaling.

Artifacts: `rf_simple_model.pkl` / `rf_tuned_model.pkl`, `rf_*_threshold.pkl`

---

## config.py

Centralizes all paths and constants. Imported automatically when running from `algorithms/`.
When running scripts directly with `--dataset` / `--output-dir`, config is not required.

Add new artifact paths here - do not hardcode paths in individual scripts.

---

## models/

Created automatically on first import of `config.py`. Contains:
- `*.pkl` - trained model, scaler, and threshold artifacts
- `plots/` - visualization outputs from `--mode viz`
