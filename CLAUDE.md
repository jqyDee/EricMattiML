# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

University ML assignment (Machine Learning PS, SS 2026): binary fraud detection on an
anonymized financial transactions dataset. Hard constraints: model file ≤ **50 MB**; must run
on the course **JupyterHub** without retraining for final evaluation.

### Dataset (`dataset/transactions.csv`)
- **227,845 rows**, 31 columns: `Time`, `Feature0`–`Feature27` (PCA-transformed, pre-scaled),
  `Amount`, `Class`
- **Extreme imbalance**: 394 fraud cases (0.17%). Accuracy is a useless metric here.
- Primary metrics: **AUPRC** (`average_precision_score`), F1-score, Recall for class 1.

### Assignment requirements
- Implement **≥ 2 distinct ML methods** (different algorithm families count; same NN with
  different activations does not).
- Hyperparameter search must be *principled* (cross-validation, grid/random search — not manual
  tuning).
- The submitted **test notebook on JupyterHub must have training disabled by default** (flag
  or separate training script). Do not rename or replace the notebook.
- Submit training+eval scripts as ZIP on OLAT. Model file must be ≤ 50 MB.

### Deadlines
| What | When |
|---|---|
| Code + notebook submission | **June 24 2026, 12:00 noon** |
| Code presentation (proseminar) | June 25–26 2026 |
| Final report (PDF on OLAT) | **July 9 2026, 23:59** |

### Report (`docs/report-template/report_latex_template/report.tex`)
2-page IEEE-format report (3 pages for 3-person teams). Required sections:
1. **Introduction** — task type, dataset description (features, instances, imbalance)
2. **Implementation / ML Process** — preprocessing, method choice + rationale,
   hyperparameter selection
3. **Results** — training + validation metrics, ≥ 2 distinct visualizations
4. **Discussion** — what worked/didn't, what was tried but discarded, potential improvements
5. **Conclusion** — test-set performance, main takeaway

See `FEATURE.md` for the dataset/method writeup (still has `[Insert...]` placeholders to fill).

## Layout

- `algorithms/` — **the active codebase.** Edit here.
- `project-moser/` — a *different* students' completed submission, kept as a reference example
  (full LR + RandomForest pipeline, notebook, report). Do not modify; read for patterns only.
- `dataset/transactions.csv` — data (gitignored; the `.zip` is committed).
- `data/` — output dir for trained artifacts (gitignored), created automatically by `config.py`.
- `docs/` — assignment instructions, report template (LaTeX), example report.

## Running

All commands run **from inside `algorithms/`** — imports are flat (`from config import ...`,
`import logistic_regression.train`) and only resolve when `algorithms/` is the working dir.

```bash
source .venv/bin/activate          # repo-root venv, Python 3.13
cd algorithms
python main.py --algo lr --mode train   # train logistic regression -> data/*.pkl
python main.py --algo lr --mode eval     # load artifacts, evaluate
python main.py --algo nn --mode train   # neural network (currently a stub)
```

`--algo` ∈ {`lr`, `nn`}, `--mode` ∈ {`train`, `eval`}. Defaults: `lr` / `train`.
No test suite or linter is configured.

## Architecture

`main.py` is a thin argparse dispatcher → calls `<algo>.train.train()` or `<algo>.evaluate.evaluate()`.
Each algorithm is a package under `algorithms/` exposing exactly that `train()` / `evaluate()` pair.
To add an algorithm: create the package, implement the two functions, add a `--algo` choice + branch
in `main.py`.

`config.py` centralizes all paths (dataset in, model + scaler `.pkl` out) relative to repo root and
creates `data/`. Add new paths there rather than hardcoding.

### Pipeline conventions (from `logistic_regression/`)
- Only `Time` and `Amount` are scaled (`RobustScaler`); `Feature0..27` are pre-PCA'd, left as-is.
- Scaler is `fit` on train only, then `transform` on val/test — never `fit` on test (data leakage).
- Imbalance handled via `class_weight="balanced"` (not resampling).
- `train_test_split(..., stratify=y, random_state=42)` to preserve fraud rate across the split.
- Primary metric is **AUPRC** (`average_precision_score`) plus the classification report — accuracy is
  meaningless here. Optimize recall/F1 for the fraud class.
- Artifacts written with `pickle` (train) but read with `joblib` (eval) — both load sklearn pickles fine.

## Status / gotchas
- `neural_network/train.py` and `evaluate.py` are stubs (`pass`) — not yet implemented.
- `evaluate()` currently points at the same `DATASET_PATH` as training; the real graders supply a
  separate hidden test file. There are `TODO`s in `lr/train.py` about JupyterHub's own validation.
