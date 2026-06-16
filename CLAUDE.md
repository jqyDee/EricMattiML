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
source .venv/bin/activate              # repo-root venv, Python 3.13
cd algorithms
python main.py --algo lr --mode train  # train LR with GridSearchCV -> models/*.pkl
python main.py --algo lr --mode eval   # evaluate LR on val set
python main.py --algo lr --mode viz    # generate LR plots -> plots/lr/
python main.py --algo rf --mode train  # train RF with fixed best params -> models/*.pkl
python main.py --algo rf --mode eval   # evaluate RF on val set
python main.py --algo rf --mode viz    # generate RF plots -> plots/rf/
python main.py --mode compare          # cross-model comparison plots -> plots/compare/
python main.py --algo both --mode train --no-tune  # train both, simple (fast)
```

`--algo` ∈ {`lr`, `rf`, `both`}, `--mode` ∈ {`train`, `eval`, `viz`, `compare`, `no-train`, `all`}.
Defaults: `rf` / `train`. No test suite or linter is configured.

## Architecture

`main.py` is a thin argparse dispatcher → calls `<algo>.train.run()`, `<algo>.evaluate.evaluate()`, or
`<algo>.visualize.visualize()`. Each algorithm package exposes exactly those three functions.

`config.py` centralizes all paths (dataset in, model `.pkl` out, plot dirs) relative to repo root and
creates `models/` and `plots/` on import. Add new paths there rather than hardcoding.

### Pipeline conventions — LR (`logistic_regression/`)
- `Time` is **dropped** (no predictive signal, negligible AUPRC delta confirmed). `Amount` scaled with
  `RobustScaler` fit on train only — scaler saved as artifact for inference.
- `Feature0`–`Feature27` are pre-PCA'd, left as-is.
- Imbalance handled via `class_weight="balanced"` (not resampling).
- `train_test_split(..., stratify=y, random_state=42)` — same split reproduced in eval + visualize.
- Hyperparameter search: `GridSearchCV` over `C ∈ {0.1, 1, 10}` × `penalty ∈ {l1, l2}` ×
  `class_weight ∈ {balanced, None}`, scored on `average_precision`.
- Primary metric: **AUPRC** (`average_precision_score`). Secondary: ROC-AUC (grader leaderboard).
- Artifacts written with `pickle` (train), read with `joblib` (eval) — both load sklearn pickles fine.

### Pipeline conventions — RF (`random_forest/`)
- `Time` dropped. No feature scaling needed (tree splits on rank, not magnitude). No scaler artifact.
- Same 80/20 stratified split (`random_state=42`). `class_weight="balanced"` critical for recall.
- Best params settled after 5 tuning runs: `n_estimators=500`, `max_depth=None`, `max_features="log2"`,
  `class_weight="balanced"`, `min_samples_split=2`, `min_samples_leaf=1`.
- `fixed_params_enabled=True` in `train_tuned()` skips GridSearchCV and instantiates directly with
  best known params (fast, deterministic). Set `False` to run RandomizedSearchCV exploration.
- F1-maximizing threshold found via PR curve sweep (stored in `rf_*_threshold.pkl`).

## Status
- **LR**: fully implemented — simple + tuned, evaluate, visualize (4 plots), GridSearch CV results.
- **RF**: fully implemented — simple + tuned, evaluate, visualize (4 plots including feature importance).
- **Compare**: `compare.py` generates 4 cross-model plots (PR curves, ROC curves, metrics bar, confusion matrices).
- **Neural network** stubs at `neural_network/` are unused — RF is the second algorithm.
- `evaluate()` reproduces the same 80/20 val split locally; graders supply a separate hidden test file.
