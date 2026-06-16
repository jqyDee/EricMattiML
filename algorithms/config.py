from pathlib import Path

from sklearn.preprocessing import RobustScaler

# Pathing
ALG_PATH = Path(__file__).resolve()
ROOT_PATH = ALG_PATH.parent
MODEL_PATH = ROOT_PATH / "models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH = ROOT_PATH / "plots"
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

DATASET_PATH = ROOT_PATH / "dataset" / "transactions.csv.zip"

# Logistic Regression
LR_SPLIT_RATIO = 0.2
LR_RANDOM_STATE = 42
LR_SCALER = RobustScaler()

## Paths
LR_MODEL_SAVE_PATH = MODEL_PATH / "lr_simple_model.pkl"
LR_SCALER_SAVE_PATH = MODEL_PATH / "lr_simple_scaler.pkl"
LR_THRESHOLD_SAVE_PATH = MODEL_PATH / "lr_simple_threshold.pkl"

LR_TUNED_MODEL_SAVE_PATH = MODEL_PATH / "lr_tuned_model.pkl"
LR_TUNED_SCALER_SAVE_PATH = MODEL_PATH / "lr_tuned_scaler.pkl"
LR_TUNED_THRESHOLD_SAVE_PATH = MODEL_PATH / "lr_tuned_threshold.pkl"
LR_TUNED_CV_RESULTS_PATH = MODEL_PATH / "lr_tuned_cv_results.pkl"

LR_PLOTS_PATH = PLOTS_PATH / "lr"
LR_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Random Forest
RF_SPLIT_RATIO = 0.2
RF_RANDOM_STATE = 42

## Paths
RF_MODEL_SAVE_PATH = MODEL_PATH / "rf_simple_model.pkl"
RF_THRESHOLD_SAVE_PATH = MODEL_PATH / "rf_simple_threshold.pkl"

RF_TUNED_MODEL_SAVE_PATH = MODEL_PATH / "rf_tuned_model.pkl"
RF_TUNED_THRESHOLD_SAVE_PATH = MODEL_PATH / "rf_tuned_threshold.pkl"
RF_TUNED_CV_RESULTS_PATH = MODEL_PATH / "rf_tuned_cv_results.pkl"

RF_PLOTS_PATH = PLOTS_PATH / "rf"
RF_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

COMPARE_PLOTS_PATH = PLOTS_PATH / "compare"
COMPARE_PLOTS_PATH.mkdir(parents=True, exist_ok=True)
