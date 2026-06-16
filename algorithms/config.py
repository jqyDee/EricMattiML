from pathlib import Path
from sklearn.preprocessing import RobustScaler

# Pathing
ALG_PATH = Path(__file__).resolve()
ROOT_PATH = ALG_PATH.parent.parent
DATA_PATH = ROOT_PATH / "data"

DATASET_PATH = ROOT_PATH / "dataset" / "transactions.csv"

LR_MODEL_SAVE_PATH = DATA_PATH / "lr_simple_model.pkl"
LR_SCALER_SAVE_PATH = DATA_PATH / "lr_simple_scaler.pkl"
LR_THRESHOLD_SAVE_PATH = DATA_PATH / "lr_simple_threshold.pkl"

LR_TUNED_MODEL_SAVE_PATH = DATA_PATH / "lr_tuned_model.pkl"
LR_TUNED_SCALER_SAVE_PATH = DATA_PATH / "lr_tuned_scaler.pkl"
LR_TUNED_THRESHOLD_SAVE_PATH = DATA_PATH / "lr_tuned_threshold.pkl"
LR_TUNED_CV_RESULTS_PATH = DATA_PATH / "lr_tuned_cv_results.pkl"

LR_PLOTS_PATH = DATA_PATH / "plots"

DATA_PATH.mkdir(parents=True, exist_ok=True)
LR_PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Logistic Regression
LR_SPLIT_RATIO = 0.2
LR_RANDOM_STATE = 42
LR_SCALER = RobustScaler()
