from pathlib import Path

ALG_PATH = Path(__file__).resolve()
ROOT_PATH = ALG_PATH.parent.parent
DATA_PATH = ROOT_PATH / "data"

DATASET_PATH = ROOT_PATH / "dataset" / "transactions.csv"

MODEL_SAVE_PATH = DATA_PATH / "lr_fraud_model.pkl"
SCALER_SAVE_PATH = DATA_PATH / "lr_robust_scaler.pkl"

DATA_PATH.mkdir(parents=True, exist_ok=True)
