Akramujjaman Akramujjaman, Moser Matteo


# Fraud Detection ML Project

Machine Learning project for fraud detection using Logistic Regression and Random Forest classifiers.

## Setup

### 1. Environment Setup
```bash
# Create virtual environment with Python 3.9
python -m venv ml_project_clean
ml_project_clean\Scripts\activate    # Windows
# source ml_project_clean/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset
Place the dataset file `transactions.csv.zip` in the `data/` folder:
```
data/
└── transactions.csv.zip
```

## Training

### Run Training
```bash
python train_script.py
```

**Training includes:**
- Logistic Regression with GridSearchCV
- Random Forest with RandomizedSearchCV (50 iterations)
- Automatic model comparison and selection
- **Duration:** ~6-10 minutes

### Output
Trained models are saved in:
- `models/` folder

**Generated files:**
```
models/
├── logreg_model.pkl
├── rf_model.pkl
└── scaler.pkl
```

## Evaluation

### Run Evaluation
```bash
python evaluate_script.py
```

**Creates visualizations in `evaluation/` folder:**
- ROC curve comparison
- Precision-Recall curves
- Threshold analysis
- Confusion matrices

## Project Structure
```
├── data/
│   └── transactions.csv.zip       # Dataset (user provided)
├── models/                        # Trained models
├── evaluation/                    # Evaluation plots
├── train_script.py               # Training pipeline
├── evaluate_script.py            # Evaluation pipeline
└── requirements.txt              # Dependencies
```

## Requirements
- Python 3.9+
- NumPy 1.22.4
- Scikit-learn 1.6.1
- Pandas 1.5.3
- See `requirements.txt` for complete list

## Model Specifications
- **Max model size:** 50MB
- **Optimization:** Hyperparameter tuning with CV
- **Metrics:** ROC-AUC for model selection
- **Output:** Probability scores for fraud detection