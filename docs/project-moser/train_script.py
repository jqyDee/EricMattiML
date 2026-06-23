import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def load_data():
    """Load the fraud detection dataset"""
    path = "data/"
    train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
    return train_data

def preprocess_data(train_data):
    """Data preprocessing - exactly like your notebook"""
    print("🔧 Data Preprocessing...")
    
    # Remove 'Time' column from train_data
    train_data = train_data.drop(columns=["Time"])

    # Update features and target variable
    X_train = train_data.drop(columns=["Class"])
    y_train = train_data["Class"]

    # Standardize 'Amount' column (all other features are already normalized)
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])

    # Split data for internal validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"📊 Training set: {X_tr.shape}")
    print(f"📊 Validation set: {X_val.shape}")
    print(f"📊 Fraud ratio: {y_train.mean():.4f}")
    
    return X_tr, X_val, y_tr, y_val, scaler

def train_logistic_regression(X_tr, y_tr):
    """Train Logistic Regression model"""
    print("🎯 Training Logistic Regression...")
    logreg_model = LogisticRegression(max_iter=1000, random_state=42)
    logreg_model.fit(X_tr, y_tr)
    return logreg_model

def train_random_forest(X_tr, y_tr):
    """Train Random Forest model"""
    print("🌲 Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_tr, y_tr)
    return rf_model



def optimized_training_logistic_regression(X_tr, y_tr):
    print("Optimized Training Logistic Regression...")
    
    param_grid = {
        'C': [0.1, 1, 10],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1
    )
    
    grid_search.fit(X_tr, y_tr)
    print(f"LogReg best: {grid_search.best_params_}, score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def optimized_training_random_forest(X_tr, y_tr):
    print("Optimized Training Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # 4×5×3×3×1×3 = 540 combinations - use RandomizedSearchCV
    rand_search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        n_iter=50,  # Test 50 random combinations
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    print("RF optimization: 50 iterations × 3 CV folds = 150 fits (~5-8 minutes)")
    rand_search.fit(X_tr, y_tr)
    print(f"RF best: {rand_search.best_params_}, score: {rand_search.best_score_:.4f}")
    return rand_search.best_estimator_


def quick_validation(logreg_model, rf_model, X_val, y_val):
    """Quick validation scores - just for training feedback"""
    print("\n📈 Quick Validation Results:")
    print("-" * 40)
    
    # Logistic Regression validation
    logreg_val_score = logreg_model.predict_proba(X_val)[:, 1]
    logreg_auc = roc_auc_score(y_val, logreg_val_score)
    print(f"🎯 Logistic Regression ROC-AUC: {logreg_auc:.4f}")

    # Random Forest validation
    rf_val_score = rf_model.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_val_score)
    print(f"🌲 Random Forest ROC-AUC: {rf_auc:.4f}")
    
    # Determine best model
    if logreg_auc > rf_auc:
        best_model = "Logistic Regression"
        best_auc = logreg_auc
    else:
        best_model = "Random Forest"
        best_auc = rf_auc
    
    print(f"🏆 Best Model: {best_model} ({best_auc:.4f})")
    
    return best_model, best_auc

def save_models(logreg_model, rf_model, scaler):
    """Save all models and scaler"""
    print("\n💾 Saving models...")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    
    # Save with joblib (for local evaluation)
    joblib.dump(logreg_model, "models/logreg_model.pkl")
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    
    
    # Check model sizes
    logreg_size = os.path.getsize("models/logreg_model.pkl") / (1024 * 1024)
    rf_size = os.path.getsize("models/rf_model.pkl") / (1024 * 1024)
    scaler_size = os.path.getsize("models/scaler.pkl") / (1024 * 1024)
    total_size = logreg_size + rf_size + scaler_size
    
    print(f"📁 Logistic Regression model: {logreg_size:.2f} MB")
    print(f"📁 Random Forest model: {rf_size:.2f} MB")
    print(f"📁 Scaler: {scaler_size:.2f} MB")
    print(f"📁 Total size: {total_size:.2f} MB")
    
    if total_size > 50:
        print("⚠️  WARNING: Total model size exceeds 50MB limit!")
    else:
        print("✅ Model size is within 50MB limit")

def main():
    """Main training pipeline"""
    print("=" * 50)
    print("🚀 FRAUD DETECTION MODEL TRAINING")
    print("=" * 50)
    
    # Load and preprocess data
    train_data = load_data()
    X_tr, X_val, y_tr, y_val, scaler = preprocess_data(train_data)
    
    # Train models
    print("\n" + "=" * 50)
    #logreg_model = train_logistic_regression(X_tr, y_tr)
    #rf_model = train_random_forest(X_tr, y_tr)

    logreg_model = optimized_training_logistic_regression(X_tr, y_tr)
    rf_model     = optimized_training_random_forest(X_tr, y_tr)
    
    # Quick validation
    print("\n" + "=" * 50)
    best_model, best_auc = quick_validation(logreg_model, rf_model, X_val, y_val)
    
    # Save models
    print("\n" + "=" * 50)
    save_models(logreg_model, rf_model, scaler)
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 50)

if __name__ == "__main__":
    main()