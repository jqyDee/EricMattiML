import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, auc, precision_recall_curve, 
                           average_precision_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, ConfusionMatrixDisplay,
                           accuracy_score, classification_report)

def load_models():
    """Load trained models and scaler"""
    try:
        print("📂 Loading trained models...")
        scaler = joblib.load("models/scaler.pkl")
        logreg_model = joblib.load("models/logreg_model.pkl")
        rf_model = joblib.load("models/rf_model.pkl")
        
        print("✅ Models loaded successfully!")
        return scaler, logreg_model, rf_model
    
    except FileNotFoundError as e:
        print(f"❌ Error loading models: {e}")
        print("Please run train_script.py first!")
        return None, None, None

def load_and_preprocess_data():
    """Load and preprocess data - same as training"""
    print("Loading and preprocessing data...")
    
    try:
        path = "data/"
        train_data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))
        
        # Remove 'Time' column
        train_data = train_data.drop(columns=["Time"])
        
        # Features and target
        X = train_data.drop(columns=["Class"])
        y = train_data["Class"]
        
        return X, y, train_data
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def prepare_validation_data(X, y, scaler):
    """Prepare validation data with same split as training"""
    print("Preparing validation data...")
    
    # Apply scaler to Amount column
    X_scaled = X.copy()
    X_scaled["Amount"] = scaler.transform(X[["Amount"]])
    
    # Same split as training
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"Validation set: {X_val.shape}")
    print(f"Fraud cases: {y_val.sum()} ({y_val.mean()*100:.2f}%)")
    
    return X_val, y_val

def comprehensive_evaluation(logreg_model, rf_model, X_val, y_val):
    """Comprehensive model evaluation with all metrics"""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)
    
    models = [logreg_model, rf_model]
    model_names = ["Logistic Regression", "Random Forest"]
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}:")
        print("-" * 40)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Print results
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Detailed classification report
        print(f"\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Normal', 'Fraud']))
    
    return results

def create_roc_comparison(results, y_val):
    """Create ROC curve comparison - exactly like your notebook"""
    print("📈 Creating ROC curve comparison...")
    
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        fpr, tpr, _ = roc_curve(y_val, data['y_pred_proba'])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison - All Models")
    plt.legend()
    plt.grid(True)
    plt.savefig('evaluation/roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_precision_recall_comparison(results, y_val):
    """Create Precision-Recall curve comparison - exactly like your notebook"""
    print("📊 Creating Precision-Recall curve comparison...")
    
    plt.figure(figsize=(10, 8))
    
    for name, data in results.items():
        precision, recall, _ = precision_recall_curve(y_val, data['y_pred_proba'])
        ap = average_precision_score(y_val, data['y_pred_proba'])
        plt.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison - All Models")
    plt.grid(True)
    plt.legend()
    plt.savefig('evaluation/pr_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def threshold_analysis(results, y_val):
    """Threshold analysis for both models - exactly like your notebook"""
    print("🎯 Performing threshold analysis...")
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    optimal_thresholds = {}
    
    for name, data in results.items():
        print(f"\n📊 Threshold analysis for {name}:")
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for t in thresholds:
            y_pred = (data['y_pred_proba'] >= t).astype(int)
            precisions.append(precision_score(y_val, y_pred, zero_division=0))
            recalls.append(recall_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred))
        
        # Find optimal threshold
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        optimal_thresholds[name] = optimal_threshold
        print(f"🎯 Optimal Threshold: {optimal_threshold:.2f}")
        
        # Plot threshold analysis
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1_scores, label="F1 Score")
        plt.axvline(optimal_threshold, color='red', linestyle='--', 
                   label=f'Optimal ({optimal_threshold:.2f})')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Precision, Recall & F1 Score vs. Threshold ({name})")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        filename = f"threshold_analysis_{name.lower().replace(' ', '_')}.png"
        plt.savefig(f'evaluation/{filename}', dpi=150, bbox_inches='tight')
        plt.show()
    
    return optimal_thresholds

def create_confusion_matrices(results, y_val, optimal_thresholds):
    """Create confusion matrices with optimal thresholds - exactly like your notebook"""
    print("📋 Creating confusion matrices...")
    
    for name, data in results.items():
        optimal_threshold = optimal_thresholds[name]
        
        # Predictions with optimal threshold
        y_pred_optimal = (data['y_pred_proba'] >= optimal_threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_optimal, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
        
        plt.figure(figsize=(6, 5))
        disp.plot()
        plt.title(f"Confusion Matrix ({name}, Optimal Threshold {optimal_threshold:.2f})")
        
        # Save plot
        filename = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
        plt.savefig(f'evaluation/{filename}', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print confusion matrix details
        print(f"\n📊 Confusion Matrix for {name} (threshold {optimal_threshold:.2f}):")
        print(f"True Negatives:  {cm[0,0]:5d}")
        print(f"False Positives: {cm[0,1]:5d}")
        print(f"False Negatives: {cm[1,0]:5d}")
        print(f"True Positives:  {cm[1,1]:5d}")

def model_comparison_summary(results):
    """Print comprehensive model comparison"""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    # Create comparison table
    print(f"{'Model':<20} {'ROC-AUC':<8} {'Accuracy':<8} {'Precision':<10} {'Recall':<8} {'F1':<8}")
    print("-" * 80)
    
    best_model = None
    best_auc = 0
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['roc_auc']:<8.4f} {metrics['accuracy']:<8.4f} "
              f"{metrics['precision']:<10.4f} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f}")
        
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_model = name
    
    print("-" * 80)
    print(f"Best Model: {best_model} (ROC-AUC: {best_auc:.4f})")
    
    return best_model

def check_model_sizes():
    """Check model file sizes"""
    print("\n📁 Model Size Check:")
    print("-" * 30)
    
    files = ["models/logreg_model.pkl", "models/rf_model.pkl", "models/scaler.pkl"]
    total_size = 0
    
    for file in files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            total_size += size_mb
            filename = os.path.basename(file)
            print(f"{filename:<18}: {size_mb:.2f} MB")
    
    print("-" * 30)
    print(f"{'Total':<18}: {total_size:.2f} MB")
    
    if total_size > 50:
        print("⚠️  WARNING: Total size exceeds 50MB limit!")
    else:
        print("✅ Total size is within 50MB limit")

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("📊 FRAUD DETECTION MODEL EVALUATION")
    print("=" * 60)
    
    # Create evaluation directory
    os.makedirs('evaluation', exist_ok=True)
    
    # Load models
    scaler, logreg_model, rf_model = load_models()
    if scaler is None:
        return
    
    # Load and preprocess data
    X, y, train_data = load_and_preprocess_data()
    if X is None:
        return
    
    # Prepare validation data
    X_val, y_val = prepare_validation_data(X, y, scaler)
    
    # Comprehensive evaluation
    results = comprehensive_evaluation(logreg_model, rf_model, X_val, y_val)
    
    # Create all visualizations
    print("\n" + "=" * 60)
    print("📈 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    create_roc_comparison(results, y_val)
    create_precision_recall_comparison(results, y_val)
    
    # Threshold analysis
    optimal_thresholds = threshold_analysis(results, y_val)
    
    # Confusion matrices
    create_confusion_matrices(results, y_val, optimal_thresholds)
    
    # Final comparison
    print("\n" + "=" * 60)
    best_model = model_comparison_summary(results)
    
    # Model size check
    print("\n" + "=" * 60)
    check_model_sizes()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best performing model: {best_model}")
    print("📊 All visualizations saved in evaluation/ directory:")
    print("- roc_comparison.png")
    print("- pr_comparison.png")
    print("- threshold_analysis_*.png")
    print("- confusion_matrix_*.png")
    print("=" * 60)

if __name__ == "__main__":
    main()