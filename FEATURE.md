# Machine Learning Programming Project (SS 2026) - Feature Specification

## 1. Project Overview
This project focuses on solving a multi-label classification problem using the **MNIST dataset**, which consists of 70,000 instances of handwritten digits (0-9) in a $28 \times 28$ pixel grayscale format. The objective is to design, implement, and rigorously evaluate machine learning models capable of accurately classifying these images while adhering to strict resource constraints.

## 2. Core Machine Learning Models

### A. Convolutional Neural Network (CNN)
* **Architecture:** Based on the LeNet-5 architecture.
* **Framework:** PyTorch.
* **Rationale:** Chosen for its natural fit for image classification tasks. Convolutional layers explicitly model the spatial relationships of neighboring pixels and provide translational pseudo-invariance through shared kernel weights.
* **Optimizer:** AdamW optimizer for 10 epochs.

### B. Support Vector Machine (SVM) Ensemble
* **Architecture:** Ensemble of 10 binary SVM classifiers (One-vs-Rest strategy for multi-label classification).
* **Framework:** scikit-learn (`sklearn.svm.SVC`).
* **Rationale:** Implemented to compare classical ML performance against deep learning. Suitable for non-linear classification via the "kernel trick."

## 3. Data Preprocessing Pipeline
To ensure optimal model performance and unbiased hyperparameter tuning, the following preprocessing steps are applied:
* **Data Splitting:** * Training Set: 50,000 images
    * Validation Set: 10,000 images (split from the original training set)
    * Test Set: 10,000 images (standard MNIST test split)
* **Normalization:** Raw pixel values are scaled by $\frac{1}{255}$ to obtain features in the range [0, 1].
* **Padding (CNN specific):** Symmetrical padding with zero values from $28 \times 28$ to $32 \times 32$. This ensures the center of the convolution in the first layer sweeps over all digit features.
* **Whitening (CNN specific):** Applied with a mean of 0.1307 and a standard deviation of 0.3081.

## 4. Hyperparameter Optimization Framework
A principled **Random Search** strategy is implemented to find the optimal configurations for both classifiers:
* **Methodology:** 60 trials per classifier, uniformly sampling configurations from defined bounds.
* **Evaluation Metric:** Validation set accuracy.
* **CNN Search Space:** * Batch size ($B$): [8, 128]
    * Learning rate ($l_R$): [0.0001, 0.1]
    * Hidden layer sizes ($n_1, n_2$): [32, 256]
    * Activation functions: {ReLU, Sigmoid}
* **SVM Search Space:** * Kernel choice: {Linear, Poly, RBF, Sigmoid}
    * Regularization parameter ($C$): $10^x$ where $x \in [-2, 3]$
    * Kernel coefficient ($\gamma$): $10^y$ where $y \in [-4, 0]$
    * *Note:* SVM hyperparameter search is conducted on a randomly sampled 10,000-instance subset to optimize computational overhead.

## 5. Performance Metrics
The final models demonstrate highly competitive performance on the hold-out test dataset:
* **CNN Accuracy:** 98.69%
* **SVM Accuracy:** 98.49%
* **Detailed Analytics:** Precision and recall metrics are captured for every individual digit class (0-9) to monitor data imbalance or specific digit misclassifications (e.g., distinguishing between 3s and 5s).

## 6. Technical Constraints & Deliverables
* **Model Size Limitation:** The serialized model parameters (weights) must be strictly under **50 MB**.
* **Inference Readiness:** Models must be capable of being stored and loaded from a file for immediate evaluation without requiring retraining.
* **Code Transparency:** The complete, runnable training and evaluation scripts are structured for deployment on JupyterHub, allowing full reproducibility.
