# Data-Science-Project

# Predictive Maintenance and Fraud Detection - Data Science Assignment

This repository contains solutions for two data science problems:  
1. Predictive Maintenance in Manufacturing (Regression)  
2. Fraud Detection in Banking (Clustering + Classification)

Both problems include data preprocessing, feature engineering, model implementation, evaluation, and discussion of results.

---

## **Problem 1: Predictive Maintenance in Manufacturing**

**Objective:**  
Predict days until machinery failure using sensor data (temperature, vibration, pressure, runtime hours) and reduce downtime costs.

**Steps Taken:**
1. **Data Understanding**  
   - Loaded dataset and explored features/target.
   - Checked for missing values and basic statistics.
   
2. **Overfitting Explanation**  
   - Initial linear regression overfits due to noise in sensor readings.
   - Bias-Variance tradeoff discussed: linear regression has high bias but low variance, overfitting occurs when variance dominates.

3. **Model Implementation**
   - Used **Random Forest Regressor** to reduce variance and improve generalization.
   - Train/test split (80/20) and k-fold cross-validation (k=5).
   - Predicted on unseen test data.

4. **Evaluation**
   - Metrics computed: RMSE and R².
   - Compared with linear regression baseline.
   - Demonstrated how ensembles reduce overfitting.

5. **Feature Engineering**
   - Added interaction term (`vibration * runtime`) and polynomial features.
   - Discussed applications like anomaly detection in maintenance logs.

**Files:**  
- `problem1_predictive_maintenance.ipynb` - full Python code and plots  
- `Question1_dataset.csv` - synthetic dataset  

---

## **Problem 2: Fraud Detection in Banking**

**Objective:**  
Detect fraudulent transactions using labeled and unlabeled transaction data (amount, time, location, merchant type).

**Steps Taken:**

1. **Unsupervised Clustering**
   - Used **K-Means** to cluster unlabeled transactions and detect anomalies.
   - Features scaled with StandardScaler; categorical features encoded with One-Hot Encoding.
   - Elbow method applied to choose optimal k; clusters visualized with PCA.

2. **Supervised Classification**
   - Labeled data used to train **Random Forest** (and optionally Naive Bayes).
   - Bias-Variance tradeoff discussed: Random Forest reduces variance, Naive Bayes has high bias but low variance.
   - Train/test split and optional SMOTE oversampling for imbalanced classes.

3. **Feature Engineering**
   - Added features: `log_amount`, `is_night`, `amount deviation from merchant mean`, `high_amount_flag`, `cluster_distance`, `merchant/location fraud rate`.
   - Showed how these features reduce variance and help detect fraud patterns.

4. **Evaluation**
   - 10-fold cross-validation F1-score, confusion matrix, precision-recall curves.
   - Threshold tuning to optimize F1.
   - Discussed comparison between unsupervised clustering and supervised classification.

**Files:**  
- `problem2_fraud_detection.ipynb` - full Python code, plots, and threshold tuning  
- `Question2_dataset.csv` - synthetic dataset  

---

## **Instructions to Run**

1. Clone the repository:
```bash
git clone <your-repo-link>
