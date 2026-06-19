# CodeAlpha Task 1: Credit Scoring Model

Predicts an individual's creditworthiness from financial history using classification algorithms.

## How it works
- Generates a realistic financial dataset (income, debts, payment history, credit lines, bankruptcies, savings, etc.)
- Engineers ratio features (debt-to-income, expense-to-income)
- Trains and compares Logistic Regression, Decision Tree, and Random Forest
- Evaluates using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Saves ROC curve, confusion matrix, and feature importance plots

## Run
```
pip install -r requirements.txt
python credit_scoring.py
```

## Outputs
- credit_data.csv — generated dataset
- model_comparison.csv — metrics for all models
- roc_curves.png, confusion_matrix.png, feature_importance.png
