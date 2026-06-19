# CodeAlpha Task 4: Disease Prediction from Medical Data

Predicts the likelihood of disease (breast cancer diagnosis: malignant vs benign) from structured medical data.

## How it works
- Uses the Breast Cancer Wisconsin diagnostic dataset (UCI ML Repository, bundled in scikit-learn)
- Trains and compares Logistic Regression, SVM, Random Forest, and XGBoost
- Evaluates using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and 5-fold cross-validated ROC-AUC
- Saves ROC curve, confusion matrix, feature importance, and correlation heatmap plots

## Run
```
pip install -r requirements.txt
python disease_prediction.py
```

## Outputs
- disease_data.csv — dataset used
- model_comparison.csv — metrics for all models
- roc_curves.png, confusion_matrix.png, feature_importance.png, correlation_heatmap.png
