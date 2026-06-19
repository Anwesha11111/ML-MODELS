import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

np.random.seed(42)

def load_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    return df, data.target_names

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }
    print(f"\n{name}")
    print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))
    return metrics, y_proba

def main():
    df, target_names = load_data()
    df.to_csv("disease_data.csv", index=False)

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        metrics, proba = evaluate_model(name, model, X_test_scaled, y_test)
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="roc_auc")
        metrics["CV_ROC_AUC_Mean"] = cv_scores.mean()
        results.append(metrics)
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv("model_comparison.csv", index=False)
    print("\nModel Comparison")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison - Disease Prediction Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    y_pred_best = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    if best_name in ["Random Forest", "XGBoost"]:
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title(f"Top 15 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 10))
    corr = df.corr()
    top_features = corr["diagnosis"].abs().sort_values(ascending=False).head(11).index
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap - Top Features")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.close()

    print(f"\nBest performing model: {best_name}")

if __name__ == "__main__":
    main()
