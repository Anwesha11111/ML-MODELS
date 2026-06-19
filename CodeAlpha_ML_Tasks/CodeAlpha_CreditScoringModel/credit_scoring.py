import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

np.random.seed(42)

def generate_dataset(n_samples=3000):
    age = np.random.randint(21, 65, n_samples)
    annual_income = np.random.normal(60000, 25000, n_samples).clip(12000, 250000)
    total_debt = np.random.normal(15000, 12000, n_samples).clip(0, 150000)
    monthly_expenses = np.random.normal(2000, 800, n_samples).clip(300, 8000)
    num_credit_lines = np.random.randint(0, 15, n_samples)
    num_late_payments = np.random.poisson(1.2, n_samples).clip(0, 20)
    credit_history_years = np.random.randint(0, 30, n_samples)
    num_bankruptcies = np.random.binomial(1, 0.06, n_samples)
    loan_amount_requested = np.random.normal(20000, 15000, n_samples).clip(1000, 100000)
    employment_years = np.random.randint(0, 35, n_samples)
    savings_balance = np.random.normal(10000, 9000, n_samples).clip(0, 100000)

    debt_to_income = total_debt / (annual_income + 1)
    expense_to_income = (monthly_expenses * 12) / (annual_income + 1)

    risk_score = (
        - 0.35 * debt_to_income
        - 0.25 * expense_to_income
        - 0.15 * num_late_payments
        - 0.6 * num_bankruptcies
        + 0.05 * credit_history_years
        + 0.000015 * annual_income
        + 0.00006 * savings_balance
        + 0.03 * employment_years
        - 0.00002 * loan_amount_requested
        - 0.05 * num_credit_lines
        + np.random.normal(0, 0.6, n_samples)
    )

    threshold = np.percentile(risk_score, 35)
    creditworthy = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "total_debt": total_debt.round(2),
        "monthly_expenses": monthly_expenses.round(2),
        "num_credit_lines": num_credit_lines,
        "num_late_payments": num_late_payments,
        "credit_history_years": credit_history_years,
        "num_bankruptcies": num_bankruptcies,
        "loan_amount_requested": loan_amount_requested.round(2),
        "employment_years": employment_years,
        "savings_balance": savings_balance.round(2),
        "debt_to_income_ratio": debt_to_income.round(4),
        "expense_to_income_ratio": expense_to_income.round(4),
        "creditworthy": creditworthy
    })
    return df

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
    print(classification_report(y_test, y_pred, target_names=["Not Creditworthy", "Creditworthy"]))
    return metrics, y_proba

def main():
    df = generate_dataset()
    df.to_csv("credit_data.csv", index=False)

    X = df.drop(columns=["creditworthy"])
    y = df["creditworthy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    }

    results = []
    roc_data = {}

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            metrics, proba = evaluate_model(name, model, X_test_scaled, y_test)
        else:
            model.fit(X_train, y_train)
            metrics, proba = evaluate_model(name, model, X_test, y_test)
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
    plt.title("ROC Curve Comparison - Credit Scoring Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=150)
    plt.close()

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    X_eval = X_test_scaled if best_name == "Logistic Regression" else X_test
    y_pred_best = best_model.predict(X_eval)
    cm = confusion_matrix(y_test, y_pred_best)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Creditworthy", "Creditworthy"],
                yticklabels=["Not Creditworthy", "Creditworthy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    if best_name == "Random Forest":
        importances = pd.Series(best_model.feature_importances_, index=X.columns)
        importances = importances.sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances.values, y=importances.index)
        plt.title("Feature Importance - Random Forest")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        plt.close()

    print(f"\nBest performing model: {best_name}")

if __name__ == "__main__":
    main()
