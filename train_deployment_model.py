
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("customer_churn_engineered_encoded.csv")

X = df.drop(columns=["CustomerID", "Churn"])

suspicious_features = [
    "Contract Length_Monthly",
    "Contract Length_Quarterly",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Support_Friction",
    "Customer_Value_Medium",
    "Customer_Value_High",
    "Customer_Value_Very High"
]

X = X.drop(columns=suspicious_features)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training compact deployment model...")

model = RandomForestClassifier(
    n_estimators=40,
    max_depth=16,
    min_samples_leaf=25,
    max_samples=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

probabilities = model.predict_proba(X_test)[:, 1]

print("\nCOMPACT MODEL RESULTS")
print("Accuracy:", accuracy_score(y_test, probabilities >= 0.5))
print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("PR-AUC:", average_precision_score(y_test, probabilities))

print("\nTHRESHOLD 0.3 RESULTS")
predictions = (probabilities >= 0.3).astype(int)
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions))

bundle = {
    "model": model,
    "feature_columns": list(X.columns),
    "threshold": 0.3
}

joblib.dump(
    bundle,
    "churn_model_bundle_deploy.joblib",
    compress=3
)

print("\nCompact model saved successfully!")

import os
print(
    "Model size:",
    round(os.path.getsize("churn_model_bundle_deploy.joblib") / (1024 * 1024), 2),
    "MB"
)
