import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

# Load engineered dataset
df = pd.read_csv("customer_churn_engineered_encoded.csv")

print("Engineered dataset loaded:", df.shape)

# Separate features and target
X = df.drop(
    ["CustomerID", "Churn"],
    axis=1
)

y = df["Churn"]

# Remove suspicious features identified during EDA
suspicious_features = [
    "Contract Length_Monthly",
    "Contract Length_Quarterly",
    "Support Calls",
    "Payment Delay",
    "Total Spend"
]

X = X.drop(columns=suspicious_features)

# Remove engineered features derived from suspicious columns
derived_suspicious_features = [
    "Support_Friction",
    "Customer_Value_Medium",
    "Customer_Value_High",
    "Customer_Value_Very High"
]

X = X.drop(columns=derived_suspicious_features)

print("\nStrict leakage-aware feature shape:", X.shape)
print("Strict features:", list(X.columns))

print("\nFeatures used:", list(X.columns))
print("Feature shape:", X.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale features for Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# THRESHOLD OPTIMIZATION

from sklearn.metrics import precision_score, recall_score, f1_score

threshold_results = []

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    threshold_pred = (rf_prob >= threshold).astype(int)

    precision = precision_score(y_test, threshold_pred)
    recall = recall_score(y_test, threshold_pred)
    f1 = f1_score(y_test, threshold_pred)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n========== THRESHOLD OPTIMIZATION ==========")
print(threshold_df)

threshold_df.to_csv("threshold_comparison.csv", index=False)

print("\nThreshold comparison saved successfully!")


# COST-BASED EVALUATION

from sklearn.metrics import confusion_matrix

# Illustrative business costs (adjust when actual costs are known)
FALSE_POSITIVE_COST = 1
FALSE_NEGATIVE_COST = 5

cost_results = []

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:

    predictions = (rf_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    total_cost = (
        fp * FALSE_POSITIVE_COST +
        fn * FALSE_NEGATIVE_COST
    )

    cost_results.append({
        "Threshold": threshold,
        "False Positives": fp,
        "False Negatives": fn,
        "Total Cost": total_cost
    })

cost_df = pd.DataFrame(cost_results)

print("\n========== COST-BASED EVALUATION ==========")
print(cost_df)

cost_df.to_csv("cost_analysis.csv", index=False)

print("\nCost analysis saved successfully!")

# Calculate class imbalance weight
scale_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

print("\nClass imbalance weight:", scale_weight)

# Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    scale_pos_weight=scale_weight
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost model trained successfully!")

# Train LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    scale_pos_weight=scale_weight
)

lgbm_model.fit(X_train, y_train)

lgbm_pred = lgbm_model.predict(X_test)
lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]

print("\nLightGBM model trained successfully!")

# Evaluate models
results = []

for name, predictions, probabilities in [
    ("Logistic Regression", lr_pred, lr_prob),
    ("Random Forest", rf_pred, rf_prob),
    ("XGBoost", xgb_pred, xgb_prob),
    ("LightGBM", lgbm_pred, lgbm_prob)
]:
    accuracy = accuracy_score(y_test, predictions)
    roc = roc_auc_score(y_test, probabilities)
    pr = average_precision_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "ROC-AUC": roc,
        "PR-AUC": pr
    })

    print(f"\n========== {name} ==========")
    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc)
    print("PR-AUC:", pr)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Save comparison
comparison = pd.DataFrame(results)

print("\n========== ENGINEERED MODEL COMPARISON ==========")
print(comparison)

comparison.to_csv(
    "engineered_model_comparison.csv",
    index=False
)

print("\nEngineered model comparison saved successfully!")

# ENGINEERED RANDOM FOREST FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== ENGINEERED FEATURE IMPORTANCE ==========")
print(feature_importance)

feature_importance.to_csv(
    "engineered_feature_importance.csv",
    index=False
)

# Plot feature importance
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Engineered Model - Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("engineered_feature_importance.png", dpi=300)
plt.show()

print("\nEngineered feature importance saved successfully!")


# MODEL SAVING AND BATCH PREDICTION

# Save reusable model bundle
model_bundle = {
    "model": rf_model,
    "feature_columns": list(X.columns),
    "threshold": 0.3
}

joblib.dump(model_bundle, "churn_model_bundle.joblib")

print("\nTrained model saved successfully!")

# Generate batch predictions for test customers
batch_predictions = (rf_prob >= 0.3).astype(int)

prediction_output = pd.DataFrame({
    "CustomerID": df.loc[X_test.index, "CustomerID"],
    "Actual_Churn": y_test,
    "Churn_Probability": rf_prob,
    "Predicted_Churn": batch_predictions
})

prediction_output.to_csv(
    "batch_churn_predictions.csv",
    index=False
)

print("\nBatch predictions generated successfully!")
print("Total customers scored:", len(prediction_output))
print("\nPrediction sample:")
print(prediction_output.head(10))

print("\nBatch prediction file saved successfully!")
