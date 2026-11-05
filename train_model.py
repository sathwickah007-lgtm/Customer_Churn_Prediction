import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())
# Separate features and target

X = df.drop(["CustomerID", "Churn"], axis=1)

# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["Gender", "Subscription Type", "Contract Length"],
    drop_first=True,
    dtype=int
)
# Save all features for the full-feature experiment
X_full = X.copy()
# Remove suspicious features for leakage-aware experiment
suspicious_features = [
    "Contract Length_Monthly",
    "Contract Length_Quarterly",
    "Support Calls",
    "Payment Delay",
    "Total Spend"
]

X_clean = X_full.drop(columns=suspicious_features)

print("\nLeakage-aware feature shape:", X_clean.shape)
print("Remaining features:", X_clean.columns.tolist())

print("\nEncoded features:")
print(X.head())

print("\nEncoded features shape:", X.shape)
y = df["Churn"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())
# Split dataset into training and testing sets
# Split leakage-aware dataset

X_clean_train, X_clean_test, y_clean_train, y_clean_test = train_test_split(
    X_clean, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nLeakage-aware training shape:", X_clean_train.shape)
print("Leakage-aware testing shape:", X_clean_test.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

# Scale full-feature dataset

scaler_full = StandardScaler()

X_train_scaled = scaler_full.fit_transform(X_train)
X_test_scaled = scaler_full.transform(X_test)


# Scale leakage-aware dataset

scaler_clean = StandardScaler()

X_clean_train_scaled = scaler_clean.fit_transform(X_clean_train)
X_clean_test_scaled = scaler_clean.transform(X_clean_test)

print("\nFeature scaling completed successfully!")
print("Full training data:", X_train_scaled.shape)
print("Leakage-aware training data:", X_clean_train_scaled.shape)

# Train Logistic Regression on full-feature dataset

model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train_scaled, y_train)

print("\nFull-feature Logistic Regression trained!")


# Train Logistic Regression on leakage-aware dataset

model_clean = LogisticRegression(max_iter=1000)
model_clean.fit(X_clean_train_scaled, y_clean_train)

print("Leakage-aware Logistic Regression trained!")

# Predictions on test data

y_pred_full = model_full.predict(X_test_scaled)
y_pred_clean = model_clean.predict(X_clean_test_scaled)


# Evaluate full-feature model

print("\n========== FULL-FEATURE MODEL ==========")

print("Accuracy:", accuracy_score(y_test, y_pred_full))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_full))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_full))


# Evaluate leakage-aware model

print("\n========== LEAKAGE-AWARE MODEL ==========")

print("Accuracy:", accuracy_score(y_clean_test, y_pred_clean))

print("\nClassification Report:")
print(classification_report(y_clean_test, y_pred_clean))

print("\nConfusion Matrix:")
print(confusion_matrix(y_clean_test, y_pred_clean))
# Calculate prediction probabilities

y_prob_full = model_full.predict_proba(X_test_scaled)[:, 1]
y_prob_clean = model_clean.predict_proba(X_clean_test_scaled)[:, 1]


# Evaluate ROC-AUC and PR-AUC

print("\n========== ROC-AUC AND PR-AUC ==========")

print("Full-feature ROC-AUC:",
      roc_auc_score(y_test, y_prob_full))

print("Full-feature PR-AUC:",
      average_precision_score(y_test, y_prob_full))

print("\nLeakage-aware ROC-AUC:",
      roc_auc_score(y_clean_test, y_prob_clean))

print("Leakage-aware PR-AUC:",
      average_precision_score(y_clean_test, y_prob_clean))

# Train Random Forest on full-feature dataset

rf_full = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_full.fit(X_train, y_train)

print("\nFull-feature Random Forest trained!")


# Train Random Forest on leakage-aware dataset

rf_clean = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_clean.fit(X_clean_train, y_clean_train)

print("Leakage-aware Random Forest trained!")
# Random Forest predictions

rf_pred_full = rf_full.predict(X_test)
rf_pred_clean = rf_clean.predict(X_clean_test)

rf_prob_full = rf_full.predict_proba(X_test)[:, 1]
rf_prob_clean = rf_clean.predict_proba(X_clean_test)[:, 1]


# Full-feature Random Forest evaluation

print("\n========== FULL-FEATURE RANDOM FOREST ==========")

print("Accuracy:", accuracy_score(y_test, rf_pred_full))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred_full))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred_full))

print("ROC-AUC:", roc_auc_score(y_test, rf_prob_full))
print("PR-AUC:", average_precision_score(y_test, rf_prob_full))


# Leakage-aware Random Forest evaluation

print("\n========== LEAKAGE-AWARE RANDOM FOREST ==========")

print("Accuracy:", accuracy_score(y_clean_test, rf_pred_clean))

print("\nClassification Report:")
print(classification_report(y_clean_test, rf_pred_clean))

print("\nConfusion Matrix:")
print(confusion_matrix(y_clean_test, rf_pred_clean))

print("ROC-AUC:", roc_auc_score(y_clean_test, rf_prob_clean))
print("PR-AUC:", average_precision_score(y_clean_test, rf_prob_clean))

# Model comparison

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Full-feature Accuracy": [
        accuracy_score(y_test, y_pred_full),
        accuracy_score(y_test, rf_pred_full)
    ],
    "Leakage-aware Accuracy": [
        accuracy_score(y_clean_test, y_pred_clean),
        accuracy_score(y_clean_test, rf_pred_clean)
    ],
    "Full-feature ROC-AUC": [
        roc_auc_score(y_test, y_prob_full),
        roc_auc_score(y_test, rf_prob_full)
    ],
    "Leakage-aware ROC-AUC": [
        roc_auc_score(y_clean_test, y_prob_clean),
        roc_auc_score(y_clean_test, rf_prob_clean)
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(comparison)

comparison.to_csv("model_comparison.csv", index=False)

print("\nModel comparison saved successfully!")

# Feature importance

feature_importance = pd.DataFrame({
    "Feature": X_clean.columns,
    "Importance": rf_clean.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

feature_importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")

# Feature importance visualization

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Customer Churn - Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()

print("\nFeature importance graph saved successfully!")

# SHAP explainability

print("\n========== SHAP EXPLANATION ==========")

explainer = shap.TreeExplainer(rf_clean)

print("SHAP explainer created successfully!")

# Calculate SHAP values

# Calculate SHAP values using a smaller sample

X_shap = X_clean_test.sample(
    n=50,
    random_state=42
)

print("Calculating SHAP values for 50 customers...")

shap_values = explainer.shap_values(X_shap)

print("\nSHAP values calculated successfully!")
print("SHAP values shape:", shap_values.shape)
# SHAP summary visualization

shap_values_churn = shap_values[:, :, 1]

shap.summary_plot(
    shap_values_churn,
    X_shap,
    show=False
)

plt.tight_layout()

plt.savefig("shap_summary.png", dpi=300)

plt.show()

print("\nSHAP summary plot saved successfully!")