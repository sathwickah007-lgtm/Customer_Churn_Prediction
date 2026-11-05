
import pandas as pd
import joblib

# Load the saved model bundle
bundle = joblib.load("churn_model_bundle.joblib")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
threshold = bundle["threshold"]

print("Saved churn model loaded successfully!")

# Load new customer data
df = pd.read_csv("new_customers.csv")

print("New customer dataset loaded:", df.shape)

# Feature engineering
df["Usage_Intensity"] = (
    df["Usage Frequency"] /
    df["Tenure"].replace(0, 1)
)

df["Inactivity_Flag"] = (
    df["Last Interaction"] >= 20
).astype(int)

# Encode categorical features
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Subscription Type"],
    drop_first=True,
    dtype=int
)

# Match the exact training features
X_new = df_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# Generate predictions
churn_probabilities = model.predict_proba(X_new)[:, 1]

predictions = (
    churn_probabilities >= threshold
).astype(int)

# Prepare output
results = pd.DataFrame({
    "CustomerID": df["CustomerID"],
    "Churn_Probability": churn_probabilities,
    "Predicted_Churn": predictions
})

results["Churn_Status"] = results["Predicted_Churn"].map({
    0: "Not Likely to Churn",
    1: "Likely to Churn"
})

# Save predictions
results.to_csv(
    "new_customer_churn_results.csv",
    index=False
)

print("\n========== NEW CUSTOMER PREDICTIONS ==========")
print(results)

print("\nPredictions saved successfully!")
