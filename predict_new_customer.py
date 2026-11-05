
import pandas as pd
import joblib

# Load saved model
bundle = joblib.load("churn_model_bundle.joblib")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
threshold = bundle["threshold"]

# Load new customer data
df = pd.read_csv("new_customers.csv")

# Apply feature engineering
df["Usage_Intensity"] = (
    df["Usage Frequency"] / df["Tenure"].replace(0, 1)
)

df["Inactivity_Flag"] = (
    df["Last Interaction"] >= 20
).astype(int)

# Encode categorical columns
df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Subscription Type"],
    drop_first=True,
    dtype=int
)

# Ensure all training features are present
X_new = df_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

# Predict churn
probabilities = model.predict_proba(X_new)[:, 1]

predictions = (probabilities >= threshold).astype(int)

# Save results
results = pd.DataFrame({
    "CustomerID": df["CustomerID"],
    "Churn_Probability": probabilities,
    "Predicted_Churn": predictions
})

results.to_csv("new_customer_churn_results.csv", index=False)

print("New customer predictions completed!")
print(results.head(10))
