import pandas as pd

# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)

# Create engineered features

# 1. Support friction
df["Support_Friction"] = (
    df["Support Calls"] * df["Payment Delay"]
)

# 2. Usage intensity
df["Usage_Intensity"] = (
    df["Usage Frequency"] / df["Tenure"]
)

# 3. Inactivity indicator
df["Inactivity_Flag"] = (
    df["Last Interaction"] >= 20
).astype(int)

# 4. Customer value category
df["Customer_Value"] = pd.cut(
    df["Total Spend"],
    bins=[0, 300, 500, 700, 1000],
    labels=["Low", "Medium", "High", "Very High"],
    include_lowest=True
)

# Save engineered dataset
df.to_csv("customer_churn_engineered.csv", index=False)

print("\nFeature engineering completed!")
print("New dataset shape:", df.shape)

print("\nNew features:")
print(df[[
    "Support_Friction",
    "Usage_Intensity",
    "Inactivity_Flag",
    "Customer_Value"
]].head())

print("\nEngineered dataset saved successfully!")
# Prepare categorical features for machine learning

df_encoded = pd.get_dummies(
    df,
    columns=["Gender", "Subscription Type", "Contract Length", "Customer_Value"],
    drop_first=True,
    dtype=int
)

df_encoded.to_csv(
    "customer_churn_engineered_encoded.csv",
    index=False
)

print("\nEncoded dataset shape:", df_encoded.shape)
print("\nEncoded dataset saved successfully!")
