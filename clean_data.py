
import pandas as pd

# Load original dataset
df = pd.read_csv("customer_churn_clean_data.csv")

# Remove completely empty rows
df = df.dropna(how="all")

# Save cleaned dataset
df.to_csv("customer_churn_cleaned.csv", index=False)

print("Dataset cleaned successfully!")
print("Cleaned dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())
