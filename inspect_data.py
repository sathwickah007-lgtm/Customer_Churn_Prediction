import pandas as pd

# Load the customer churn dataset
df = pd.read_csv("customer_churn_clean_data.csv")

# Display the dataset size
print("Dataset shape:", df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display missing values
print("\nMissing values:")
print(df.isnull().sum())
# Investigate missing-value rows
print("\nRows with any missing values:")
print(df[df.isnull().any(axis=1)])

print("\nCompletely empty rows:")
print(df.isnull().all(axis=1).sum())
# Remove completely empty rows
df = df.dropna(how="all")

# Check the updated dataset
print("\nDataset shape after cleaning:", df.shape)

# Check missing values again
print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# Check duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# Check duplicate Customer IDs
print("Duplicate Customer IDs:", df["CustomerID"].duplicated().sum())
# Check data types
print("\nColumn data types:")
print(df.dtypes)
# Statistical summary
print("\nStatistical summary:")
print(df.describe())
# Count churned and non-churned customers
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Calculate percentages
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Inspect categorical columns
categorical_columns = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

for column in categorical_columns:
    print(f"\n{column} categories:")
    print(df[column].value_counts())

# Check minimum and maximum values
numerical_columns = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

print("\nMinimum values:")
print(df[numerical_columns].min())

print("\nMaximum values:")
print(df[numerical_columns].max())
    