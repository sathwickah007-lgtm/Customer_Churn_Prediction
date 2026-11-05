
import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("customer_churn_cleaned.csv")

# Count churn and non-churn customers
churn_counts = df["Churn"].value_counts().sort_index()

# Create bar chart
plt.figure(figsize=(7, 5))

plt.bar(
    ["Not Churned", "Churned"],
    churn_counts.values
)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("churn_distribution.png")
plt.show()

# Churn rate by contract length
contract_churn = (
    df.groupby("Contract Length")["Churn"].mean() * 100
)

# Arrange contracts in order
contract_churn = contract_churn.reindex(
    ["Monthly", "Quarterly", "Annual"]
)

print("\nChurn rate by contract length:")
print(contract_churn)

# Create bar chart
plt.figure(figsize=(8, 5))

bars = plt.bar(
    contract_churn.index,
    contract_churn.values
)

plt.title("Churn Rate by Contract Length")
plt.xlabel("Contract Length")
plt.ylabel("Churn Rate (%)")

# Display percentages above bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{height:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("churn_by_contract.png")
plt.show()
# Verify churn counts by contract length
print("\nContract length vs Churn:")
print(pd.crosstab(
    df["Contract Length"],
    df["Churn"],
    margins=True
))

# Detailed churn analysis by contract length
contract_analysis = df.groupby("Contract Length")["Churn"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Average_Churn_Rate="mean"
)

contract_analysis["Average_Churn_Rate"] *= 100

print("\nDetailed Contract Analysis:")
print(contract_analysis.to_string())
# Check churn rate by number of support calls
support_churn = (
    df.groupby("Support Calls")["Churn"].agg(
        Total_Customers="count",
        Churned_Customers="sum",
        Churn_Rate="mean"
    )
)

support_churn["Churn_Rate"] *= 100

print("\nChurn analysis by support calls:")
print(support_churn.to_string())

# Check churn rate by number of support calls

support_churn = (
    df.groupby("Support Calls")["Churn"].agg(
        Total_Customers="count",
        Churned_Customers="sum",
        Churn_Rate="mean"
    )
)

support_churn["Churn_Rate"] *= 100

print("\nChurn analysis by support calls:")
print(support_churn.to_string())

# Contract Length vs Support Calls

contract_support = pd.crosstab(
    df["Contract Length"],
    df["Support Calls"],
    margins=True
)

print("\nContract Length vs Support Calls:")
print(contract_support.to_string())

# Churn analysis by Payment Delay

payment_churn = df.groupby("Payment Delay")["Churn"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Churn_Rate="mean"
)

payment_churn["Churn_Rate"] *= 100

print("\nChurn analysis by Payment Delay:")
print(payment_churn.to_string())
# Churn analysis by Last Interaction

interaction_churn = df.groupby("Last Interaction")["Churn"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Churn_Rate="mean"
)

interaction_churn["Churn_Rate"] *= 100

print("\nChurn analysis by Last Interaction:")
print(interaction_churn.to_string())
# Churn analysis by Tenure

tenure_churn = df.groupby("Tenure")["Churn"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Churn_Rate="mean"
)

tenure_churn["Churn_Rate"] *= 100

print("\nChurn analysis by Tenure:")
print(tenure_churn.to_string())

# Churn analysis by Total Spend

df["Spend Group"] = pd.cut(
    df["Total Spend"],
    bins=[0, 300, 500, 700, 1000],
    labels=["Low", "Medium", "High", "Very High"]
)

spend_churn = df.groupby(
    "Spend Group", observed=False
)["Churn"].agg(
    Total_Customers="count",
    Churned_Customers="sum",
    Churn_Rate="mean"
)

spend_churn["Churn_Rate"] *= 100

print("\nChurn analysis by Total Spend:")
print(spend_churn.to_string())

# Checking suspicious churn patterns

print("\nChecking suspicious patterns:")

print("\nChurn rate by contract length:")
print(df.groupby("Contract Length")["Churn"].mean() * 100)

print("\nChurn rate by payment delay above 20:")
print(df[df["Payment Delay"] > 20]["Churn"].value_counts())

print("\nChurn rate by total spend below 300:")
print(df[df["Total Spend"] < 300]["Churn"].value_counts())

print("\nChurn rate by support calls 6 or more:")
print(df[df["Support Calls"] >= 6]["Churn"].value_counts())
print("\nChurn by contract length among customers with zero support calls:")

zero_calls = df[df["Support Calls"] == 0]

print(
    zero_calls.groupby("Contract Length")["Churn"].mean() * 100
)
# Checking overlap between suspicious conditions

suspicious_customers = df[
    (df["Contract Length"] == "Monthly") &
    (df["Support Calls"] >= 6)
]

print("\nMonthly contract AND 6+ support calls:")
print("Number of customers:", len(suspicious_customers))
print("Churned customers:", suspicious_customers["Churn"].sum())
# Check identical feature combinations with conflicting churn labels

features = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction"
]

combination_check = df.groupby(features)["Churn"].nunique()

print("\nIdentical feature combinations with different churn labels:")
print((combination_check > 1).sum())