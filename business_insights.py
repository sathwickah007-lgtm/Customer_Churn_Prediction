import pandas as pd

df = pd.read_csv("customer_churn_cleaned.csv")

# Group customers by risk-related factors
df["Payment_Delay_Group"] = pd.cut(
    df["Payment Delay"],
    bins=[-1, 5, 10, 20, 30],
    labels=["0-5", "6-10", "11-20", "21-30"]
)

df["Support_Call_Group"] = pd.cut(
    df["Support Calls"],
    bins=[-1, 2, 4, 10],
    labels=["0-2", "3-4", "5-10"]
)

df["Inactivity_Group"] = pd.cut(
    df["Last Interaction"],
    bins=[0, 7, 15, 30],
    labels=["1-7", "8-15", "16-30"]
)

groups = {
    "Contract Length": "contract_insights.csv",
    "Payment_Delay_Group": "payment_insights.csv",
    "Support_Call_Group": "support_insights.csv",
    "Inactivity_Group": "inactivity_insights.csv"
}

for column, filename in groups.items():
    result = df.groupby(column, observed=False)["Churn"].agg(
        Customers="count",
        Churn_Rate="mean"
    ).reset_index()

    result["Churn_Rate"] *= 100
    result.to_csv(filename, index=False)

print("Business insights generated successfully!")
print("\nChurn by contract:")
print(df.groupby("Contract Length")["Churn"].mean() * 100)

print("\nChurn by payment delay:")
print(df.groupby("Payment_Delay_Group", observed=False)["Churn"].mean() * 100)

print("\nChurn by support calls:")
print(df.groupby("Support_Call_Group", observed=False)["Churn"].mean() * 100)

print("\nChurn by inactivity:")
print(df.groupby("Inactivity_Group", observed=False)["Churn"].mean() * 100)