import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("customer_churn_cleaned.csv")

df["Tenure_Group"] = pd.cut(
    df["Tenure"],
    bins=[0, 6, 12, 24, 36, 48, 60],
    labels=["1-6", "7-12", "13-24", "25-36", "37-48", "49-60"]
)

retention = df.groupby("Tenure_Group", observed=False)["Churn"].agg(
    Total_Customers="count",
    Churn_Rate="mean"
).reset_index()

retention["Retention_Rate"] = 1 - retention["Churn_Rate"]

retention.to_csv("retention_analysis.csv", index=False)

plt.figure(figsize=(9, 5))
plt.bar(retention["Tenure_Group"].astype(str),
        retention["Retention_Rate"] * 100)

plt.xlabel("Customer Tenure (Months)")
plt.ylabel("Retention Rate (%)")
plt.title("Tenure-Based Customer Retention Analysis")
plt.tight_layout()
plt.savefig("retention_analysis.png", dpi=150)
plt.close()

print("Retention analysis completed!")
print(retention)
print("CSV and chart saved successfully!")