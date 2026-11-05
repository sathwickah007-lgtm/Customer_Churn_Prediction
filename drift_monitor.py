import pandas as pd
import numpy as np

# Load reference and incoming customer data
training = pd.read_csv("customer_churn_cleaned.csv")
incoming = pd.read_csv("new_customers.csv")

numeric_features = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Last Interaction"
]

categorical_features = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

results = []

# Numeric drift using Population Stability Index (PSI)
for column in numeric_features:
    reference = training[column].dropna()
    current = incoming[column].dropna()

    _, bins = pd.qcut(
        reference,
        q=10,
        retbins=True,
        duplicates="drop"
    )

    bins[0] = -np.inf
    bins[-1] = np.inf

    reference_counts = pd.cut(reference, bins=bins).value_counts(
        normalize=True, sort=False
    )

    current_counts = pd.cut(current, bins=bins).value_counts(
        normalize=True, sort=False
    )

    expected = reference_counts.values + 0.0001
    actual = current_counts.values + 0.0001

    psi = np.sum((actual - expected) * np.log(actual / expected))

    results.append({
        "Feature": column,
        "Drift_Score": round(psi, 4),
        "Drift_Type": "Numeric",
        "Status": "Potential Drift" if psi >= 0.2 else "No Significant Drift"
    })

# Categorical distribution comparison
for column in categorical_features:
    reference = training[column].value_counts(normalize=True)
    current = incoming[column].value_counts(normalize=True)

    categories = reference.index.union(current.index)

    difference = (
        current.reindex(categories, fill_value=0)
        - reference.reindex(categories, fill_value=0)
    ).abs().sum() / 2

    results.append({
        "Feature": column,
        "Drift_Score": round(difference, 4),
        "Drift_Type": "Categorical",
        "Status": "Potential Drift" if difference >= 0.2 else "No Significant Drift"
    })

drift_df = pd.DataFrame(results)
drift_df.to_csv("drift_report.csv", index=False)

print("Drift monitoring completed!")
print(drift_df.to_string(index=False))
print("\nDrift report saved successfully!")
print("Incoming sample size:", len(incoming))

if len(incoming) < 100:
    print("WARNING: Small incoming sample. Results are only illustrative.")