import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

bundle = joblib.load("churn_model_bundle.joblib")

model = bundle["model"]
features = bundle["feature_columns"]

df = pd.read_csv("customer_churn_engineered_encoded.csv")
X = df[features].sample(n=20, random_state=42)

print("Generating FAST approximate SHAP explanations...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(
    X,
    approximate=True,
    check_additivity=False
)

if isinstance(shap_values, list):
    shap_values = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

# SHAP summary plot
shap.summary_plot(shap_values, X, show=False)
plt.savefig("final_shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# SHAP feature importance plot
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)
plt.savefig("final_shap_importance.png", dpi=120, bbox_inches="tight")
plt.close()

print("FINAL MODEL SHAP PLOTS SAVED!")
