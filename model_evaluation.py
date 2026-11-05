import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

df = pd.read_csv("batch_churn_predictions.csv")

y_true = df["Actual_Churn"]
y_pred = df["Predicted_Churn"]
y_prob = df["Churn_Probability"]

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Customer Churn - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
auc = roc_auc_score(y_true, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Customer Churn - ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# Classification report
report = classification_report(y_true, y_pred)
with open("classification_report.txt", "w") as file:
    file.write(report)

print("Model evaluation completed!")
print("ROC-AUC:", round(auc, 4))
print(report)
print("Evaluation files saved!")