# Customer Churn Prediction — Model Assessment

## 1. Dataset Overview
- Total customer records: 440,832
- Churned customers: 249,999
- Non-churned customers: 190,833
- Churn rate: 56.71%
- Non-churn rate: 43.29%

## 2. Model Comparison

| Model | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 61.93% | 0.6867 | 0.7851 |
| Random Forest | 71.78% | 0.7911 | 0.8743 |
| XGBoost | 74.92% | 0.8108 | 0.8860 |
| LightGBM | 74.90% | 0.8112 | 0.8861 |

LightGBM achieved the highest ROC-AUC and PR-AUC,
while XGBoost achieved marginally higher accuracy.
Their performance was very similar.

Random Forest was selected for the current prediction
pipeline and threshold-based cost analysis.

## 3. Selected Model Evaluation

- Model: Random Forest
- ROC-AUC: 0.7911
- Prediction threshold: 0.30
- Accuracy at this threshold: 64%
- Churn recall: 86%
- Churn precision: 64%

The selected threshold prioritises identifying churned
customers while accepting additional false positives.

## 4. Threshold and Cost Analysis

False positive cost assumption: 1 unit.
False negative cost assumption: 5 units.

Threshold 0.30 produced the lowest estimated cost
among the five tested thresholds.

These are illustrative cost units, not actual
business or monetary costs.

## 5. Dataset Limitations

1. The dataset does not contain verified event timestamps.
2. True time-based customer cohorts cannot be constructed.
3. Kaplan-Meier survival analysis cannot be validated
   without reliable duration and censoring information.
4. Several variables show suspicious deterministic
   relationships with the churn label.
5. These patterns may reflect dataset-generation rules;
   their cause cannot be confirmed.
6. The dataset does not provide detailed billing events,
   support conversation history or product event logs.
7. Synthetic incoming customers were used to demonstrate
   prediction and drift-monitoring workflows.

## 6. Model Limitations

1. Model performance is dataset-dependent.
2. Predictions do not establish causal reasons for churn.
3. The selected threshold requires validation using
   actual business intervention costs.
4. The current pipeline is a local prototype.
5. External production integrations are not implemented.

## 7. Future Enhancements

- Collect timestamped customer activity.
- Perform genuine cohort and survival analysis.
- Integrate Apache Airflow for pipeline scheduling.
- Integrate MLflow for experiment tracking.
- Connect predictions to CRM systems.
- Implement automated drift alerts.
- Evaluate models on independent real-world data.
- Explore SHAP-based customer-level explanations.
- Investigate uplift modelling for retention campaigns.