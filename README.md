# Customer Churn Prediction

## Project Overview
A machine learning project that analyses customer information and predicts customer churn risk. The project includes data cleaning, exploratory data analysis, feature engineering, model comparison, explainability, and an interactive Streamlit dashboard.

## Objectives
- Analyse customer churn patterns.
- Engineer useful customer behaviour features.
- Compare multiple classification algorithms.
- Evaluate model performance using classification metrics.
- Explain model predictions using SHAP.
- Build an interactive churn prediction application.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost and LightGBM
- SHAP
- Matplotlib
- Streamlit
- Joblib

## Dataset
Source: [Customer Churn Dataset – Hugging Face](https://huggingface.co/datasets/MatanKriel/Matan-Kriel-Customer-Churn-dataset)

The cleaned dataset contains 440,832 customer records and 12 original columns.

## Model Performance

| Model | Accuracy | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 61.93% | 0.6867 | 0.7851 |
| Random Forest | 71.78% | 0.7911 | 0.8743 |
| XGBoost | 74.92% | 0.8108 | 0.8860 |
| LightGBM | 74.90% | 0.8112 | 0.8861 |
| Compact Deployment Random Forest | 74.97% | 0.8085 | 0.8849 |

The compact Random Forest is used in the interactive application. Its classification threshold is 0.30.

At this threshold, its test-set churn recall is approximately 95.07%, with precision of 59.41%.

## Interactive Dashboard

The Streamlit application accepts customer details and displays:
- Estimated churn probability.
- Higher or lower churn-risk classification.
- Model threshold used for classification.
**Live Application:** https://customerchurnprediction-t8gtx87g3mxqk4s6f8tfpa.streamlit.app/
## Run Locally

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sathwickah007-lgtm/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your default browser.

## Project Limitations
- The dataset is cross-sectional and does not contain a complete customer event timeline.
- Retention analysis is a tenure-based proxy, not true cohort or Kaplan-Meier survival analysis.
- Some strong dataset associations may reflect deterministic data-generation rules.
- Drift analysis uses a small synthetic sample and is illustrative.
- Predictions represent prototype model outputs, not guaranteed real-world customer behaviour.
- Automated retraining, CRM integration, Airflow and MLflow deployment are proposed future enhancements, not completed integrations.

## Future Enhancements
- Integrate timestamp-based customer activity data.
- Implement true survival analysis.
- Add larger-scale batch prediction.
- Introduce production drift monitoring.
- Integrate with customer retention platforms.

