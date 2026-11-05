# Customer Churn Prediction
## Proposed Production Architecture

### 1. Data Collection
- Customer account information
- Billing and payment information
- Product usage information
- Customer support interactions

### 2. Data Pipeline
- Apache Airflow: Schedule and orchestrate data pipelines.
- Data validation: Check missing values and invalid records.
- Feature engineering: Generate customer-level features.

### 3. Machine Learning
- Logistic Regression: Baseline model.
- Random Forest: Selected prediction model.
- XGBoost and LightGBM: Comparison models.
- MLflow: Track experiments, metrics and model versions.

### 4. Prediction
- Batch customer scoring.
- Churn probability calculation.
- Threshold-based churn classification.

### 5. Business Integration
- CRM / Reverse ETL: Send high-risk customer predictions
  to customer retention teams.
- Retention teams can plan targeted campaigns.

### 6. Monitoring
- Data drift monitoring.
- Model performance monitoring.
- Periodic model evaluation and retraining.

## Current Implementation Status

| Component | Status |
|---|---|
| Data cleaning | Implemented |
| Feature engineering | Implemented |
| Model training | Implemented |
| Batch prediction | Implemented |
| Drift monitoring | Prototype |
| Airflow | Proposed |
| MLflow | Proposed |
| CRM integration | Proposed |
| Automated retraining | Proposed |

## Architecture Flow

Raw Data
    |
    v
Data Validation
    |
    v
Feature Engineering
    |
    v
Machine Learning Model
    |
    v
Churn Probability
    |
    v
Risk Classification
    |
    v
CRM / Retention Team

Airflow coordinates the pipeline.
MLflow tracks experiments and models.
Monitoring checks incoming data and model performance.