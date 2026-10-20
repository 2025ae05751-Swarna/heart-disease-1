# Clinical Heart Risk Prediction Project

## Project Summary

This project implements a compact machine-learning pipeline for predicting heart disease risk from patient health indicators. The solution combines data preparation, model evaluation, API serving, and monitoring into a single, reproducible workflow that can be run locally or in a containerized deployment environment.

The focus is on building a practical system that can process patient records, identify risk patterns, and expose predictions through a lightweight API without overcomplicating the deployment stack.

---

## Objective

The main goal is to build a predictive model that uses clinical and demographic attributes to estimate whether the patient is likely to have heart disease. The system is designed to be explainable, reproducible, and easy to operate in a development or small production setting.

---

## Data Overview

The project uses a heart disease dataset containing a mix of numeric and categorical medical attributes such as age, blood pressure, cholesterol level, resting ECG readings, chest pain type, maximum heart rate, and exercise-induced symptoms. The target variable indicates whether heart disease is present or absent.

The data pipeline cleans the dataset, standardizes feature naming, and prepares a consistent schema before training and evaluation.

---

## Pipeline Workflow

The workflow follows a simple MLOps style structure:

1. Data ingestion and validation
2. Feature cleaning and preprocessing
3. Train/test split with class preservation
4. Model comparison and tuning
5. Artifact generation and model persistence
6. API deployment for inference
7. Monitoring and operational tracking

This design allows the project to remain transparent while still supporting a realistic production-oriented setup.

---

## Modeling Strategy

Two classification models are compared to find the most reliable approach:

- Logistic Regression for an interpretable baseline
- Random Forest for a more flexible non-linear model

Both models are evaluated using validation and comparison metrics such as accuracy, precision, recall, and ROC-AUC. The selected model is then saved with its preprocessing step so that future predictions use the same transformation logic as training.

---

## Preprocessing and Validation

The preprocessing stage is built to avoid leakage and maintain consistency across training and inference. Numeric inputs are scaled and categorical values are transformed appropriately before the model sees them. The train/test split is organized to preserve class balance and to reflect the true operating conditions of the model.

This makes the pipeline more robust and reduces the risk of over-optimistic evaluation results.

---

## Experiment Tracking

The project uses MLflow to log model parameters, validation results, and artifact metadata. This provides traceability across different runs and helps with comparing model iterations. The tracking layer is especially useful when testing alternative preprocessing choices or hyperparameter values.

---

## API Service

The trained model is exposed through a FastAPI service with endpoints for health checks and prediction requests. The API accepts a patient record in JSON format and returns a prediction result along with model confidence values, making it suitable for integration with dashboards, applications, or internal decision-support tools.

---

## Deployment and Monitoring

The application is packaged using Docker and can be deployed in a Kubernetes-based environment for more structured orchestration. Monitoring is handled with Prometheus and Grafana, which provide visibility into request volume, response latency, and general service health.

This combination helps connect training and inference with operational observability, which is essential for a reliable ML deployment.

---

## Testing and Quality Checks

The project includes automated tests for core functions such as dataframe cleaning, target handling, and API response validation. This helps catch regressions early and provides confidence that the data pipeline and inference service still behave as expected after code changes.

---

## Conclusion

This project demonstrates a practical end-to-end machine learning workflow for a healthcare use case. It combines data preparation, classification modeling, reproducible artifact handling, API deployment, and observability in a way that is clear, modular, and operationally useful.

The result is a small but realistic MLOps-style application that balances model performance, maintainability, and deployment readiness.

---

## Key Artifacts

- Training and validation pipeline: src/train.py
- Data preparation logic: src/data.py
- API service: src/api.py
- Monitoring dashboard: monitoring/grafana/dashboards/ml-api-dashboard.json
- Model artifacts: artifacts/
- Sample request: sample_request.json

