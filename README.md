# Cardiovascular Risk Modeling Pipeline
Name: Swarnalatha S
ID: 2025AE05751

This repository contains a practical end-to-end workflow for building a heart disease prediction service using structured clinical data.

The project covers the main MLOps stages in a compact, reproducible setup:

1. Data download and quality review
2. Exploratory analysis and feature review
3. Model training and comparison
4. Experiment tracking with MLflow
5. Automated checks with pytest and linting
6. API serving with FastAPI
7. Container orchestration with Docker and Compose
8. Monitoring with Prometheus and Grafana
9. Documentation and deployment notes

## Project structure

```text
.
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── train.py
│   ├── predict.py
│   └── api.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── plots/
│   └── metrics/
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── k8s/
│   ├── namespace.yaml
│   ├── api-deployment.yaml
│   ├── api-service.yaml
│   ├── prometheus-config.yaml
│   ├── prometheus-deployment.yaml
│   ├── grafana-config.yaml
│   ├── grafana-dashboard-configmap.yaml
│   └── grafana-deployment.yaml
├── monitoring/prometheus/prometheus.yml
├── monitoring/grafana/dashboards/ml-api-dashboard.json
├── report.md
├── screenshots/
└── download_data.py
```

## 1. Local environment

Python 3.11 is recommended for this project.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 2. Dataset retrieval

```bash
python download_data.py
```

This step downloads the UCI heart disease dataset through the UC Irvine repository interface, standardizes the target label into a binary format, normalizes missing-value markers, and writes the downloaded files into the project data folder.

```text
data/raw/heart_disease.csv
data/raw/heart_disease_metadata.json
```

## 3. Exploratory analysis

```bash
python -m src.eda
```

This produces the key review artifacts used to understand the feature distribution and data quality, including:

```text
artifacts/plots/feature_histograms.png
artifacts/plots/correlation_heatmap.png
artifacts/plots/class_balance.png
artifacts/plots/boxplots_numeric.png
artifacts/metrics/data_summary.json
```

## 4. Model training and comparison

```bash
python -m src.train
```

The training pipeline does the following:

- builds a preprocessing strategy for numeric and categorical variables;
- fits a logistic regression model and a random forest baseline;
- uses cross-validation to tune parameters;
- evaluates performance using accuracy, precision, recall, and ROC-AUC;
- logs training metadata and artifacts to MLflow;
- stores model comparison results and the final chosen pipeline;
- saves the trained model for later inference.

Start the MLflow interface:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open the browser at http://localhost:5000.

## 5. Testing and linting

```bash
pytest -q
```

```bash
ruff check .
```

## 6. Serving the model locally

The training process writes the final artifact to artifacts/model.joblib.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Check the service health:

```bash
curl http://localhost:8000/health
```

Review runtime metrics:

```bash
curl http://localhost:8000/metrics
```

Send a sample prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

The response returns both a class label and the confidence score.

## 7. Container deployment

Build the image:

```bash
docker build -t heart-disease-mlops:latest .
```

Run it locally:

```bash
docker run --rm -p 8000:8000 heart-disease-mlops:latest
```

Check the service with:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Launch the Compose stack:

```bash
docker compose up --build
```

The stack includes:

- FastAPI service on port 8000
- MLflow tracking on port 5000
- Prometheus on port 9090
- Grafana on port 3000

Default Grafana credentials:

```text
username: admin
password: admin
```

These can be changed before a production rollout.

## 8. Kubernetes and local cluster testing

Build the image first:

```bash
docker build -t heart-disease-mlops:latest .
```

For Docker Desktop Kubernetes:

```bash
kubectl config use-context docker-desktop
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl rollout status deployment/heart-api -n heart-mlops
kubectl get pods -n heart-mlops
kubectl get svc -n heart-mlops
```

On Windows, reach the API from the host through a local port-forward (no Minikube tunnel required):

```bash
kubectl port-forward -n heart-mlops service/heart-api 18080:80
```

With the command running, open `http://127.0.0.1:18080/docs`.

For Minikube instead, start the cluster, load the image with `minikube image load heart-disease-mlops:latest`, and use `minikube tunnel` if you need its LoadBalancer address.

Then view the generated service endpoints. Prometheus is exposed internally, while Grafana can be opened through the cluster service as needed.

## 9. CI pipeline

The GitHub Actions workflow runs checks in order:

1. dependency installation
2. Ruff linting
3. pytest execution
4. data download
5. training run
6. artifact upload

Push the repository to GitHub and enable Actions to automate validation.

## 10. Evidence collection

Capture your own screenshots in the screenshots folder for:

- EDA charts
- feature correlation views
- class distribution plots
- MLflow run history
- model comparison output
- passing tests
- successful CI run
- Docker build verification
- API health check
- prediction response example
- metrics endpoint output
- Kubernetes pod and service state
- Grafana dashboard view
- final deployment access point

## 11. Submission checklist

- [ ] GitHub repository
- [ ] project code
- [ ] requirements file
- [ ] dataset download script
- [ ] cleaned dataset
- [ ] EDA outputs
- [ ] training pipeline
- [ ] inference service
- [ ] tests
- [ ] CI workflow
- [ ] Docker configuration
- [ ] Kubernetes manifests
- [ ] screenshots
- [ ] project report
- [ ] deployed API URL or local instructions
- [ ] short pipeline walkthrough

- [ ] code
- [ ] requirements.txt
- [ ] dataset download script
- [ ] cleaned dataset
- [ ] EDA scripts/plots
- [ ] training script
- [ ] inference/API code
- [ ] tests
- [ ] GitHub Actions workflow
- [ ] Dockerfile
- [ ] Kubernetes manifests
- [ ] screenshots
- [ ] 10-page report
- [ ] deployed API URL or local access instructions
- [ ] short pipeline video

# trigger
