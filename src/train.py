import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    FEATURES,
    METRICS_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PLOTS_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data import clean_data, load_raw_data, save_clean_data

EXPERIMENT_NAME = "Heart-Disease-MLOps-Assignment-01"


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def _candidate_models():
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }


def model_specs():
    return _candidate_models()


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions, zero_division=0),
        "test_recall": recall_score(y_test, predictions, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def save_roc_plot(model_directory, X_test, y_test):
    fig, axis = plt.subplots(figsize=(8, 6))
    for label, model in model_directory.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axis, name=label)
    axis.set_title("ROC Curves — Model Comparison")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

def main():
    for directory in [ARTIFACT_DIR, PLOTS_DIR, METRICS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    raw_frame = load_raw_data()
    cleaned_frame = clean_data(raw_frame)
    save_clean_data(cleaned_frame)

    feature_matrix = cleaned_frame[FEATURES]
    target_vector = cleaned_frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, target_vector, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target_vector
    )

    cross_validator = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{(Path.cwd() / 'mlruns').resolve()}"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_results = {}
    comparison_rows = []

    with mlflow.start_run(run_name="assignment_01_model_comparison"):
        mlflow.log_params({
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "feature_count": len(FEATURES),
            "numeric_features": ",".join(NUMERIC_FEATURES),
            "categorical_features": ",".join(CATEGORICAL_FEATURES),
        })

        for model_name, (model_instance, param_grid) in model_specs().items():
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", model_instance),
            ])

            with mlflow.start_run(run_name=model_name, nested=True):
                searcher = GridSearchCV(
                    pipeline,
                    param_grid=param_grid,
                    scoring="roc_auc",
                    cv=cross_validator,
                    n_jobs=-1,
                    return_train_score=True,
                    refit=True,
                )
                searcher.fit(X_train, y_train)

                metrics = evaluate(searcher.best_estimator_, X_test, y_test)
                model_results[model_name] = searcher.best_estimator_

                mlflow.log_params({
                    "model": model_name,
                    "best_params": json.dumps(searcher.best_params_, default=str),
                    "cv_scoring": "roc_auc",
                })
                mlflow.log_metrics({
                    "best_cv_roc_auc": float(searcher.best_score_),
                    "test_accuracy": float(metrics["test_accuracy"]),
                    "test_precision": float(metrics["test_precision"]),
                    "test_recall": float(metrics["test_recall"]),
                    "test_roc_auc": float(metrics["test_roc_auc"]),
                })

                cross_val_table = pd.DataFrame(searcher.cv_results_)
                cv_path = METRICS_DIR / f"{model_name}_cv_results.csv"
                cross_val_table.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path))

                report_path = METRICS_DIR / f"{model_name}_classification_report.json"
                report_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
                mlflow.log_artifact(str(report_path))

                mlflow.sklearn.log_model(searcher.best_estimator_, artifact_path=f"{model_name}_model")

                comparison_rows.append({
                    "model": model_name,
                    "best_cv_roc_auc": searcher.best_score_,
                    "test_accuracy": metrics["test_accuracy"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "test_roc_auc": metrics["test_roc_auc"],
                    "best_params": json.dumps(searcher.best_params_, default=str),
                })

        save_roc_plot(model_results, X_test, y_test)
        mlflow.log_artifact(str(PLOTS_DIR / "roc_curves.png"))

        comparison = pd.DataFrame(comparison_rows).sort_values("test_roc_auc", ascending=False)
        comparison_path = METRICS_DIR / "model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        mlflow.log_artifact(str(comparison_path))

        final_name = comparison.iloc[0]["model"]
        final_model = model_results[final_name]

        joblib.dump(final_model, MODEL_PATH)
        mlflow.log_param("selected_model", final_name)
        mlflow.log_param("selection_criterion", "best_cv_roc_auc")
        mlflow.log_artifact(str(MODEL_PATH))

        final_metrics = {
            "selected_model": final_name,
            "selection_criterion": "best_cv_roc_auc (5-fold stratified CV on the training split)",
            "selection_note": "Model chosen on cross-validated ROC-AUC; the test set was not used for selection and is reported as the final held-out estimate.",
            "results": comparison_rows,
        }
        (METRICS_DIR / "final_metrics.json").write_text(
            json.dumps(final_metrics, indent=2, default=str), encoding="utf-8"
        )
        mlflow.log_artifact(str(METRICS_DIR / "final_metrics.json"))

    print(comparison.to_string(index=False))
    print(f"\nSaved reproducible pipeline to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
