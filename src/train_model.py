import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB

from ml_utils import load_data, split_data, dataset_statistics

MODEL_DIR = "models"
METRICS_DIR = "metrics"


def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def main(timestamp):
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    X, y = load_data()
    stats = dataset_statistics(X, y)
    print("Dataset statistics:", json.dumps(stats, indent=2))

    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(X, y)
    model = train_model(X_train, y_train)

    y_pred = model.predict(X_train)
    train_metrics = {
        "train_accuracy": round(float(accuracy_score(y_train, y_pred)), 4),
        "train_f1_macro": round(float(f1_score(y_train, y_pred, average="macro")), 4),
    }
    print("Training metrics:", train_metrics)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}_nb_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    stats_path = os.path.join(METRICS_DIR, f"{timestamp}_dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "dataset": stats,
            "split_sizes": {"train": len(y_train), "calibration": len(y_cal), "test": len(y_test)},
        }, f, indent=4)

    # Track the run with MLflow (stored locally in ./mlruns, uploaded as a workflow artifact).
    # A full file:// path (not "./mlruns") so MLflow never reuses a stale relative location.
    mlruns_dir = Path("mlruns").resolve()
    mlruns_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment("Digits_GaussianNB")
    with mlflow.start_run(run_name=f"train_{timestamp}"):
        mlflow.log_params({
            "dataset": "sklearn digits",
            "model": "GaussianNB",
            "n_train": len(y_train),
            "n_features": X.shape[1],
            "n_classes": stats["n_classes"],
        })
        mlflow.log_metrics(train_metrics)
        mlflow.set_tag("timestamp", timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    main(parser.parse_args().timestamp)
