import argparse
import glob
import json
import os
import re

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed on the CI runner
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from ml_utils import load_data, split_data, probability_metrics

MODEL_DIR = "models"
METRICS_DIR = "metrics"
MODEL_PATTERN = re.compile(r"^model_(\d+)_nb_model\.joblib$")


# Newest uncalibrated model in models/ (timestamps sort chronologically)
def find_latest_model(model_dir=MODEL_DIR):
    candidates = sorted(
        p for p in glob.glob(os.path.join(model_dir, "model_*_nb_model.joblib"))
        if MODEL_PATTERN.match(os.path.basename(p))
    )
    if not candidates:
        raise FileNotFoundError(f"No trained model found in {model_dir}/")
    latest = candidates[-1]
    return latest, MODEL_PATTERN.match(os.path.basename(latest)).group(1)


# Fit a calibrator on the held-out calibration split without retraining the model
def calibrate(model, X_cal, y_cal, method):
    try:  # scikit-learn >= 1.6
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method)
    except ImportError:  # scikit-learn < 1.6 (our pinned version)
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_cal, y_cal)
    return calibrated


def score(model, X_test, y_test):
    result = {"accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 4)}
    result.update(probability_metrics(y_test, model.predict_proba(X_test), model.classes_))
    return result


# Reliability diagram: predicted confidence vs. actual accuracy per bin (diagonal = perfect)
def plot_reliability(models, X_test, y_test, out_path, n_bins=10, min_count=5):
    edges = np.linspace(0, 1, n_bins + 1)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for name, model in models.items():
        proba = model.predict_proba(X_test)
        conf = proba.max(axis=1)
        correct = model.classes_[proba.argmax(axis=1)] == y_test
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.sum() >= min_count:  # skip near-empty bins (too noisy to read)
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        plt.plot(xs, ys, "o-", label=name)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Actual accuracy")
    plt.title("Reliability diagram (test set)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main(timestamp=None):
    if timestamp:
        model_path = os.path.join(MODEL_DIR, f"model_{timestamp}_nb_model.joblib")
    else:
        model_path, timestamp = find_latest_model()
    print(f"Calibrating {model_path}")
    model = joblib.load(model_path)

    X, y = load_data()
    _, X_cal, X_test, _, y_cal, y_test = split_data(X, y)

    candidates = {
        "sigmoid": calibrate(model, X_cal, y_cal, "sigmoid"),
        "isotonic": calibrate(model, X_cal, y_cal, "isotonic"),
    }

    results = {"uncalibrated": score(model, X_test, y_test)}
    for method, calibrated in candidates.items():
        results[method] = score(calibrated, X_test, y_test)

    best = min(candidates, key=lambda m: results[m]["brier_score"])
    print("Calibration results:", json.dumps(results, indent=2))
    print(f"Best method (lowest Brier score): {best}")

    os.makedirs(METRICS_DIR, exist_ok=True)
    calibrated_path = os.path.join(MODEL_DIR, f"model_{timestamp}_calibrated_{best}.joblib")
    joblib.dump(candidates[best], calibrated_path)

    with open(os.path.join(METRICS_DIR, f"{timestamp}_calibration.json"), "w") as f:
        json.dump({"timestamp": timestamp, "source_model": os.path.basename(model_path),
                   "best_method": best, "results": results}, f, indent=4)

    plot_reliability(
        {"Uncalibrated": model, "Sigmoid (Platt)": candidates["sigmoid"],
         "Isotonic": candidates["isotonic"]},
        X_test, y_test, os.path.join(METRICS_DIR, f"{timestamp}_reliability.png"),
    )
    print(f"Calibrated model saved to {calibrated_path}")

    write_step_summary(build_summary(timestamp, results, best))


# Markdown table shown on the GitHub Actions run page
def build_summary(timestamp, results, best):
    lines = [f"## Model calibration — `{timestamp}`", "",
             "| Method | Accuracy | Log loss | Brier score | ECE |", "|---|---|---|---|---|"]
    for name, r in results.items():
        label = f"**{name}** (selected)" if name == best else name
        lines.append(f"| {label} | {r['accuracy']:.4f} | {r['log_loss']:.4f} | "
                     f"{r['brier_score']:.4f} | {r['ece']:.4f} |")
    return "\n".join(lines) + "\n"


def write_step_summary(markdown):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, default=None,
                        help="Model timestamp to calibrate (default: latest in models/)")
    main(parser.parse_args().timestamp)
