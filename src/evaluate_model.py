import argparse
import glob
import json
import os
import re
import sys

import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from ml_utils import load_data, split_data, probability_metrics

MODEL_DIR = "models"
METRICS_DIR = "metrics"
METRICS_PATTERN = re.compile(r"^(\d+)_metrics\.json$")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
    }
    metrics.update(probability_metrics(y_test, proba, model.classes_))
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    return metrics


# Most recent metrics file from an EARLIER run (the currently deployed model)
def find_previous_metrics(current_timestamp, metrics_dir=METRICS_DIR):
    earlier = sorted(
        p for p in glob.glob(os.path.join(metrics_dir, "*_metrics.json"))
        if (m := METRICS_PATTERN.match(os.path.basename(p))) and m.group(1) < current_timestamp
    )
    if not earlier:
        return None
    with open(earlier[-1]) as f:
        return json.load(f)["test_metrics"]


# CI/CD quality gate: a model is only versioned if it is good enough AND
# not meaningfully worse than the model it would replace
def quality_gate(metrics, previous, min_accuracy, max_drop):
    reasons = []
    if metrics["accuracy"] < min_accuracy:
        reasons.append(f"accuracy {metrics['accuracy']:.4f} is below the minimum {min_accuracy:.4f}")
    if previous is not None and metrics["accuracy"] < previous["accuracy"] - max_drop:
        reasons.append(f"accuracy {metrics['accuracy']:.4f} dropped more than {max_drop:.4f} "
                       f"from the previous model ({previous['accuracy']:.4f})")
    return len(reasons) == 0, reasons


# Markdown table shown on the GitHub Actions run page
def build_summary(timestamp, metrics, previous, passed, reasons):
    lines = [f"## Model evaluation — `{timestamp}`", "",
             "| Metric | This model | Previous model |", "|---|---|---|"]
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "log_loss", "brier_score", "ece"]:
        prev = f"{previous[key]:.4f}" if previous and key in previous else "—"
        lines.append(f"| {key} | {metrics[key]:.4f} | {prev} |")
    lines.append("")
    if passed:
        lines.append("**Quality gate: ✅ PASSED** — model will be versioned and committed.")
    else:
        lines.append("**Quality gate: ❌ FAILED** — model rejected:")
        lines.extend(f"- {r}" for r in reasons)
    return "\n".join(lines) + "\n"


def write_step_summary(markdown):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown)


def main(timestamp, min_accuracy=0.75, max_drop=0.02):
    model_path = os.path.join(MODEL_DIR, f"model_{timestamp}_nb_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)

    # Evaluate on the held-out test split (never seen during training)
    X, y = load_data()
    _, _, X_test, _, _, y_test = split_data(X, y)

    metrics = evaluate_model(model, X_test, y_test)
    previous = find_previous_metrics(timestamp)
    passed, reasons = quality_gate(metrics, previous, min_accuracy, max_drop)

    summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    print("Test metrics:", json.dumps(summary, indent=2))

    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = os.path.join(METRICS_DIR, f"{timestamp}_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"timestamp": timestamp, "model": "GaussianNB", "test_metrics": metrics,
                   "quality_gate": {"passed": passed, "min_accuracy": min_accuracy,
                                    "max_drop": max_drop, "reasons": reasons}}, f, indent=4)
    print(f"Metrics saved to {out_path}")

    write_step_summary(build_summary(timestamp, metrics, previous, passed, reasons))

    if passed:
        print("Quality gate PASSED")
    else:
        print("Quality gate FAILED:\n  - " + "\n  - ".join(reasons))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp from GitHub Actions")
    parser.add_argument("--min-accuracy", type=float, default=0.75,
                        help="Reject the model if test accuracy is below this")
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="Reject the model if accuracy drops more than this vs. the previous model")
    args = parser.parse_args()
    sys.exit(0 if main(args.timestamp, args.min_accuracy, args.max_drop) else 1)
