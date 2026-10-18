import json
import os

import joblib
import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB

import calibrate_model
import evaluate_model
import train_model
from ml_utils import (dataset_statistics, expected_calibration_error, load_data,
                      multiclass_brier, probability_metrics, split_data)

CLASSES = np.arange(10)


@pytest.fixture(scope="module")
def data():
    X, y = load_data()
    return X, y, split_data(X, y)


@pytest.fixture(scope="module")
def trained(data):
    _, _, (X_train, _, _, y_train, _, _) = data
    return train_model.train_model(X_train, y_train)


# ----------------- Data ----------------- #
def test_load_data(data):
    X, y, _ = data
    assert X.shape == (1797, 64)
    assert set(np.unique(y)) == set(range(10))


def test_split_data(data):
    X, y, (X_train, X_cal, X_test, y_train, y_cal, y_test) = data
    assert len(y_train) + len(y_cal) + len(y_test) == len(y)
    assert abs(len(y_train) / len(y) - 0.6) < 0.01
    # Stratified: every class appears in every split
    for split in (y_train, y_cal, y_test):
        assert set(np.unique(split)) == set(range(10))


def test_split_is_reproducible():
    X, y = load_data()
    first, second = split_data(X, y), split_data(X, y)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_dataset_statistics(data):
    X, y, _ = data
    stats = dataset_statistics(X, y)
    assert stats["n_samples"] == 1797
    assert stats["n_features"] == 64
    assert stats["n_classes"] == 10
    assert stats["missing_values"] == 0
    assert sum(stats["class_distribution"].values()) == 1797
    assert 0 < stats["class_balance_ratio"] <= 1
    assert stats["pixel_value_range"] == [0.0, 16.0]
    json.dumps(stats)  # must be JSON-serializable


# ----------------- Metric helpers ----------------- #
def test_metrics_perfect_predictions():
    y = np.array([0, 1, 2, 3])
    proba = np.eye(10)[y]
    assert multiclass_brier(y, proba, CLASSES) == pytest.approx(0.0)
    assert expected_calibration_error(y, proba, CLASSES) == pytest.approx(0.0)


def test_metrics_uniform_predictions():
    y = np.array([0, 1, 2, 3])
    proba = np.full((4, 10), 0.1)
    # (1 - 0.1)^2 + 9 * 0.1^2 = 0.9
    assert multiclass_brier(y, proba, CLASSES) == pytest.approx(0.9)
    metrics = probability_metrics(y, proba, CLASSES)
    assert metrics["log_loss"] == pytest.approx(np.log(10), abs=1e-3)


# ----------------- Training ----------------- #
def test_train_model(trained):
    assert isinstance(trained, GaussianNB)
    assert hasattr(trained, "predict_proba")
    assert list(trained.classes_) == list(range(10))


# ----------------- Evaluation ----------------- #
def test_evaluate_model(trained, data):
    _, _, (_, _, X_test, _, _, y_test) = data
    metrics = evaluate_model.evaluate_model(trained, X_test, y_test)
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "ece"]:
        assert 0.0 <= metrics[key] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 2.0
    assert metrics["log_loss"] > 0
    assert len(metrics["confusion_matrix"]) == 10
    assert sum(map(sum, metrics["confusion_matrix"])) == len(y_test)
    assert metrics["accuracy"] > 0.7  # sanity floor for GaussianNB on digits


# ----------------- Calibration ----------------- #
def test_find_latest_model(tmp_path):
    for name in ["model_20260101000000_nb_model.joblib",
                 "model_20261231000000_nb_model.joblib",
                 "model_20261231000000_calibrated_isotonic.joblib"]:
        (tmp_path / name).write_text("x")
    path, ts = calibrate_model.find_latest_model(str(tmp_path))
    assert ts == "20261231000000"
    assert path.endswith("model_20261231000000_nb_model.joblib")


def test_find_latest_model_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        calibrate_model.find_latest_model(str(tmp_path))


def test_calibration_improves_probabilities(trained, data):
    _, _, (_, X_cal, X_test, _, y_cal, y_test) = data
    before = calibrate_model.score(trained, X_test, y_test)
    calibrated = calibrate_model.calibrate(trained, X_cal, y_cal, "isotonic")
    after = calibrate_model.score(calibrated, X_test, y_test)
    assert after["brier_score"] < before["brier_score"]
    assert after["ece"] < before["ece"]


# ----------------- End-to-end ----------------- #
def test_full_pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ts = "20260101120000"
    train_model.main(ts)
    assert evaluate_model.main(ts) is True
    calibrate_model.main()

    assert os.path.exists(f"models/model_{ts}_nb_model.joblib")
    assert os.path.exists(f"metrics/{ts}_dataset_stats.json")
    assert os.path.exists(f"metrics/{ts}_metrics.json")
    assert os.path.exists(f"metrics/{ts}_reliability.png")
    with open(f"metrics/{ts}_calibration.json") as f:
        report = json.load(f)
    assert report["best_method"] in ("sigmoid", "isotonic")
    calibrated = joblib.load(f"models/model_{ts}_calibrated_{report['best_method']}.joblib")
    assert hasattr(calibrated, "predict_proba")


# ----------------- Quality gate (CI/CD) ----------------- #
GOOD = {"accuracy": 0.82}


def test_quality_gate_passes_first_model():
    passed, reasons = evaluate_model.quality_gate(GOOD, None, min_accuracy=0.75, max_drop=0.02)
    assert passed and reasons == []


def test_quality_gate_rejects_low_accuracy():
    passed, reasons = evaluate_model.quality_gate({"accuracy": 0.60}, None, 0.75, 0.02)
    assert not passed
    assert "below the minimum" in reasons[0]


def test_quality_gate_rejects_regression():
    passed, reasons = evaluate_model.quality_gate({"accuracy": 0.78}, {"accuracy": 0.82}, 0.75, 0.02)
    assert not passed
    assert "dropped" in reasons[0]


def test_quality_gate_allows_small_drop():
    passed, _ = evaluate_model.quality_gate({"accuracy": 0.81}, {"accuracy": 0.82}, 0.75, 0.02)
    assert passed


def test_find_previous_metrics(tmp_path):
    for ts, acc in [("20260101000000", 0.80), ("20260201000000", 0.82), ("20260301000000", 0.99)]:
        (tmp_path / f"{ts}_metrics.json").write_text(json.dumps({"test_metrics": {"accuracy": acc}}))
    prev = evaluate_model.find_previous_metrics("20260301000000", str(tmp_path))
    assert prev["accuracy"] == 0.82  # latest EARLIER run, not the current one
    assert evaluate_model.find_previous_metrics("20250101000000", str(tmp_path)) is None


def test_gate_blocks_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ts = "20260101130000"
    train_model.main(ts)
    assert evaluate_model.main(ts, min_accuracy=0.99) is False  # impossible bar -> rejected
    with open(f"metrics/{ts}_metrics.json") as f:
        assert json.load(f)["quality_gate"]["passed"] is False


def test_step_summary_written(tmp_path, monkeypatch):
    summary_file = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))
    evaluate_model.write_step_summary("## hello\n")
    assert "## hello" in summary_file.read_text()
