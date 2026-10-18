"""Shared data and metric helpers used by training, evaluation and calibration."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


# ----------------- Data ----------------- #
# Digits dataset: 1,797 8x8 handwritten digit images, 64 pixel features, 10 classes (0-9)
def load_data():
    digits = load_digits()
    return digits.data, digits.target


# Stratified 60/20/20 split: train the model / fit the calibrator / final test.
# Fixed random_state so every script (and both workflows) gets the identical split.
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


# ----------------- Statistics ----------------- #
def dataset_statistics(X, y):
    classes, counts = np.unique(y, return_counts=True)
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(classes)),
        "missing_values": int(np.isnan(X).sum()),
        "class_distribution": {int(c): int(n) for c, n in zip(classes, counts)},
        "class_balance_ratio": round(float(counts.min() / counts.max()), 4),
        "pixel_value_range": [float(X.min()), float(X.max())],
        "mean_pixel_intensity": round(float(X.mean()), 4),
        "zero_pixel_fraction": round(float((X == 0).mean()), 4),
        "constant_features": int((X.std(axis=0) == 0).sum()),
    }


# Multiclass Brier score: mean squared distance between predicted probabilities
# and the one-hot true label (0 = perfect, lower is better)
def multiclass_brier(y_true, proba, classes):
    one_hot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


# Expected Calibration Error: gap between confidence and accuracy, averaged over
# 10 confidence bins (0 = perfectly calibrated)
def expected_calibration_error(y_true, proba, classes, n_bins=10):
    confidence = proba.max(axis=1)
    predicted = np.asarray(classes)[proba.argmax(axis=1)]
    correct = (predicted == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def probability_metrics(y_true, proba, classes):
    return {
        "log_loss": round(float(log_loss(y_true, proba, labels=classes)), 4),
        "brier_score": round(multiclass_brier(y_true, proba, classes), 4),
        "ece": round(expected_calibration_error(y_true, proba, classes), 4),
    }
