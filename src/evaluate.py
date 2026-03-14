"""
evaluate.py — Binary Incident Prediction Evaluation
=====================================================
Evaluation utilities for the incident classifier produced by train.py.

Responsibilities
----------------
  1. Compute the standard binary classification metrics at a given threshold.
  2. Apply a threshold to predicted probabilities to generate alert flags.
  3. Compare how those metrics shift across a range of thresholds, so an
     operator can pick the right precision/recall trade-off for their context.

Threshold intuition
-------------------
  A classifier outputs P(incident) ∈ [0, 1].  We convert that to a binary
  alert by picking a cut-off:
    - Low threshold (e.g. 0.2) → alert fires often → high recall, low precision
      ("catch everything, accept more false alarms")
    - High threshold (e.g. 0.7) → alert fires rarely → high precision, low recall
      ("only alert when very confident, miss some real events")

  The default threshold (0.5) is rarely optimal for imbalanced data.
  This module makes it easy to explore alternatives before deployment.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)

# Default probability cut-off used when none is specified by the caller.
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Evaluate binary incident predictions at a single probability threshold.

    Parameters
    ----------
    y_true    : true binary labels (0 = no incident, 1 = incident)
    y_prob    : predicted probability of the positive class, in [0, 1]
    threshold : probability cut-off used to convert y_prob → binary alert

    Returns
    -------
    dict with keys:
        threshold  — the cut-off used
        precision  — of the windows flagged as incidents, fraction that are real
        recall     — of all real incidents, fraction that were flagged
        f1         — harmonic mean of precision and recall
        roc_auc    — area under the ROC curve (threshold-independent)
        n_alerts   — number of windows where the alert fired
        n_positive — total number of true incident windows in y_true

    Notes
    -----
    ROC-AUC is computed from y_prob directly and does not depend on the
    threshold — it measures how well the model *ranks* incident windows
    above non-incident ones across all possible cut-offs.
    """
    y_pred = apply_threshold(y_prob, threshold)

    # zero_division=0 returns 0 instead of raising when no positives are
    # predicted — prevents crashes when threshold is set very high.
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    auc       = roc_auc_score(y_true, y_prob)

    metrics = {
        "threshold":  threshold,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "roc_auc":    round(auc,       4),
        "n_alerts":   int(y_pred.sum()),
        "n_positive": int(y_true.sum()),
    }

    log.info(
        "threshold=%.2f | precision=%.3f | recall=%.3f | f1=%.3f | "
        "roc_auc=%.3f | alerts=%d / %d positive",
        threshold, precision, recall, f1, auc,
        metrics["n_alerts"], metrics["n_positive"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Threshold application
# ---------------------------------------------------------------------------

def apply_threshold(y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Convert predicted probabilities to binary alert flags.

    A window is flagged as an alert when its predicted probability of an
    incident meets or exceeds ``threshold``.

    Parameters
    ----------
    y_prob    : 1-D array of predicted probabilities in [0, 1]
    threshold : cut-off value in (0, 1]

    Returns
    -------
    np.ndarray of dtype int (0 = no alert, 1 = alert)

    Raises
    ------
    ValueError if threshold is outside (0, 1].
    """
    if not (0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}.")

    return (y_prob >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Multi-threshold comparison
# ---------------------------------------------------------------------------

def compare_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate metrics at multiple thresholds and return a summary table.

    Useful for choosing an operating point before deployment.  A typical
    workflow is to call this function, inspect the table, and then pick the
    threshold that gives the desired recall for an acceptable precision.

    Parameters
    ----------
    y_true     : true binary labels
    y_prob     : predicted probabilities
    thresholds : list of cut-offs to evaluate.  Defaults to
                 [0.10, 0.20, …, 0.90] if not provided.

    Returns
    -------
    pd.DataFrame
        One row per threshold, columns matching compute_metrics() output,
        sorted by threshold ascending.

    Example output
    --------------
        threshold  precision  recall    f1   roc_auc  n_alerts  n_positive
             0.10      0.312   0.981  0.474    0.961      1822         234
             0.20      0.489   0.936  0.642    0.961       449         234
             0.30      0.631   0.897  0.741    0.961       333         234
             0.50      0.812   0.821  0.816    0.961       237         234
             0.70      0.923   0.718  0.808    0.961       182         234
             0.90      0.971   0.496  0.656    0.961       119         234
    """
    if thresholds is None:
        # Ten evenly spaced cut-offs; skipping 0.0 (every row becomes an alert)
        # and 1.0 (no row ever becomes an alert, division by zero risk).
        thresholds = [round(t, 2) for t in np.arange(0.10, 1.00, 0.10)]

    rows = [compute_metrics(y_true, y_prob, threshold=t) for t in thresholds]
    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Alert summary helper
# ---------------------------------------------------------------------------

def alert_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Return a plain-English summary of alert quality at the chosen threshold.

    Intended for quick reporting — e.g. printing to a dashboard or notebook
    — rather than further computation.

    Returns a dict with:
        true_positives  — real incidents that were correctly alerted
        false_positives — alerts that fired on non-incident windows
        false_negatives — real incidents that were missed
        true_negatives  — non-incident windows that were correctly silent
        alert_rate      — fraction of all windows that triggered an alert
        miss_rate       — fraction of real incidents that were missed (1 - recall)
    """
    y_pred = apply_threshold(y_prob, threshold)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    n = len(y_true)

    return {
        "threshold":      threshold,
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives":  tn,
        "alert_rate":      round((tp + fp) / n, 4),
        "miss_rate":       round(fn / max(tp + fn, 1), 4),
    }


# ---------------------------------------------------------------------------
# CLI — quick demo using the full pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data.synthetic_generator import generate_dataset
    from src.features import extract_features
    from src.train import train_model
    from src.windowing import make_windows

    print("Generating data and training model…")
    raw_df = generate_dataset(n_steps=10_000)
    sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
    dataset = make_windows(raw_df, feature_cols=sensor_cols, label_col="incident",
                           window_size=30, horizon=10)
    X = extract_features(dataset)
    result = train_model(X, dataset.y)

    y_true = result.y_test
    y_prob = result.y_prob

    print("\n── Metrics at default threshold (0.5) ──────────────────────────")
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    for k, v in metrics.items():
        print(f"  {k:<16} {v}")

    print("\n── Alert summary at threshold 0.5 ──────────────────────────────")
    summary = alert_summary(y_true, y_prob, threshold=0.5)
    for k, v in summary.items():
        print(f"  {k:<20} {v}")

    print("\n── Threshold comparison ─────────────────────────────────────────")
    comparison = compare_thresholds(y_true, y_prob)
    print(comparison.to_string(index=False))
