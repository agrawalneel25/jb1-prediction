"""
train.py — Incident Prediction Classifier
==========================================
Trains a supervised classifier on the tabular feature matrix produced by
features.py and evaluates it on a held-out test period.

Pipeline position
-----------------
  synthetic_generator.py  →  windowing.py  →  features.py  →  train.py
  (raw time series)          (windows + y)    (feature table)  (fit + eval)

Design choices
--------------
  Time-aware split
      Windows are already in chronological order (windowing.py preserves row
      order).  We cut at a single index — first 80% train, last 20% test —
      so the model never sees future data during training.  A random split
      would leak information because adjacent windows share most of their
      timesteps.

  RandomForestClassifier as default
      Handles the 24-column feature table without scaling, gives feature
      importances for free, and is robust to the class imbalance common in
      incident data.  LogisticRegression is provided as a fast linear baseline.

  class_weight="balanced"
      Incidents are rare.  Without balancing, a classifier that always predicts
      "no incident" scores high accuracy but is useless.  Balanced weighting
      upweights the minority class proportionally to its scarcity.

  No data leakage in the scaler
      StandardScaler is fit only on X_train and then applied to X_test.
      Fitting on the whole dataset before splitting would leak test-set
      statistics into the scaler — a subtle but real form of leakage.
      sklearn's Pipeline handles this automatically.

  TrainResult dataclass
      Bundles the fitted model, test arrays, and metrics into one object so
      callers can do threshold-tuning, SHAP analysis, or plotting without
      re-running training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of windows (from the end of the timeline) reserved for testing.
TEST_FRAC = 0.20

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Everything returned by train_model() bundled into one place."""
    model: Pipeline          # fitted sklearn Pipeline (scaler + classifier)
    X_train: pd.DataFrame    # training feature matrix
    X_test: pd.DataFrame     # held-out feature matrix
    y_train: np.ndarray      # training labels
    y_test: np.ndarray       # held-out true labels
    y_pred: np.ndarray       # class predictions on the test set (0 or 1)
    y_prob: np.ndarray       # predicted probability of incident on the test set
    split_index: int         # row index where train ends and test begins
    metrics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Step 1 — Time-aware split
# ---------------------------------------------------------------------------

def time_split(
    X: pd.DataFrame,
    y: np.ndarray,
    test_frac: float = TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    """
    Split feature matrix and labels into train and test by position.

    Because windows are already time-ordered (windowing.py preserves order),
    a positional cut is a time-aware cut.  No datetime column is needed.

    Parameters
    ----------
    X         : feature DataFrame, one row per window, time-ordered
    y         : binary label array, same length as X
    test_frac : fraction of rows to hold out from the *end* of the series

    Returns
    -------
    X_train, X_test, y_train, y_test, split_index
    """
    n = len(X)
    split_index = int(n * (1 - test_frac))

    X_train = X.iloc[:split_index].reset_index(drop=True)
    X_test  = X.iloc[split_index:].reset_index(drop=True)
    y_train = y[:split_index]
    y_test  = y[split_index:]

    log.info(
        "Time split — train: %d windows (rows 0–%d) | test: %d windows (rows %d–%d)",
        len(X_train), split_index - 1,
        len(X_test),  split_index, n - 1,
    )
    return X_train, X_test, y_train, y_test, split_index


# ---------------------------------------------------------------------------
# Step 2 — Build model pipeline
# ---------------------------------------------------------------------------

def build_pipeline(model_type: str = "rf") -> Pipeline:
    """
    Assemble a preprocessing + classifier Pipeline.

    Steps
    -----
    imputer
        Fills any NaNs with the column median.  The engineered features from
        features.py should not produce NaNs, but this is cheap insurance and
        makes the pipeline safe for real sensor data that may have gaps.
    scaler
        StandardScaler — fit on train only (Pipeline handles this).
        Has no effect on RandomForest (scale-invariant) but is essential for
        LogisticRegression's gradient convergence.
    clf
        The classifier.  Both options use class_weight="balanced" to handle
        the rarity of incidents without manual oversampling.

    Parameters
    ----------
    model_type : "rf" for RandomForestClassifier, "lr" for LogisticRegression
    """
    if model_type == "rf":
        classifier = RandomForestClassifier(
            n_estimators=300,        # enough trees for stable importances
            min_samples_leaf=5,      # avoids memorising tiny noisy leaves
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,               # use all CPU cores
        )
    elif model_type == "lr":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'rf' or 'lr'.")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     classifier),
    ])


# ---------------------------------------------------------------------------
# Step 3 — Evaluate predictions
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute classification metrics and log them.

    ROC-AUC is the primary metric: it measures ranking quality across all
    thresholds and is not fooled by class imbalance the way accuracy is.
    Precision/recall/F1 for the positive class (incidents) are also returned
    so the caller can choose an operating point.

    Returns a flat dict so values can be logged, saved, or compared easily.
    """
    auc = roc_auc_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)

    log.info("ROC-AUC : %.4f", auc)
    log.info("\n%s", classification_report(y_true, y_pred))

    # classification_report keys the positive class as the string "1"
    pos = report.get("1", {})
    return {
        "roc_auc":    auc,
        "precision":  pos.get("precision"),
        "recall":     pos.get("recall"),
        "f1":         pos.get("f1-score"),
        "support":    pos.get("support"),
    }


# ---------------------------------------------------------------------------
# Step 4 — Log feature importances (RandomForest only)
# ---------------------------------------------------------------------------

def log_feature_importances(pipeline: Pipeline, feature_names: list[str], top_n: int = 10) -> None:
    """
    Log the top-N feature importances from the fitted RandomForest.

    Feature importances are the mean decrease in impurity across all trees.
    They tell us which (sensor, statistic) pairs the forest found most useful
    for splitting — a quick sanity check that the model learned real signal
    (e.g. sensor_a__slope should rank highly if stress ramp-ups are present).
    """
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return  # LogisticRegression doesn't have this attribute

    importances = pd.Series(clf.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    log.info("Top-%d feature importances:\n%s", top_n, top.to_string())


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    model_type: str = "rf",
    test_frac: float = TEST_FRAC,
) -> TrainResult:
    """
    Train and evaluate an incident prediction classifier.

    This function expects the output of features.extract_features() as X and
    the label array from windowing.WindowedDataset.y as y.  Both must be in
    the same chronological order — no shuffling should occur between windowing
    and this call.

    Parameters
    ----------
    X          : feature DataFrame from features.extract_features()
    y          : binary label array from WindowedDataset.y
    model_type : "rf" (RandomForestClassifier) or "lr" (LogisticRegression)
    test_frac  : fraction of the timeline to reserve for testing

    Returns
    -------
    TrainResult
        Contains the fitted pipeline, train/test splits, predictions,
        probabilities, and evaluation metrics.
    """
    # 1. Split by time position — no shuffling
    X_train, X_test, y_train, y_test, split_index = time_split(X, y, test_frac)

    # 2. Build pipeline (imputer → scaler → classifier)
    pipeline = build_pipeline(model_type)

    # 3. Fit on training data only
    #    Pipeline.fit() calls imputer.fit_transform → scaler.fit_transform → clf.fit
    #    so scaler statistics are never contaminated by test-set values.
    log.info("Training %s on %d windows…", model_type.upper(), len(X_train))
    pipeline.fit(X_train, y_train)

    # 4. Predict on test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # P(incident)

    # 5. Evaluate
    metrics = evaluate(y_test, y_pred, y_prob)

    # 6. Show which features mattered (RF only)
    log_feature_importances(pipeline, list(X.columns))

    return TrainResult(
        model=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        split_index=split_index,
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# CLI — runs the full pipeline end-to-end for a quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data.synthetic_generator import generate_dataset
    from src.features import extract_features
    from src.windowing import make_windows

    parser = argparse.ArgumentParser(description="Train the incident prediction model.")
    parser.add_argument("--model", choices=["rf", "lr"], default="rf")
    parser.add_argument("--window", type=int, default=30, help="Window size W (default: 30)")
    parser.add_argument("--horizon", type=int, default=10, help="Prediction horizon H (default: 10)")
    parser.add_argument("--n", type=int, default=10_000, help="Timesteps to generate (default: 10000)")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC)
    args = parser.parse_args()

    log.info("Generating synthetic data (%d timesteps)…", args.n)
    raw_df = generate_dataset(n_steps=args.n)

    sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
    dataset = make_windows(raw_df, feature_cols=sensor_cols, label_col="incident",
                           window_size=args.window, horizon=args.horizon)

    log.info("Extracting features…")
    X = extract_features(dataset)
    y = dataset.y

    result = train_model(X, y, model_type=args.model, test_frac=args.test_frac)

    print("\n── Final metrics ──────────────────────────")
    for k, v in result.metrics.items():
        print(f"  {k:<12} {v:.4f}" if isinstance(v, float) else f"  {k:<12} {v}")
