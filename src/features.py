"""
features.py — Tabular Feature Extraction from Sliding Windows
=============================================================
Takes the 3-D window array produced by windowing.py and computes a flat
table of hand-crafted features — one row per window, one column per
(metric, statistic) pair.

Features computed per sensor/metric
------------------------------------
  mean   — average level over the window
  std    — variability / spread
  min    — lowest reading in the window
  max    — highest reading in the window
  last   — most recent value (the reading right before prediction time)
  slope  — linear trend (OLS slope) — positive means rising, negative falling

These six statistics are computed independently for every feature column
supplied, giving  n_features × 6  output columns total.

Why these six?
--------------
  They cover the three things a classifier needs to detect pre-incident stress:
    1. Level shift   → mean, last, min, max
    2. Volatility    → std
    3. Direction     → slope (is the sensor rising toward a threshold?)

  They are also easy to explain in an interview without any domain jargon.

Column naming convention
------------------------
  "<original_col>__<stat>"  e.g.  "sensor_a__mean", "sensor_b__slope"
  The double-underscore separator makes it trivial to split the name back
  into (column, statistic) if needed later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.windowing import WindowedDataset


# ---------------------------------------------------------------------------
# Individual statistic extractors
# Each function receives a 2-D array of shape (n_windows, W) — one sensor's
# readings across all windows — and returns a 1-D array of shape (n_windows,).
# Keeping them separate makes unit-testing and swapping easy.
# ---------------------------------------------------------------------------

def compute_mean(windows: np.ndarray) -> np.ndarray:
    """Mean reading over each window."""
    return windows.mean(axis=1)


def compute_std(windows: np.ndarray) -> np.ndarray:
    """
    Standard deviation over each window.

    ddof=0 (population std) is used because the window is the full population
    of observations we have — there is no larger sample being estimated.
    """
    return windows.std(axis=1, ddof=0)


def compute_min(windows: np.ndarray) -> np.ndarray:
    """Minimum reading over each window."""
    return windows.min(axis=1)


def compute_max(windows: np.ndarray) -> np.ndarray:
    """Maximum reading over each window."""
    return windows.max(axis=1)


def compute_last(windows: np.ndarray) -> np.ndarray:
    """
    Most recent reading in each window (the value at time t-1).

    This is the single most recent observation before the prediction horizon
    begins, so it carries the highest recency weight of any statistic.
    """
    return windows[:, -1]


def compute_slope(windows: np.ndarray) -> np.ndarray:
    """
    Ordinary-least-squares slope of the readings over each window.

    Fits a line  y = slope * x + intercept  where x = [0, 1, …, W-1].
    A positive slope means the sensor is trending upward; negative means
    it is falling.

    The closed-form OLS formula is used directly — no scipy/statsmodels
    dependency needed:
        slope = (n * Σ(x·y) - Σx · Σy) / (n * Σ(x²) - (Σx)²)

    Because x is the same for every window we pre-compute its summary
    statistics once and reuse them.
    """
    n_windows, W = windows.shape
    x = np.arange(W, dtype=float)          # time indices: 0, 1, …, W-1

    # Pre-compute x summaries (same for all windows)
    sum_x  = x.sum()                       # scalar
    sum_x2 = (x ** 2).sum()               # scalar
    n      = float(W)

    # Per-window y summaries  — shape (n_windows,)
    sum_y  = windows.sum(axis=1)
    sum_xy = windows @ x                   # dot product along time axis

    numerator   = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x2 - sum_x ** 2  # constant across windows

    # denominator is zero only when W == 1; guard against division by zero
    if denominator == 0:
        return np.zeros(n_windows, dtype=float)

    return numerator / denominator


# ---------------------------------------------------------------------------
# Statistic registry
# Each entry is (suffix, function).  Order here becomes column order in output.
# To add a new statistic, append one line to this list.
# ---------------------------------------------------------------------------

_STATISTICS: list[tuple[str, callable]] = [
    ("mean",  compute_mean),
    ("std",   compute_std),
    ("min",   compute_min),
    ("max",   compute_max),
    ("last",  compute_last),
    ("slope", compute_slope),
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(dataset: WindowedDataset) -> pd.DataFrame:
    """
    Compute tabular features from a ``WindowedDataset``.

    For every feature column in the dataset, each statistic in ``_STATISTICS``
    is applied to the (n_windows, W) slice for that column.  Results are
    assembled into a single DataFrame.

    Parameters
    ----------
    dataset : WindowedDataset
        Output of ``windowing.make_windows``.  Must have at least one window.

    Returns
    -------
    pd.DataFrame
        Shape: (n_windows, n_feature_cols × 6).
        Column names follow the pattern ``"<col>__<stat>"``.

    Example column names (four sensors, six stats each = 24 columns)
    -----------------------------------------------------------------
        sensor_a__mean,  sensor_a__std,  sensor_a__min, …
        sensor_b__mean,  sensor_b__std,  …
        sensor_c__mean,  …
        sensor_d__mean,  …  sensor_d__slope

    Example
    -------
    >>> from data.synthetic_generator import generate_dataset
    >>> from src.windowing import make_windows
    >>> df = generate_dataset(n_steps=10_000)
    >>> sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
    >>> ds = make_windows(df, feature_cols=sensor_cols, label_col="incident",
    ...                   window_size=30, horizon=10)
    >>> features_df = extract_features(ds)
    >>> features_df.shape
    (9961, 24)
    """
    columns: dict[str, np.ndarray] = {}

    for col_idx, col_name in enumerate(dataset.feature_names):
        # Extract the (n_windows, W) slice for this single sensor/metric
        sensor_windows = dataset.X[:, :, col_idx]

        for stat_name, stat_fn in _STATISTICS:
            col_key = f"{col_name}__{stat_name}"
            columns[col_key] = stat_fn(sensor_windows)

    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from data.synthetic_generator import generate_dataset
    from src.windowing import make_windows

    df = generate_dataset(n_steps=10_000)
    sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]

    ds = make_windows(df, feature_cols=sensor_cols, label_col="incident",
                      window_size=30, horizon=10)

    features_df = extract_features(ds)
    print(f"Feature matrix: {features_df.shape[0]:,} rows × {features_df.shape[1]} columns")
    print(f"Columns: {list(features_df.columns)}")
    print(features_df.describe().round(3))
