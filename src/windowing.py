"""
windowing.py — Sliding-Window Supervised Dataset Builder
=========================================================
Converts a multivariate time series into (X, y) pairs suitable for a
supervised classifier.

Terminology
-----------
  W  (window_size) : number of past timesteps used as input features
  H  (horizon)     : number of future timesteps to look ahead for a label

For each position t in the series:
  X[t]  = sensor readings over [t - W, t)          — shape (W, n_features)
  y[t]  = 1 if any incident occurs in [t, t + H)   — scalar 0 or 1

                          ◄── W ──►   ◄─ H ─►
  timeline:  … ─────────[==========][→→→→→→]─ …
                          input        label
                          window       horizon

The first valid output is at index W (needs W past steps).
The last  valid output is at index n - H (needs H future steps).
Total windows produced: n - W - H + 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class WindowedDataset:
    """
    Container returned by ``make_windows``.

    Attributes
    ----------
    X : np.ndarray, shape (n_windows, W, n_features)
        3-D array of input windows.  Each slice X[i] is one (W × n_features)
        snapshot of the past.
    y : np.ndarray, shape (n_windows,)
        Binary label per window.  1 if any incident occurs in the next H
        timesteps after the window ends, else 0.
    feature_names : list[str]
        Names of the feature columns, in the same order as the last axis of X.
    window_size : int
        W used to build the dataset.
    horizon : int
        H used to build the dataset.
    """
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    window_size: int
    horizon: int

    def __repr__(self) -> str:
        pos = int(self.y.sum())
        neg = len(self.y) - pos
        return (
            f"WindowedDataset("
            f"windows={len(self.y):,}, "
            f"W={self.window_size}, H={self.horizon}, "
            f"features={len(self.feature_names)}, "
            f"positive={pos:,} ({pos / len(self.y):.1%}), "
            f"negative={neg:,} ({neg / len(self.y):.1%}))"
        )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    window_size: int,
    horizon: int,
) -> WindowedDataset:
    """
    Build a sliding-window supervised dataset from a time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Time-ordered DataFrame.  Rows must already be sorted by time;
        no sorting is performed here.
    feature_cols : list[str]
        Column names to use as input features.  All must be present in ``df``.
    label_col : str
        Column name of the binary incident flag (0 or 1).
    window_size : int  (W)
        Number of past timesteps in each input window.  Must be >= 1.
    horizon : int  (H)
        Number of future timesteps to look ahead when computing the label.
        y = 1 if *any* of the next H rows has label_col == 1.  Must be >= 1.

    Returns
    -------
    WindowedDataset
        Named container with arrays X (shape n_windows × W × n_features)
        and y (shape n_windows).

    Raises
    ------
    ValueError
        If any requested column is missing, parameters are out of range, or
        there are not enough rows to form at least one window.

    Examples
    --------
    >>> from data.synthetic_generator import generate_dataset
    >>> df = generate_dataset(n_steps=10_000)
    >>> sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]
    >>> ds = make_windows(df, feature_cols=sensor_cols, label_col="incident",
    ...                   window_size=30, horizon=10)
    >>> ds.X.shape
    (9961, 30, 4)
    >>> ds.y.shape
    (9961,)
    """
    # --- validate inputs ---
    _validate_inputs(df, feature_cols, label_col, window_size, horizon)

    n = len(df)
    n_features = len(feature_cols)

    # Extract raw numpy arrays once — avoids repeated pandas overhead inside the loop
    feature_array = df[feature_cols].to_numpy(dtype=float)   # shape (n, n_features)
    label_array   = df[label_col].to_numpy(dtype=int)        # shape (n,)

    # Number of windows we can produce
    # First window starts at index W (needs rows 0 … W-1 as input).
    # Last  window starts at index n - H (its horizon covers rows n-H … n-1).
    n_windows = n - window_size - horizon + 1

    # Pre-allocate output arrays
    X = np.empty((n_windows, window_size, n_features), dtype=float)
    y = np.empty(n_windows, dtype=int)

    for i in range(n_windows):
        # Input: rows [i, i + W)
        X[i] = feature_array[i : i + window_size]

        # Label: any incident in rows [i + W, i + W + H)
        # Using max() is equivalent to "any" for a binary array and avoids
        # creating a temporary boolean array on every iteration.
        y[i] = int(label_array[i + window_size : i + window_size + horizon].max())

    return WindowedDataset(
        X=X,
        y=y,
        feature_names=list(feature_cols),
        window_size=window_size,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_windows(ds: WindowedDataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten the 3-D window array into a 2-D feature matrix.

    Reshapes X from (n_windows, W, n_features) to (n_windows, W * n_features)
    so it can be passed directly to sklearn estimators that expect a 2-D input.

    Column order: all features for timestep 0, then all for timestep 1, etc.
    (i.e., C-order / row-major flattening of the last two axes)

    Returns
    -------
    X_flat : np.ndarray, shape (n_windows, W * n_features)
    y      : np.ndarray, shape (n_windows,)
    """
    n_windows, W, n_features = ds.X.shape
    X_flat = ds.X.reshape(n_windows, W * n_features)
    return X_flat, ds.y


def _validate_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    window_size: int,
    horizon: int,
) -> None:
    """Raise ValueError with a clear message if any argument is invalid."""
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns not found in DataFrame: {missing_features}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}.")

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")

    min_rows = window_size + horizon
    if len(df) < min_rows:
        raise ValueError(
            f"DataFrame has {len(df)} rows but at least {min_rows} are needed "
            f"(window_size={window_size} + horizon={horizon})."
        )


# ---------------------------------------------------------------------------
# CLI — quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from any working directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.synthetic_generator import generate_dataset

    df = generate_dataset(n_steps=10_000)
    sensor_cols = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]

    ds = make_windows(df, feature_cols=sensor_cols, label_col="incident",
                      window_size=30, horizon=10)
    print(ds)

    X_flat, y = flatten_windows(ds)
    print(f"Flattened X shape : {X_flat.shape}")
    print(f"Label distribution: {int(y.sum())} positive / {int((y == 0).sum())} negative")
