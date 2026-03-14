"""
synthetic_generator.py — Synthetic Multivariate Incident Dataset
================================================================
Generates a realistic multivariate time series for incident prediction.

Sensor relationships
--------------------
  sensor_b  : autonomous AR(1) process; the "upstream" signal
  sensor_a  : influenced by sensor_b (with a short lag) + own noise
  sensor_c  : influenced by both sensor_a and sensor_b + own noise
  sensor_d  : independent sensor; no causal link to others

Incident structure
------------------
  Incidents do not appear as isolated spikes.  They are generated as
  contiguous windows (clusters) so that the flag stays high for several
  consecutive timesteps, matching real operational events.

  Before each incident window there is a "stress build-up" period where
  all causally-linked sensors drift upward.  This makes the incidents
  somewhat predictable from the recent history — the key property needed
  for a useful classifier.

Usage
-----
  python data/synthetic_generator.py                 # saves data/synthetic_data.csv
  python data/synthetic_generator.py --n 5000 --seed 7

  Or import directly:
      from data.synthetic_generator import generate_dataset
      df = generate_dataset(n_steps=10_000, seed=42)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — tweak these to change the "feel" of the data
# ---------------------------------------------------------------------------

# Baseline (non-incident) noise amplitude for each sensor
NOISE_STD = {
    "sensor_a": 0.5,
    "sensor_b": 0.6,
    "sensor_c": 0.4,
    "sensor_d": 0.7,
}

# How strongly each sensor responds to the pre-incident stress signal
STRESS_AMPLITUDE = {
    "sensor_a": 2.5,
    "sensor_b": 2.0,
    "sensor_c": 1.8,
    "sensor_d": 0.3,   # sensor_d is largely unaffected — a useful "distractor"
}

# AR(1) auto-correlation coefficient for sensor_b (0 = white noise, 1 = random walk)
AR_COEFF = 0.7

# How much sensor_b feeds into sensor_a
SENSOR_B_TO_A_WEIGHT = 0.6

# How much sensor_a and sensor_b each feed into sensor_c
SENSOR_A_TO_C_WEIGHT = 0.5
SENSOR_B_TO_C_WEIGHT = 0.3

# Lag (in timesteps) with which sensor_b influences sensor_a
LAG_B_TO_A = 3


# ---------------------------------------------------------------------------
# Step 1 — Generate incident windows
# ---------------------------------------------------------------------------

def _generate_incident_windows(
    n_steps: int,
    n_incidents: int,
    incident_duration_range: tuple[int, int],
    min_gap: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Place ``n_incidents`` non-overlapping incident windows inside [0, n_steps).

    Each window has a random duration drawn from ``incident_duration_range``
    and is separated from its neighbours by at least ``min_gap`` steps.

    Returns a list of (start, end) index pairs (end is exclusive).
    """
    windows: list[tuple[int, int]] = []
    cursor = min_gap  # leave breathing room at the very start

    for _ in range(n_incidents):
        duration = int(rng.integers(incident_duration_range[0], incident_duration_range[1] + 1))
        # latest possible start that still fits before the end of the series
        latest_start = n_steps - duration - min_gap
        if cursor > latest_start:
            break  # no room for more incidents
        start = int(rng.integers(cursor, latest_start + 1))
        end = start + duration
        windows.append((start, end))
        cursor = end + min_gap  # enforce minimum gap to the next window

    return windows


# ---------------------------------------------------------------------------
# Step 2 — Build the stress signal
# ---------------------------------------------------------------------------

def _build_stress_signal(
    n_steps: int,
    incident_windows: list[tuple[int, int]],
    buildup_steps: int,
) -> np.ndarray:
    """
    Return a continuous stress signal in [0, 1] that ramps up linearly over
    ``buildup_steps`` timesteps before each incident window, stays at 1.0
    during the incident, then drops to 0 immediately after.

    This encodes the "pre-incident stress pattern" requirement: sensors start
    drifting *before* the incident flag turns on, giving a classifier something
    predictive to learn from.
    """
    stress = np.zeros(n_steps, dtype=float)

    for start, end in incident_windows:
        buildup_start = max(0, start - buildup_steps)

        # Ramp from 0 → 1 over the buildup period
        ramp_length = start - buildup_start
        if ramp_length > 0:
            ramp = np.linspace(0.0, 1.0, ramp_length)
            stress[buildup_start:start] = np.maximum(stress[buildup_start:start], ramp)

        # Hold at 1.0 during the incident itself
        stress[start:end] = 1.0

    return stress


# ---------------------------------------------------------------------------
# Step 3 — Generate individual sensors
# ---------------------------------------------------------------------------

def _generate_sensor_b(
    n_steps: int,
    stress: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    sensor_b: an AR(1) process centred at 0, plus a stress-driven drift.

    sensor_b[t] = AR_COEFF * sensor_b[t-1] + noise + stress_component
    """
    noise = rng.normal(0, NOISE_STD["sensor_b"], size=n_steps)
    values = np.zeros(n_steps)
    for t in range(1, n_steps):
        values[t] = AR_COEFF * values[t - 1] + noise[t] + STRESS_AMPLITUDE["sensor_b"] * stress[t]
    return values


def _generate_sensor_a(
    sensor_b: np.ndarray,
    stress: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    sensor_a: partly driven by sensor_b (with a short lag) plus its own noise
    and stress response.

    sensor_a[t] = weight * sensor_b[t - LAG] + own_noise + stress_component

    The lag means sensor_b changes are visible in sensor_a a few steps later,
    which is a realistic causal structure.
    """
    n_steps = len(sensor_b)
    noise = rng.normal(0, NOISE_STD["sensor_a"], size=n_steps)
    values = np.zeros(n_steps)
    for t in range(n_steps):
        lag_b = sensor_b[t - LAG_B_TO_A] if t >= LAG_B_TO_A else 0.0
        values[t] = (
            SENSOR_B_TO_A_WEIGHT * lag_b
            + noise[t]
            + STRESS_AMPLITUDE["sensor_a"] * stress[t]
        )
    return values


def _generate_sensor_c(
    sensor_a: np.ndarray,
    sensor_b: np.ndarray,
    stress: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    sensor_c: a downstream aggregation of sensor_a and sensor_b, plus noise.

    sensor_c[t] = w_a * sensor_a[t] + w_b * sensor_b[t] + noise + stress_component

    Because it inherits from both upstream sensors, its signal is the "richest"
    for detecting pre-incident patterns.
    """
    n_steps = len(sensor_a)
    noise = rng.normal(0, NOISE_STD["sensor_c"], size=n_steps)
    values = (
        SENSOR_A_TO_C_WEIGHT * sensor_a
        + SENSOR_B_TO_C_WEIGHT * sensor_b
        + noise
        + STRESS_AMPLITUDE["sensor_c"] * stress
    )
    return values


def _generate_sensor_d(
    n_steps: int,
    stress: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    sensor_d: an independent sensor with no causal link to a/b/c.

    It has a mild, noisy stress response so it is not completely useless, but
    its correlation with incidents is weak.  This tests whether the classifier
    can identify which sensors actually matter.
    """
    noise = rng.normal(0, NOISE_STD["sensor_d"], size=n_steps)
    # Add a slow sinusoidal drift to make it look "sensor-like"
    t = np.arange(n_steps)
    drift = 0.3 * np.sin(2 * np.pi * t / 500)
    values = drift + noise + STRESS_AMPLITUDE["sensor_d"] * stress
    return values


# ---------------------------------------------------------------------------
# Step 4 — Build the incident label array
# ---------------------------------------------------------------------------

def _build_incident_labels(
    n_steps: int,
    incident_windows: list[tuple[int, int]],
) -> np.ndarray:
    """Return a binary array: 1 during every incident window, 0 elsewhere."""
    labels = np.zeros(n_steps, dtype=int)
    for start, end in incident_windows:
        labels[start:end] = 1
    return labels


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def generate_dataset(
    n_steps: int = 10_000,
    n_incidents: int = 15,
    incident_duration_range: tuple[int, int] = (8, 25),
    buildup_steps: int = 50,
    min_gap: int = 120,
    start_time: str = "2023-01-01",
    freq: str = "1min",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic multivariate time series for incident prediction.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to generate.  Default: 10,000.
    n_incidents : int
        Number of incident windows to inject.  Default: 15.
    incident_duration_range : tuple[int, int]
        (min, max) duration in timesteps for each incident window.
    buildup_steps : int
        How many timesteps *before* each incident the stress ramp begins.
        Larger values give the classifier a longer look-ahead window.
    min_gap : int
        Minimum number of non-incident timesteps between consecutive windows.
    start_time : str
        ISO-8601 start datetime for the timestamp index.
    freq : str
        Pandas frequency string for the timestamp spacing (e.g. "1min", "5min").
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp, sensor_a, sensor_b, sensor_c, sensor_d, incident
        Indexed 0 … n_steps-1.

    Notes
    -----
    Sensor causal graph (at generation time):
        sensor_b  ──(lag 3)──►  sensor_a
        sensor_b  ─────────────►  sensor_c
        sensor_a  ─────────────►  sensor_c
        sensor_d  (independent)

    All sensors are additionally nudged upward by a shared stress signal that
    ramps up in the ``buildup_steps`` window before each incident.
    """
    rng = np.random.default_rng(seed)

    # --- incident windows & stress signal ---
    incident_windows = _generate_incident_windows(
        n_steps=n_steps,
        n_incidents=n_incidents,
        incident_duration_range=incident_duration_range,
        min_gap=min_gap,
        rng=rng,
    )
    log.info("Generated %d incident windows.", len(incident_windows))

    stress = _build_stress_signal(
        n_steps=n_steps,
        incident_windows=incident_windows,
        buildup_steps=buildup_steps,
    )

    # --- sensors ---
    sensor_b = _generate_sensor_b(n_steps, stress, rng)
    sensor_a = _generate_sensor_a(sensor_b, stress, rng)
    sensor_c = _generate_sensor_c(sensor_a, sensor_b, stress, rng)
    sensor_d = _generate_sensor_d(n_steps, stress, rng)

    # --- labels ---
    incident = _build_incident_labels(n_steps, incident_windows)

    # --- assemble DataFrame ---
    timestamps = pd.date_range(start=start_time, periods=n_steps, freq=freq)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_a": sensor_a,
            "sensor_b": sensor_b,
            "sensor_c": sensor_c,
            "sensor_d": sensor_d,
            "incident": incident,
        }
    )

    incident_rate = incident.mean() * 100
    log.info(
        "Dataset: %d rows | %d incident windows | %.1f%% incident timesteps",
        n_steps,
        len(incident_windows),
        incident_rate,
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic incident time-series data.")
    parser.add_argument("--n", type=int, default=10_000, help="Number of timesteps (default: 10000)")
    parser.add_argument("--incidents", type=int, default=15, help="Number of incident windows (default: 15)")
    parser.add_argument("--buildup", type=int, default=50, help="Stress buildup steps before each incident (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "synthetic_data.csv",
        help="Output CSV path (default: data/synthetic_data.csv)",
    )
    args = parser.parse_args()

    df = generate_dataset(
        n_steps=args.n,
        n_incidents=args.incidents,
        buildup_steps=args.buildup,
        seed=args.seed,
    )

    df.to_csv(args.out, index=False)
    print(f"Saved {len(df):,} rows to {args.out}")
    print(df.head())
    print(f"\nIncident rate: {df['incident'].mean():.2%}")
