"""
main.py — Incident Prediction Pipeline
=======================================
End-to-end script: generate data → window → extract features → train → evaluate.

Usage
-----
  python main.py                        # defaults
  python main.py --model lr             # logistic regression baseline
  python main.py --window 20 --horizon 5
  python main.py --n 5000 --threshold 0.3
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from the project root regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.synthetic_generator import generate_dataset
from src.evaluate import alert_summary, compare_thresholds, compute_metrics
from src.features import extract_features
from src.train import train_model
from src.windowing import make_windows

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

SENSOR_COLS = ["sensor_a", "sensor_b", "sensor_c", "sensor_d"]


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    width = 52
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print('─' * width)


def print_dict(d: dict) -> None:
    for k, v in d.items():
        value = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k:<20} {value}")


# ---------------------------------------------------------------------------
# Pipeline steps (each is one clear action)
# ---------------------------------------------------------------------------

def step_generate(n_steps: int, seed: int):
    print_section("1 · Generate synthetic data")
    df = generate_dataset(n_steps=n_steps, seed=seed)
    print(f"  rows           {len(df):,}")
    print(f"  columns        {list(df.columns)}")
    print(f"  incident rate  {df['incident'].mean():.2%}")
    return df


def step_window(df, window_size: int, horizon: int):
    print_section("2 · Sliding windows")
    dataset = make_windows(
        df,
        feature_cols=SENSOR_COLS,
        label_col="incident",
        window_size=window_size,
        horizon=horizon,
    )
    print(f"  {dataset}")
    return dataset


def step_extract(dataset):
    print_section("3 · Feature extraction")
    X = extract_features(dataset)
    print(f"  windows        {X.shape[0]:,}")
    print(f"  features       {X.shape[1]}  ({', '.join(X.columns[:4])} …)")
    return X


def step_train(X, y, model_type: str, test_frac: float):
    print_section("4 · Training")
    print(f"  model          {model_type.upper()}")
    print(f"  train/test     {1 - test_frac:.0%} / {test_frac:.0%}")
    result = train_model(X, y, model_type=model_type, test_frac=test_frac)
    print(f"  train windows  {len(result.X_train):,}")
    print(f"  test windows   {len(result.X_test):,}")
    return result


def step_evaluate(result, threshold: float):
    print_section("5 · Evaluation")
    metrics = compute_metrics(result.y_test, result.y_prob, threshold=threshold)
    print_dict(metrics)

    print_section("6 · Alert breakdown")
    summary = alert_summary(result.y_test, result.y_prob, threshold=threshold)
    print_dict(summary)

    print_section("7 · Threshold comparison")
    comparison = compare_thresholds(result.y_test, result.y_prob)
    print(comparison.to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Incident prediction pipeline.")
    p.add_argument("--n",         type=int,   default=10_000, help="Timesteps to generate (default: 10000)")
    p.add_argument("--window",    type=int,   default=30,     help="Window size W (default: 30)")
    p.add_argument("--horizon",   type=int,   default=10,     help="Prediction horizon H (default: 10)")
    p.add_argument("--model",     choices=["rf", "lr"], default="rf", help="Classifier (default: rf)")
    p.add_argument("--test-frac", type=float, default=0.20,   help="Test fraction (default: 0.20)")
    p.add_argument("--threshold", type=float, default=0.50,   help="Alert threshold (default: 0.50)")
    p.add_argument("--seed",      type=int,   default=42,     help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df      = step_generate(args.n, args.seed)
    dataset = step_window(df, args.window, args.horizon)
    X       = step_extract(dataset)
    result  = step_train(X, dataset.y, args.model, args.test_frac)
    step_evaluate(result, args.threshold)

    print("\n")


if __name__ == "__main__":
    main()
