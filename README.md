# JB1 Incident Prediction

A small end-to-end pipeline for predicting incidents from 4 sensors.

---

## Problem statement

Given a stream of sensor readings, predict whether an incident will occur in the
next H timesteps. The output is a binary alert.

---

Here's my thought process:

synthetic data  →  sliding windows  →  feature extraction  →  train  →  evaluate
---

The pipeline runs end-to-end from `main.py`, but, this is modular.

---

## Synthetic data design

the dataset is generated in `data/synthetic_generator.py`.

Four sensors are simulated:

| Sensor   | Behaviour |
|----------|-----------|
| sensor_b | Autonomous AR(1) process - the upstream driver |
| sensor_a | Influenced by sensor_b with a 3-step lag, plus own noise |
| sensor_c | aggregation of sensor_a and sensor_b |
| sensor_d | Independent - mostly noise with a slow sinusoidal drift |

Incidents are placed as contiguous windows (8–25 steps each) with a minimum gap
between them, so they behave like real events rather than isolated spikes.

--
Trends
--

For the 50 timesteps before each incident window, a shared stress signal ramps from 0 to 1 and pushes all
linked sensors upward. 
This creates the inherent predictability that classifiers use.


`sensor_d` is given only a tiny stress response, making it a distractor.
A well-trained model should learn to down-weight it.

---

## Sliding-window formulation

Raw time series data is not directly consumable.
`src/windowing.py` converts it into (X, y) pairs:

```
timeline:  … ─────────[==========][→→→→→→]─ …
                        ◄── W ──►   ◄─ H ─►
                        input        label
                        window       horizon
```

- **X[i]** - the sensor readings over the W timesteps ending at position i
- **y[i]** - 1 if any incident occurs in the next H timesteps, else 0

With W=30 and H=10 on 10,000 timesteps, this produces 9,961 labelled windows.


N.b. this has the affect of detecting incidents premptively. Intuitively, this is useful in practice.

---

## Feature extraction

`src/features.py` converts each (W × n_sensors) window into a flat row of
summary statistics. Six statistics are computed per sensor:

| Statistic | What it captures | Why it's included |
|-----------|-----------------|-------------------|
| mean      | Average level over the window | Detects sustained baseline shifts during stress build-up |
| std       | spread | Elevated variance often precedes instability |
| min       | Lowest reading | Catches brief dips that mean/last would smooth over |
| max       | Highest reading | Same as above |
| last      | Most recent value before the horizon | To predict |
| slope     | Linear trend | Directly encodes whether the sensor is rising toward a threshold |

With 4 sensors × 6 statistics = **24 features** per window.

These six were chosen because together they cover the three things that matter
for detecting the pre-incident stress ramp: level (mean, last, min, max),
volatility (std), and direction (slope). More complex features - rolling z-scores, autocorrelation
coefficients - would add noise for little gain on a 30-step window.

The slope is the most discriminating feature in this dataset because the data
has a consistent upward trend in the W steps before
an incident. Mean and last are the next most useful.

---

## Model choice

Two classifiers are available, selectable via `--model`:

**RandomForestClassifier (default)**
- Handles the 24-column feature table without rescaling
- Gives feature importances
- `class_weight="balanced"` compensates for the ~4% positive rate without oversampling
- `min_samples_leaf=5` prevents overfitting to noisy leaf nodes

**LogisticRegression (baseline)**
- Much faster to train
- Linear decision boundary - useful for checking whether the RF's nonlinearity
  is actually buying anything
- Requires StandardScaler, which the pipeline applies automatically

Both are wrapped in a `sklearn.Pipeline` that includes a median imputer and
standard scaler. The pipeline is fit only on training data, so test-set
statistics never contaminate the scaler.

---

## Train/test split

A **time-aware split** is used: the first 80% of windows go to training, the
last 20% go to testing. No shuffling.

This matters because adjacent windows share most of their timesteps. A random
split would leak near-identical windows into both train and test, producing
inflated metrics that would collapse on real deployment. 

---

## Evaluation metrics

```
python main.py
```

Reports at the chosen threshold:

| Metric    | Why it's here |
|-----------|---------------|
| precision | Of the alerts that fired, how many were real incidents? |
| recall    | Of all real incidents, how many triggered an alert? |
| f1        | Harmonic mean - single number balancing the two |
| roc_auc   | Threshold-independent ranking quality |
| n_alerts  | How noisy would this threshold be in production? |
| miss_rate | Fraction of real incidents that were silently missed |

**Why not accuracy?** With ~4% positive rate, a classifier that always predicts
"no incident" achieves 96% accuracy. Accuracy is useless here. Precision and
recall force the model to actually find incidents.

**Why ROC-AUC as the headline metric?** It measures how well the model ranks
incident windows above non-incident ones across *all* thresholds simultaneously.
It does not depend on picking a threshold, and it is not fooled by class
imbalance. A score near 1.0 means the model's probability estimates are
well-ordered even if the absolute values are miscalibrated.

**Why report both precision and recall separately rather than just F1?** Because
the right trade-off is application-specific. F1 assumes precision and recall are
equally important, which is rarely true. An operator choosing a threshold needs
to see both numbers, not their average.

---

## Thresholding and alerting

The classifier outputs P(incident) ∈ [0, 1]. Converting that to a binary alert
requires choosing a threshold - and 0.5 is rarely the right choice for
imbalanced data. When only 4% of windows are positive, a calibrated model
should output P(incident) ≈ 0.04 on average; a 0.5 threshold would suppress
almost all alerts.

`src/evaluate.py` includes a `compare_thresholds()` function that prints a
table like this:

```
threshold  precision  recall    f1   roc_auc  n_alerts  n_positive
     0.10      0.231   0.991  0.374    0.973      1987         227
     0.20      0.412   0.960  0.575    0.973       543         227
     0.30      0.591   0.934  0.724    0.973       370         227
     0.50      0.798   0.876  0.835    0.973       249         227
     0.70      0.912   0.769  0.834    0.973       191         227
     0.90      0.970   0.557  0.707    0.973       127         227
```

**Note on choosing threshold in practice:**

- Lowering the threshold (e.g. 0.10 → 0.30) improves recall - more real
  incidents are caught - but precision drops sharply because many more
  non-incident windows now trigger alerts. In a busy operations centre this
  causes alert fatigue, where operators start ignoring alerts because too many
  are false alarms.

- Raising the threshold (e.g. 0.50 → 0.90) improves precision - fewer false
  alarms - but recall falls. Some real incidents are missed entirely. For a
  safety-critical system, a missed incident is worse than a false alarm.

The right choice depends on the cost asymmetry in the specific domain.

---

## Analysis of results

Running `python main.py` with default settings (RandomForest, W=30, H=10,
threshold=0.5) typically produces ROC-AUC ≈ 0.97 and F1 ≈ 0.83 on the test
set. This is high, but it should be: the synthetic data was designed with a
consistent, cleanly shaped stress ramp, which is exactly the kind of pattern
a random forest is good at.

A few observations worth making explicit:

**Recall improves as the threshold is lowered, but not for free.**
At threshold=0.10, recall approaches 0.99 - almost every real incident triggers
an alert. But n_alerts rises from ~250 to ~2000. Most of those extra alerts are
false positives. The model is casting a wide net, not becoming smarter.

**Precision drops because more non-incident windows get flagged.**
When the threshold falls, windows with moderate probability scores (0.10–0.50)
start triggering alerts. Most of those windows are not actually pre-incident -
they just have slightly elevated readings from background noise. This is the
precision/recall trade-off.

**The slope and mean features do most of the work.**
Feature importance scores from the RandomForest consistently rank
`sensor_a__slope`, `sensor_b__slope`, and `sensor_c__mean` in the top positions.
This matches what the data generator does: the stress ramp creates a rising
trend in sensors a, b, and c before each incident. The model found the right
signal.

**sensor_d contributes little.**
`sensor_d` was designed as a distractor - only weakly affected by stress -
and its feature importances confirm this. It appears near the bottom of the
ranking.

**LogisticRegression performs noticeably worse.**
The LR baseline typically scores ROC-AUC ≈ 0.90–0.92 versus RF's ≈ 0.97.
The gap suggests the pre-incident patterns are not linearly separable in the
24-feature space - likely because the relationship between slope and incident
probability is nonlinear (small slopes are safe, but there is a threshold above
which they become very predictive). The forest handles this; the linear model
does not.

---

## Limitations

N.b. would love to discuss potential solutions in an interview ;)

**Synthetic data may not match real operational noise.**
The stress ramp is perfectly consistent: same shape, same duration, same
amplitude for every incident. Real sensor data has sensor drift, calibration
offsets, missing readings, and incidents that give little or no prior warning.
A model trained on this data would likely overfit to the specific ramp shape
baked into the generator and underperform on anything messier.

**Incident patterns may drift over time.**
The generator produces stationary statistics - the same incident structure
appears throughout the 10,000 timesteps. In real systems, failure modes change
as equipment ages, operating conditions shift, or new sensor types are
introduced. A model trained on last year's patterns may not recognise this
year's incidents. This is concept drift, and the current pipeline has no
mechanism to detect or respond to it.

**Window-based features may miss long-range dependencies.**
The 24 features only describe the 30-step window immediately before the
prediction horizon. There is no feature that captures "sensor_a has been
elevated for the past 6 hours". In practice, slow baseline drift over a long
period is often more predictive than the last 30 readings. Extending the window
helps, but only up to a point - eventually you need a different architecture
(e.g. rolling aggregates at multiple timescales, or an RNN).

**Class imbalance can affect performance.**
At ~4% positive rate, `class_weight="balanced"` compensates during training,
but the predicted probabilities reflect this reweighting and are not calibrated
to the true base rate. The raw probability P(incident) = 0.4 does not mean a
40% chance of an incident - it means the model is relatively confident, but the
absolute value is unreliable. This makes threshold selection harder than it
looks. A calibration step (Platt scaling or isotonic regression) would make the
probabilities more interpretable.

**Binary labels lose resolution.**
Labelling the entire H-step horizon as 1 if *any* incident occurs conflates
"incident in 1 step" with "incident in 10 steps". The model has no incentive to
predict *when* within the horizon the incident will occur, and a window labelled
1 because an incident is 9 steps away is used for training the same way as one
where the incident is 1 step away.

**No temporal cross-validation.**
A single train/test split is used. Walk-forward validation across multiple
rolling folds would give a more honest picture of variance in performance and
how quickly the model degrades as the training data ages.

---

## Possible production extensions

- **Calibration** - apply `sklearn.calibration.CalibratedClassifierCV` so that
  P(incident) = 0.4 actually means 40% of similarly-scored windows become incidents.

- **Walk-forward validation** - evaluate on multiple rolling test windows to
  measure how quickly the model degrades and when it should be retrained.

- **Online feature store** - replace the batch windowing step with a streaming
  feature computation layer so the model can score every new sensor reading in
  real time.

- **Longer-range features** - add features that aggregate over the past hour,
  shift, or day (not just the current window) to capture slow baseline drift.

- **Model monitoring** - track the distribution of P(incident) over time.
  A sudden shift in the score distribution often signals sensor failure or
  concept drift before it is visible in labelled outcomes.

- **Anomaly pre-filter** - run a lightweight isolation forest or z-score check
  upstream to flag obviously anomalous readings before they reach the classifier,
  reducing noise and improving precision.

---

## Running the pipeline

```bash
# install dependencies
pip install numpy pandas scikit-learn

# run with defaults (RandomForest, W=30, H=10, 10 000 timesteps)
python main.py

# logistic regression baseline
python main.py --model lr

# tighter horizon, lower alert threshold
python main.py --horizon 5 --threshold 0.3

# all options
python main.py --help
```

**File layout**

```
.
├── main.py                      # end-to-end entry point
├── data/
│   └── synthetic_generator.py  # multivariate time series generator
└── src/
    ├── windowing.py             # sliding-window (X, y) builder
    ├── features.py              # per-window tabular feature extraction
    ├── train.py                 # time-aware split, model training
    └── evaluate.py              # metrics, thresholding, alert summary
```
