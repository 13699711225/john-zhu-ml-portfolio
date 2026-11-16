"""Synthetic manufacturing failure early-warning demo. Not real factory data."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

SEED = 42
HORIZON_HOURS = 6
FEATURES = ["temperature", "vibration", "current", "temp_mean_6", "temp_std_6",
            "vibration_mean_6", "vibration_std_6", "current_mean_6", "current_std_6"]
OUTPUT = Path("artifacts")


def make_synthetic_data(n_machines=12, n_hours=240):
    """Fixed-seed sensor simulator: deteriorating readings precede scheduled failures."""
    rng = np.random.default_rng(SEED)
    frames = []
    for m in range(n_machines):
        failure_hours = [72 + (m % 7) - 3, 150 + (m % 9) - 4, 220 + (m % 7) - 3]
        hours = np.arange(n_hours)
        # A simulated deterioration signal; no future observations enter features.
        proximity = np.zeros(n_hours)
        for failure in failure_hours:
            distance = failure - hours
            proximity += np.where((distance >= 0) & (distance <= 24),
                                  (24 - distance) / 24, 0.0)
        label = np.array([int(any(0 < f - h <= HORIZON_HOURS for f in failure_hours))
                          for h in hours])
        df = pd.DataFrame({
            "machine_id": f"M{m:02d}",
            "hour": hours,
            "temperature": 52 + 0.2 * m + 12 * proximity + rng.normal(0, 2, n_hours),
            "vibration": 1.5 + 1.7 * proximity + rng.normal(0, 0.25, n_hours),
            "current": 8 + 1.4 * proximity + rng.normal(0, 0.35, n_hours),
            "failure_next_6h": label,
        })
        frames.append(df)
    return add_past_only_features(pd.concat(frames, ignore_index=True))


def add_past_only_features(df):
    """Rolling windows only use current and earlier rows within each machine."""
    df = df.sort_values(["machine_id", "hour"]).copy()
    for column, prefix in [("temperature", "temp"), ("vibration", "vibration"), ("current", "current")]:
        rolling = df.groupby("machine_id")[column].rolling(window=6, min_periods=1)
        df[f"{prefix}_mean_6"] = rolling.mean().reset_index(level=0, drop=True).to_numpy()
        df[f"{prefix}_std_6"] = rolling.std(ddof=0).reset_index(level=0, drop=True).to_numpy()
    return df.reset_index(drop=True)


def evaluate(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # Illustrative, explicitly assumed cost per false alert/missed warning.
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "assumed_cost_cny": int(300 * fp + 6000 * fn),
    }


def run():
    df = make_synthetic_data()
    # Chronological split shared across machines. Prevents random-row time leakage.
    train = df[df.hour < 140]
    val = df[(df.hour >= 140) & (df.hour < 190)]
    test = df[df.hour >= 190]
    assert train.hour.max() < val.hour.min() < test.hour.min()
    assert min(train.failure_next_6h.sum(), val.failure_next_6h.sum(), test.failure_next_6h.sum()) > 0
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", min_samples_leaf=3,
                                   random_state=SEED, n_jobs=-1)
    model.fit(train[FEATURES], train.failure_next_6h)
    val_prob = model.predict_proba(val[FEATURES])[:, 1]
    # Selection uses validation labels only. Test labels never tune the threshold.
    candidates = np.linspace(0.05, 0.95, 91)
    threshold = min(candidates, key=lambda t: (evaluate(val.failure_next_6h, val_prob, t)["assumed_cost_cny"], t))
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    results = {
        "dataset": "synthetic equipment sensor simulation; no real factory observations",
        "target": "failure within next 6 hours",
        "split": "hour <140 train; 140-189 validation; >=190 test",
        "threshold_selection": "minimum assumed validation cost only",
        "assumed_false_positive_cost_cny": 300,
        "assumed_false_negative_cost_cny": 6000,
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_default_0_5": evaluate(test.failure_next_6h, test_prob, 0.5),
        "test_validation_selected": evaluate(test.failure_next_6h, test_prob, threshold),
    }
    OUTPUT.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": float(threshold)},
                OUTPUT / "model.joblib")
    test[["machine_id", "hour", "failure_next_6h"]].assign(
        predicted_risk=test_prob, alert=(test_prob >= threshold).astype(int)
    ).to_csv(OUTPUT / "test_predictions.csv", index=False)
    (OUTPUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
