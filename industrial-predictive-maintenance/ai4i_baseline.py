"""UCI AI4I 2020 current-row failure classification baseline.

The public AI4I dataset is SYNTHETIC, and does not validate real-world
early warning, lead time, maintenance savings, or Flex deployment.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "Machine failure"
NUMERIC = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
]
FEATURES = NUMERIC + ["type_L", "type_M", "type_H"]
# UDI and Product ID are identifiers; TWF, HDF, PWF, OSF, RNF are target labels.
EXCLUDED = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
FP_COST, FN_COST = 300, 6000  # Illustrative, NOT empirical maintenance costs.


def prepare_features(df):
    required = set(NUMERIC + ["Type", TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("Empty dataset")
    types = df["Type"].astype("string")
    if types.isna().any() or not types.isin(["L", "M", "H"]).all():
        raise ValueError("Type must be L, M or H with no missing values")
    numeric = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Numeric features must be finite and nonmissing")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Machine failure target must be 0 or 1")
    x = numeric.copy()
    for code in ["L", "M", "H"]:
        x[f"type_{code}"] = (types == code).astype(int)
    return x[FEATURES], y.astype(int)


def make_splits(y):
    """Stratified RANDOM rows: 60% train, 20% validation, 20% test.
    These are not chronological splits; AI4I labels describe current failure.
    """
    ix = np.arange(len(y))
    train, rest = train_test_split(ix, test_size=0.4, stratify=y, random_state=SEED)
    val, test = train_test_split(
        rest, test_size=0.5, stratify=y.iloc[rest], random_state=SEED
    )
    return train, val, test


def evaluate(y, probability, threshold):
    y = np.asarray(y)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(round(threshold, 4)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "average_precision": float(average_precision_score(y, probability)) if both_classes else None,
        "roc_auc": float(roc_auc_score(y, probability)) if both_classes else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "assumed_decision_cost_cny": int(FP_COST * fp + FN_COST * fn),
    }


def run(csv_path, output_dir="artifacts/ai4i", n_estimators=200):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    x, y = prepare_features(pd.read_csv(csv_path))
    train, val, test = make_splits(y)
    model = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced",
        min_samples_leaf=3, random_state=SEED, n_jobs=-1,
    )
    model.fit(x.iloc[train], y.iloc[train])
    val_p = model.predict_proba(x.iloc[val])[:, 1]
    # The threshold is selected ONLY on validation labels.
    thresholds = np.linspace(0.05, 0.95, 91)
    selected = min(
        thresholds,
        key=lambda t: (evaluate(y.iloc[val], val_p, float(t))["assumed_decision_cost_cny"], t),
    )
    test_p = model.predict_proba(x.iloc[test])[:, 1]
    result = {
        "dataset": "Public UCI AI4I 2020 SYNTHETIC tabular dataset",
        "task": "Current-row machine failure classification; NOT future failure prediction",
        "split": "60/20/20 stratified RANDOM rows; NOT time-based",
        "seed": SEED,
        "features": FEATURES,
        "excluded_from_training": EXCLUDED,
        "costs_are_illustrative_not_measured": {"fp_cny": FP_COST, "fn_cny": FN_COST},
        "counts": {
            "train": len(train), "validation": len(val), "test": len(test),
            "train_positive": int(y.iloc[train].sum()),
            "val_positive": int(y.iloc[val].sum()),
            "test_positive": int(y.iloc[test].sum()),
        },
        "test_default_0_5": evaluate(y.iloc[test], test_p, 0.5),
        "test_validation_selected": evaluate(y.iloc[test], test_p, float(selected)),
        "limitations": "No event-level lead time, true plant economics, or field deployment evidence",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({
        "source_row_index": test, "true_failure": y.iloc[test].to_numpy(),
        "predicted_probability": test_p,
        "selected_threshold_alert": (test_p >= selected).astype(int),
    }).to_csv(out / "test_predictions.csv", index=False)
    joblib.dump({"model": model, "features": FEATURES, "threshold": float(selected)}, out / "model.joblib")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Downloaded UCI ai4i2020.csv")
    parser.add_argument("--output-dir", default="artifacts/ai4i")
    args = parser.parse_args()
    run(args.csv, args.output_dir)
