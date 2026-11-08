"""
Threshold stability validation.

Goal:
    Compare whether each model can work with default threshold 0.5,
    and whether the validation-selected threshold is stable.

Key metrics:
    F1@0.5
    F1 selected threshold
    selected_threshold mean/std/median/min/max

Interpretation:
    best threshold near 0.5:
        probability mapping is more natural and directly usable.

    best threshold near 0.01 or 0.02:
        model still requires artificial threshold adjustment.

    high threshold std:
        threshold is unstable across folds.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="random, highconf, activitymatched")
    args = parser.parse_args()

    in_dir = Path(f"cv5_inner_threshold_{args.tag}_8m_9m")
    file = in_dir / "fold_level_metrics.csv"

    if not file.exists():
        raise FileNotFoundError(f"Cannot find {file}")

    df = pd.read_csv(file)

    rows = []

    for model, g in df.groupby("model"):
        rows.append({
            "setting": args.tag,
            "version": g["version"].iloc[0],
            "model": model,
            "f1_at_0.5_mean": g["f1_at_0.5"].mean(),
            "f1_at_0.5_std": g["f1_at_0.5"].std(),
            "f1_selected_threshold_mean": g["f1_selected_threshold"].mean(),
            "f1_selected_threshold_std": g["f1_selected_threshold"].std(),
            "selected_threshold_mean": g["selected_threshold"].mean(),
            "selected_threshold_std": g["selected_threshold"].std(),
            "selected_threshold_median": g["selected_threshold"].median(),
            "selected_threshold_min": g["selected_threshold"].min(),
            "selected_threshold_max": g["selected_threshold"].max(),
            "f1_best_test_diagnostic_mean": g["f1_best_test_diagnostic"].mean(),
        })

    out = pd.DataFrame(rows)

    order = {"V0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}
    out["order"] = out["version"].map(order)
    out = out.sort_values("order").drop(columns=["order"])

    out_file = in_dir / "threshold_stability_summary.csv"
    out.to_csv(out_file, index=False)

    print("=" * 80)
    print(f"Threshold stability summary: {args.tag}")
    print("=" * 80)
    print(out.to_string(index=False))

    print("\nSaved:")
    print(out_file)


if __name__ == "__main__":
    main()