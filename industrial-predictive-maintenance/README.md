# Industrial Predictive Maintenance — Synthetic Engineering Demo

**Purpose:** Show a manufacturing-oriented ML workflow: sensor data → past-only feature engineering → chronological train/validation/test split → failure risk model → maintenance-cost threshold → auditable predictions.

**Important distinction:** This is a newly created **synthetic-data demonstration**, **not a real Flex factory project**, not a production installation, and not the earlier AI4I-based experiment with reported F1 0.511→0.649 / simulated cost −27%. The synthetic simulator is designed to contain an informative deterioration signal. Its test results must **not** be represented as real-world evidence of model effectiveness.

## Business problem

An industrial team needs an early warning when a machine may fail within the next six hours. False alerts waste inspection effort, while missed events can lead to unplanned downtime. Here, *assumed illustration-only costs* are CNY 300 per false alert and CNY 6,000 per miss; these are **not measured customer or factory costs**.

## What is implemented

1. Reproducible seeded simulation of 12 machines × 240 hourly observations.
2. Rolling six-hour means and standard deviations for temperature, vibration, and current; computed per machine from **current/past readings**.
3. Chronological data split: hours 0–139 training, 140–189 validation, 190–239 test.
4. Random Forest failure classification; validation-only threshold selection for minimum assumed cost.
5. Held-out test evaluation: Precision, Recall, F1, average precision, confusion-matrix counts, and simulated cost for default and selected thresholds.
6. Exports `artifacts/model.joblib`, `artifacts/test_predictions.csv` and `artifacts/metrics.json`.

## How to run

From the repository root:

```bash
cd industrial-predictive-maintenance
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The script prints metrics and creates the `artifacts/` folder. **Verified in GitHub Actions on 2026-11-13:** the unit tests and end-to-end synthetic pipeline completed successfully on Python 3.11 (Linux). The values below are for synthetic data only. `artifacts/`, `.venv/`, and cache files are not committed.

## Design choices / trade-offs

- **Why RF?** Strong structured/tabular baseline; no sequence model is necessary to establish a first baseline.
- **Why chronological split?** A random split of overlapping sensor windows can create optimistic estimates.
- **Why tune a decision threshold?** The business cost of missed faults may exceed the cost of unnecessary inspections.
- **How is leakage controlled?** Window features use no future rows, and final test labels are never used to select the decision threshold. The simulator's event schedule generates a deliberate precursor signal: this is synthetic signal design, not proof of realistic forecasting.
- **What needs work before industrial use?** Verified equipment timestamps, actual failure/maintenance events, censoring and downtime handling, machine/batch-level validation, class-imbalance evaluation, lead-time requirements, calibration, monitoring, system integration, cybersecurity, and engineering review.

## Evidence map for interviews

| Business question | Evidence in this repo |
| --- | --- |
| Can you convert a reliability problem into an ML task? | `failure_next_6h` target and manufacturing-oriented cost assumptions |
| Can you prepare sensor features without reading the future? | `add_past_only_features()` and chronology tests |
| Can you avoid optimizing on the test set? | Validation-selected threshold and untouched test evaluation |
| Can you quantify the error trade-off? | Confusion-matrix counts and assumed-cost comparison |
| Can another engineer inspect your result? | Exported metrics, predictions, fixed random seed, readable code |

**Next integration step:** Replace synthetic data with the previously developed industrial fault classification prototype and its real source dataset, validate the claim F1 0.511→0.649 separately, then develop serving and monitoring layers. Do not combine the original prototype's metrics with these synthetic test metrics.

## Verified synthetic test results (GitHub Actions)

These are outputs from the automated run on 2026-11-13, **not** measurements from factory equipment.

| Held-out test metric | Default threshold 0.50 | Validation cost-selected threshold 0.22 |
| --- | ---: | ---: |
| Precision | 0.697 | 0.655 |
| Recall | 0.958 | 1.000 |
| F1 | 0.807 | 0.791 |
| False alarms | 30 | 38 |
| Missed warnings | 3 | 0 |
| Assumed total cost (CNY) | 27,000 | 11,400 |

**Interpretation:** Under the explicitly assumed false-alarm/miss costs (CNY 300/6,000), the validation-selected threshold reduced simulated held-out cost by approximately **57.8%**, at the expense of eight additional false alarms and a lower F1. This is **not** evidence of savings at any real factory. The data generator intentionally embeds a detectable signal; future validation with authentic sensor data is essential.

CI run: https://github.com/13699711225/john-zhu-ml-portfolio/actions/runs/43394532524
