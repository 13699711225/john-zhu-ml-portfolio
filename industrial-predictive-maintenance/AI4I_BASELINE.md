# AI4I 2020 — Public Synthetic Tabular Classification Benchmark

This is an **independent, reproducible portfolio benchmark**. The UCI authors explicitly describe AI4I as **synthetic**, designed to resemble industrial predictive maintenance data. It is **not plant telemetry** and **not a time-series early-warning evaluation**.

**Dataset:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (ID 601), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), CC BY 4.0. Dataset citation: *AI4I 2020 Predictive Maintenance Dataset* (2020), UCI Machine Learning Repository. Original paper: S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications* (2020).

## Why this is a separate experiment

The existing `demo.py` simulates machine telemetry and labels failures occurring in the **next six hours**, using chronological splits and rolling history.

AI4I provides records of equipment operating conditions and a **same-row** `Machine failure` label. Although UCI catalog metadata includes “Time-Series,” this script uses no validated per-asset event-time history or forward failure horizon. It is **tabular current-condition fault classification**, using a **seeded stratified random** 60/20/20 train/validation/test split. Do not compare its metrics directly to the six-hour early-warning task.

## Run with the official CSV

1. Download `ai4i2020.csv` from UCI. Save it locally as `industrial-predictive-maintenance/data/ai4i2020.csv`. **Do not commit the raw file**; the project ignores `data/`.
2. From the repository root:

```bash
cd industrial-predictive-maintenance
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ai4i_baseline.py --csv data/ai4i2020.csv
python -m unittest discover -s tests -v
```

Outputs are written to the ignored `artifacts/ai4i/` directory:

- `metrics.json`: held-out baseline versus validation cost-selected threshold, sample counts, caveats
- `test_predictions.csv`: per-row test predictions and selected-threshold alerts
- `model.joblib`: Random Forest model, exact feature list and selected threshold

**CI exercises only fabricated, tiny test fixtures**. The repository does **not** claim to have executed and published official full-dataset AI4I benchmark scores yet. Record exact versions and results after independently running the official CSV.

## Leakage and validation controls

- **Included model features:** air/process temperature, rotational speed, torque, tool wear, and one-hot encoded product variant (`L/M/H`).
- **Excluded:** `UDI`, `Product ID` (identifiers) and `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure-mode target labels). Using the latter as model inputs would trivially leak fault-label information.
- Input data must have complete numeric fields and valid binary targets; invalid records fail with an error rather than silently imputing values.
- Fixed random seed and explicit train/validation/test indices. Split is **row-level**, not a claim of independence across related product or operating conditions.
- Random Forest is trained only on training rows.
- Threshold is chosen using **validation labels only**, from 0.05 to 0.95.
- Test labels are evaluated exactly once under both the default 0.50 and validation-chosen thresholds. No threshold tuning based on test performance.
- `average_precision` and ROC-AUC describe ranking, not calibrated probability reliability. Calibration is not implemented.
- The CNY 300 false-positive and CNY 6,000 false-negative *row-level* costs are illustrative, **not** actual maintenance costs. Rows are not confirmed work orders.

## Next engineering step

After validating and documenting official benchmark numbers, implement a **read-only simulated MQTT → PostgreSQL → Python inference → Grafana** integration. Keep simulated and public benchmark data clearly labeled and never imply deployment to any industrial plant.
