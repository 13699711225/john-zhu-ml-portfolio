# Cross-Domain Model Reliability Experiments

An offline research project evaluating how reliably a **frozen Bitcoin GNN encoder + Random Forest classifier** transfers to Ethereum USDT/USDC transaction data. It compares direct transfer against score calibration and target-domain behavioral features, with a focus on held-out performance, threshold stability, and limitations.

**Status:** Research / offline evaluation; **not a production deployment**. The repository now includes selected transaction data and frozen model artifacts, but an independent end-to-end reproduction in a clean environment has **not yet been verified**.

## Research questions

1. Does a Bitcoin-trained anomaly detector retain useful ranking performance on Ethereum addresses?
2. Can Ethereum behavioral covariates help explain or improve cross-domain predictions?
3. Do improvements remain stable under held-out validation and validation-selected thresholds?
4. How do sampling choices affect apparent performance?

## Repository contents

| Path | Description |
| --- | --- |
| `bitcoin_encoder.pt` | Frozen Bitcoin GNN encoder weights. |
| `bitcoin_rf.pkl` | Serialized Bitcoin Random Forest classifier. |
| `bitcoin_scaler.pkl` | Serialized feature scaler. |
| `exp_1a_small_block_window_8m_9m/small_window_transfers.csv` | Selected Ethereum transaction window. |
| `exp_1a_small_block_window_8m_9m/positive_seed.csv` | Positive seed addresses (20 rows in the supplied file). |
| `exp_1a_small_block_window_8m_9m/activity_matched_negative_candidates.csv` | Activity-matched *candidate* negatives. |
| `exp_1a_small_block_window_8m_9m/random_negative_candidates.csv` | Randomly selected *candidate* negatives. |
| `experiments/run_m1_m4_model_comparison_8m_9m.py` | Main V0–M4 comparison; activity-matched negative pool by default. |
| `experiments/run_m1_m4_highconf_comparison_8m_9m.py` | Preserved parallel experiment for the high-confidence negative pool. |
| `evaluation/validate_5fold_inner_threshold_8m_9m.py` | Outer stratified 5-fold evaluation with inner validation threshold selection. |
| `evaluation/summarize_threshold_stability_8m_9m.py` | Summarizes fold-level threshold and F1 variation. |
| `results/fold_level_metrics.csv` | Saved fold-level metrics (5 models × 5 folds). Sampling provenance for this saved result still requires confirmation. |

The high-confidence input file `exp_1a_small_block_window_8m_9m/negative_candidates.csv` is **not included**. Do not assume the high-confidence experiment can be reproduced from the files currently in this repository.

## Model comparisons

| Model | Inputs | Interpretation |
| --- | --- | --- |
| **V0** | Frozen Bitcoin model score | Uncalibrated direct-transfer baseline |
| **M1** | Raw score only | Logistic / Platt-style calibration baseline |
| **M2** | Ethereum behavioral covariates only | Target-domain behavior baseline; **not** Platt scaling |
| **M3** | Raw score + behavioral covariates | Tests incremental value of the frozen score |
| **M4** | M3 inputs + sample weighting | Robustness check, including potential calibration trade-offs |

## Environment

The scripts use Python with `numpy`, `pandas`, `torch`, `torch-geometric`, `scikit-learn`, and `joblib` (plus their transitive dependencies). Exact versions used to train the serialized models have not been established in this repository, so compatibility must be tested before publishing a pinned, verified environment.

Example local setup on macOS with an available Python 3.11 installation:

```bash
cd blockchain-model-reliability
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn torch torch-geometric joblib
```

**Security:** `.pkl`/joblib files are Python serialized objects. Load only artifacts you trust; deserialization can execute code.

## Reproduce the offline experiments

All commands below are run **inside** `blockchain-model-reliability/`, because the scripts use paths relative to the current working directory.

### A. Activity-matched candidate negatives

```bash
python experiments/run_m1_m4_model_comparison_8m_9m.py \
  --neg-file exp_1a_small_block_window_8m_9m/activity_matched_negative_candidates.csv \
  --tag activitymatched
```

Expected experiment outputs:

- `m1_m4_activitymatched_probs_8m_9m.csv` (address-level scores and behavioral features)
- `m1_m4_activitymatched_comparison_8m_9m.csv`
- `m1_m4_activitymatched_comparison_8m_9m.json`

Run nested validation:

```bash
python evaluation/validate_5fold_inner_threshold_8m_9m.py \
  --probs-file m1_m4_activitymatched_probs_8m_9m.csv \
  --tag activitymatched

python evaluation/summarize_threshold_stability_8m_9m.py --tag activitymatched
```

Expected validation directory: `cv5_inner_threshold_activitymatched_8m_9m/`, containing `fold_level_metrics.csv`, `cv_summary_table.csv`, `pooled_test_predictions.csv`, `reliability_bins.csv`, `final_comparison_table.csv`, `validation_config.json`, and `threshold_stability_summary.csv`.

### B. Random candidate negatives

```bash
python experiments/run_m1_m4_model_comparison_8m_9m.py \
  --neg-file exp_1a_small_block_window_8m_9m/random_negative_candidates.csv \
  --tag random

python evaluation/validate_5fold_inner_threshold_8m_9m.py \
  --probs-file m1_m4_random_probs_8m_9m.csv \
  --tag random

python evaluation/summarize_threshold_stability_8m_9m.py --tag random
```

### C. High-confidence candidate negatives — unavailable input

The parallel experiment requires `exp_1a_small_block_window_8m_9m/negative_candidates.csv`, which has **not** been uploaded. Only run this experiment after verifying the provenance of that input:

```bash
python experiments/run_m1_m4_highconf_comparison_8m_9m.py \
  --neg-file exp_1a_small_block_window_8m_9m/negative_candidates.csv \
  --tag highconf
```

The commands describe the scripts' intended workflow; **they have not been verified by a fresh end-to-end run in this repository**.

## Evaluation and interpretation

- **Primary held-out metric:** `f1_selected_threshold`, with the threshold selected on **inner validation** and applied to the outer held-out fold.
- Also review `precision_selected_threshold`, `recall_selected_threshold`, `auc`, `brier`, and fold-to-fold variation in `selected_threshold`.
- `f1_at_0.5` is a fixed-threshold reference point.
- `f1_best_test_diagnostic` optimizes a threshold using **test labels**: this is **diagnostic only** and must not be reported as an unbiased deployment-ready result.
- Compare **M2 vs. M3** to assess whether the frozen Bitcoin score adds information beyond Ethereum behavioral features.
- Verify whether apparent F1 gains are accompanied by acceptable probability calibration and threshold stability.

### Saved results

`results/fold_level_metrics.csv` contains 25 model-fold rows for V0–M4. It is an existing research artifact, **not a fresh reproduction performed by this repository**. The specific negative-sampling setting for this saved file has not been independently verified; avoid calling it the activity-matched, random, or high-confidence result until its run configuration or provenance is recovered.

## Limitations

1. **Scarce positives:** approximately four positives per test fold in the saved evaluation; performance estimates can vary substantially.
2. **Candidate negatives:** sampling rules do not independently establish that sampled addresses are benign.
3. **Sampling sensitivity:** comparisons must be reported separately for the different negative pools.
4. **Possible upstream leakage:** cross-validation over precomputed features/scores does not independently verify leakage-free graph construction, pretraining, or feature preparation.
5. **Cross-domain feature assumptions:** mapping Bitcoin-trained feature representations to Ethereum data is experimental and may not preserve feature semantics.
6. **Deployment gap:** there is no production inference service, real-time monitoring, or validated industrial operating environment in this repository.

## Next improvements

- Run the complete workflow in a clean environment; record tested dependency versions, commands, runtime, and reproducible outputs.
- Recover or regenerate missing experiment provenance and high-confidence input, if permitted.
- Add model loading/data-schema smoke tests and automated checks.
- Add clearly labeled summary tables or plots generated from validated outputs.
