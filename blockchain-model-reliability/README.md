# Cross-Domain Model Reliability Experiments

Research code for investigating whether a frozen Bitcoin GNN + Random Forest model transfers reliably to Ethereum transaction data, and whether target-domain behavior covariates and probability calibration improve held-out evaluation.

## Included scripts

| File | Purpose |
| --- | --- |
| `experiments/run_m1_m4_model_comparison_8m_9m.py` | Main V0–M4 comparison, defaulting to activity-matched negative candidates. |
| `experiments/run_m1_m4_highconf_comparison_8m_9m.py` | Original parallel experiment script, defaulting to high-confidence negative candidates. Its code largely duplicates the main script and is retained for experimental provenance. |
| `evaluation/validate_5fold_inner_threshold_8m_9m.py` | Outer stratified five-fold cross-validation, inner validation threshold selection, held-out fold evaluation, and calibration metrics. |
| `evaluation/summarize_threshold_stability_8m_9m.py` | Summarizes threshold variation and F1 across outer folds. |

## Compared models

- **V0**: frozen model raw score, with no calibration.
- **M1**: Platt-style logistic calibration using only the raw score.
- **M2**: target-domain Ethereum behavioral features only (not Platt scaling).
- **M3**: raw score plus Ethereum behavioral features.
- **M4**: M3 inputs with sample weighting as a robustness check.

## Example workflow

Run from the `blockchain-model-reliability/` directory once all required input data and frozen artifacts are available.

```bash
python experiments/run_m1_m4_model_comparison_8m_9m.py --tag activitymatched
python evaluation/validate_5fold_inner_threshold_8m_9m.py \
  --probs-file m1_m4_activitymatched_probs_8m_9m.csv \
  --tag activitymatched
python evaluation/summarize_threshold_stability_8m_9m.py --tag activitymatched
```

For the high-confidence negative pool, use the second experiment script with `--tag highconf`, followed by validation with `m1_m4_highconf_probs_8m_9m.csv` and the same tag.

## Required inputs (not included)

- `exp_1a_small_block_window_8m_9m/small_window_transfers.csv`
- `exp_1a_small_block_window_8m_9m/positive_seed.csv`
- `exp_1a_small_block_window_8m_9m/activity_matched_negative_candidates.csv` or `negative_candidates.csv`
- `bitcoin_encoder.pt`, `bitcoin_rf.pkl`, `bitcoin_scaler.pkl`
- Python dependencies include `numpy`, `pandas`, `torch`, `torch-geometric`, `scikit-learn`, and `joblib`.

The repository intentionally does not include raw transaction datasets or serialized model artifacts. The four scripts have been syntax-checked, but **the end-to-end experiments have not been rerun or reproduced in this repository**.

## Evaluation notes and limitations

- The main held-out result is `f1_selected_threshold`, using a threshold chosen on the **inner validation** split and applied to an **outer test** fold.
- `f1_best_test_diagnostic` selects a threshold on test labels and is **diagnostic only**. It is not an unbiased deployable performance estimate.
- Positive labels are scarce and negative labels are derived from candidate sampling rules; they are not proof that the candidate addresses are legitimate.
- Outer-fold evaluation of a precomputed feature/probability CSV does not independently rule out leakage in upstream graph construction or feature generation.
- This is research code, **not production deployment**. Reported performance should not be interpreted as validated on a production system.
