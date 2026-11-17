# John Zhu — Machine Learning & Reliability Engineering Portfolio

Focus: **manufacturing reliability, fault detection, anomaly detection, and model validation**.

## Start here: business problem → ML evidence

| Project | Business question | Evidence and current status |
| --- | --- | --- |
| [Industrial Predictive Maintenance Demo](industrial-predictive-maintenance/) | How can sensor readings become actionable early warnings while balancing false alerts and missed faults? | **New synthetic-data prototype**: rolling features, chronological split, RF classifier, validation-only cost-sensitive threshold, held-out metrics and automated tests. Not production or real factory data. |
| [Cross-Domain Model Reliability Experiments](blockchain-model-reliability/) | Does an anomaly model remain reliable after data distribution changes? | GNN/DGI + RF research, negative sampling comparisons, calibration and cross-validation. **Legacy encoder/RF feature dimension mismatch remains unresolved**; historical results are not independently reproduced from current artifacts. |

## Engineering practices and focus areas

1. **Translate operations into measurable ML tasks:** failure-risk classification, lead-time definition, action thresholds and maintenance trade-offs.
2. **Build and evaluate robust baselines:** Python, pandas, scikit-learn, PyTorch, holdout validation, data leakage checks, error analysis.
3. **Protect decision quality:** data normalization, cross-system verification, model transfer checks, calibration and reporting.
4. **Communicate with engineers and stakeholders:** document assumptions, explain false alarms versus missed failures, and state limits transparently.

## Reproducibility

Start with the README within each project. Industrial demo results are based on synthetic data, and the independently documented AI4I benchmark is a separate current-row classification task. Neither should be presented as a deployed factory system or evidence of real equipment downtime reduction.
