# Industrial Graph Learning — Experimental Roadmap

> Research proposal for an independent industrial-AI portfolio, **not** a Flex implementation. The current runnable demo uses **Random Forest**, not GNN or GDN.

## Core hypothesis

For a production machine with several sensors, the relationships between temperature, vibration, current and process state may contain useful information beyond sensor-wise rolling statistics.

**Testable question:** Does learned sensor-relationship modeling improve anomaly detection or fault localization over a carefully tuned RF / gradient-boosted-tree / simple time-series baseline on an **independent industrial dataset**?

A GNN is justified only if it improves a pre-agreed operational metric without unacceptable implementation and maintenance overhead.

## Candidate references to study

- Deng & Hooi, *Graph Neural Network-Based Anomaly Detection in Multivariate Time Series* (AAAI 2021, GDN): https://ojs.aaai.org/index.php/AAAI/article/view/16523
- Kipf & Welling, *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017): https://arxiv.org/abs/1609.02907
- Veličković et al., *Graph Attention Networks* (ICLR 2018): https://arxiv.org/abs/1710.10903

Reading a paper is not evidence of a successful reproduction. Capture dataset/version, training configuration, feature and graph construction, seeds and actual outputs in a separate experiment report.

## Proposed graph representation

| Component | Candidate definition | Evidence required |
| --- | --- | --- |
| Node | Sensor channel, equipment component or process stage | Engineering taxonomy and stable IDs |
| Edge | Physical connection, process dependency or learned correlation | Clear semantics and leakage checks |
| Node features | Recent historical time-window values; missingness and operation-state indicators | Timestamp alignment and unit normalization |
| Target | Future failure event, forecast residual, or current anomaly | Operationally agreed label; never confuse anomaly with failure |
| Explanation | Sensor-level residual and attention/neighbor importance, validated with expert review | Attention **does not** establish causal root cause |

For GDN specifically, understand the distinction between learned graph structure, graph attention aggregation and prediction-based deviation scores. The learned graph is not necessarily an exact physical wiring map.

## Experimental gates

1. **Data contract:** Document sensor units, timestamp origin, maintenance events, device lifecycle and allowed uses. Use only authorized or public datasets.
2. **Baseline first:** Seasonal/rolling residual baseline, RF or gradient boosting, and a simple temporal network if justified. Fit all transforms using training data alone.
3. **Graph construction:** Physical adjacency versus data-learned Top-K neighbor graph. Compare with random and fully connected edge controls.
4. **Time-aware validation:** Validation threshold selection; unseen future test interval. If intended for multiple assets, add machine-holdout evaluation. Ensure rolling windows, normalization and target creation do not leak future events.
5. **Metrics:** PR-AUC, precision/recall, **event-level** detection recall, detection lead time, false-alert episodes per asset-day, inference latency and explainability review.
6. **Ablations:** Remove edges, scramble sensor relations, eliminate attention, compare graph construction choices, and compare to no-graph baseline.
7. **Robustness:** Missing sensors, faulty readings, drift in process settings, maintenance resets and different machine operating regimes.
8. **Decision:** Deploy only if benefits exceed added data, training, monitoring and governance costs.

## Transfer from the Bitcoin graph project

Bitcoin uses observed transaction-network edges and a GIN+DGI encoder. The sensor graph would need a **new node definition, edge rule, temporal forecasting objective, labels and validation protocol**. We can reuse sound research practices, but **not** assume that the Bitcoin weights or its classification metrics transfer.

## Definition of done for a credible future GDN portfolio addition

- A publicly reproducible dataset loader and precise train/validation/test split.
- Runnable baseline and graph model, pinned environment and CI smoke tests.
- Documented graph-learning and temporal leakage safeguards.
- Results generated from the repository code rather than copied from other experiments.
- Error analysis, failure cases, per-asset considerations and honest interpretation.
- Optional read-only risk dashboard; no claims of plant deployment.

**Not yet done:** Public industrial dataset integration, GDN implementation, measured GNN-versus-RF comparison, production integration.
