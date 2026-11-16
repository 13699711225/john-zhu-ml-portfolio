# Industrial AI Reliability — Flex Interview Case Study

> **Independent portfolio exercise.** This is **not** a Flex project, a collaboration with Flex, a Flex system design, or an analysis of proprietary Flex factory data. "Flex" is the prospective employer motivating the interview exercise. The dataset and test numbers below come from a **deliberately synthetic simulator**.

## One-minute executive overview (English)

**Business question:** Can an engineering team use equipment sensor readings to prioritize inspections before a possible failure, while balancing missed failures and nuisance alerts?

**What I built:** A Python/Random Forest engineering demonstration with simulated temperature, vibration, and current readings; past-only rolling features; chronological train/validation/test partitions; validation-only decision-threshold selection; and exports for audit and review.

**What the demo showed (synthetic held-out test):**

| Decision rule | Precision | Recall | F1 | False alerts | Missed warnings | Assumed total cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF score >= 0.50 | 0.697 | 0.958 | 0.807 | 30 | 3 | CNY 27,000 |
| RF score >= 0.22, selected on validation | 0.655 | 1.000 | 0.791 | 38 | 0 | CNY 11,400 |

The cost assumptions are **CNY 300 per false alert and CNY 6,000 per missed warning**, not measured factory economics. Under these assumptions the second rule reduces **simulated** held-out decision cost by **57.8%** while increasing false alerts by eight. The model's synthetic data generator intentionally embeds pre-failure signals, so these metrics **cannot predict real-world performance**.

The **business lesson** is not "AI saves 57.8% at Flex." It is that **the decision threshold is an engineering and business choice**, not merely a machine-learning hyperparameter.

## The five-slide interview narrative

### Slide 1 — Problem: reliability is an operational decision
- Unplanned downtime can disrupt delivery, quality, and maintenance scheduling.
- Every alert has an investigation cost; every missed failure can have a higher operational or safety cost.
- Start by defining: *What is an actionable event? How far in advance must we alert? Who acts?*

**What to say:** "I would not optimize the F1-score before agreeing with equipment engineers on failure definitions, lead times, and what action follows an alert."

### Slide 2 — Workflow: from sensors to risk scores
```text
Equipment sensors / authorized data feed
      ↓
Timestamp and data-quality checks
      ↓
Machine-specific past-only rolling features
      ↓
Train → Validation → Test (chronological)
      ↓
RF baseline and calibrated/validated alert rule
      ↓
Human-reviewed alert and dashboard
      ↓
Maintenance feedback and model monitoring
```

**Actually implemented:** Synthetic sensor generator, historical features, RF, chronological splitting, threshold selection, test reporting, CSV/JSON/model exports. **Not implemented:** MQTT ingestion, PostgreSQL, Grafana, production serving, model calibration, equipment control, human workflow.

### Slide 3 — Results: evaluate trade-offs, not just model score
Explain how a validation-chosen threshold can lower an *assumed* cost function while reducing F1. Show the confusion matrix and discuss lead time and false alerts per machine per day. Avoid presenting sample-level classification counts as actual maintenance work orders.

**What to say:** "My prototype tests whether the alert decision changes when false negatives are more expensive than false positives. I would need real incident and maintenance logs to estimate those costs credibly."

### Slide 4 — Technical strength: dependable experimentation
- Features use present and historical sensor readings only.
- Training, validation, and testing are separated chronologically.
- Threshold selection uses validation labels, not test labels.
- Fixed seed, automated unit tests, transparent experiment artifacts.
- Known limitations: repeated overlapping windows, simulated precursor signals, no production evaluation, no calibration or deployment.

**Transferable prior work (separate project):** Rebuilt Bitcoin GIN+DGI+RF model pipeline using training-only scaling, temporal graph splits, model saving and reproducibility checks. A local 300-epoch experiment reported AF+DNE F1 **0.805** and ROC-AUC **0.926**, but the exact training recovery files are **not yet in this repository**. Do not claim the Bitcoin results are currently reproduced by this industrial demo or deployed for manufacturing.

### Slide 5 — What I would deliver during the first 90 days
| Period | Objective | Evidence of success |
| --- | --- | --- |
| Days 1–30 | Understand one equipment family, failure modes, data ownership, data quality, and alert actions | Data contract, event definitions, leakage checklist, agreed KPI definitions |
| Days 31–60 | Develop transparent baselines using authorized historical data | Reproducible benchmark, time-aware split, per-machine error analysis, approved demo |
| Days 61–90 | Pilot a reviewed alert workflow if data and safety approvals permit | Shadow-mode dashboard, monitored alert log, maintenance-team feedback and go/no-go recommendation |

These are **proposed milestones**, conditional on assignment, data access, and internal governance, not promised factory outcomes.

## Operational KPIs to agree with the engineering team

| KPI | Working definition | Why it matters |
| --- | --- | --- |
| Alert precision | Confirmed relevant alerts / investigated alerts | Prevent inspection fatigue |
| Event recall | Failure events with at least one useful preceding alert / observable failure events | Avoid missed failures |
| Early warning lead time | Failure timestamp minus first useful alert | Make preventive action possible |
| False alerts / asset-day | Non-actionable alert episodes / observed asset-days | Align offline scoring with daily workload |
| Downtime hours | Actual unplanned machine unavailability | Operational value, requires controlled measurement |
| Intervention cost | Inspection + maintenance + avoided loss, with documented assumptions | Economic evaluation |
| Data validity | Missingness, stale timestamps, out-of-range and disconnected sensors | Trust in the system |

**Important methodological distinction:** Current repository metrics are **row/window-level**, not incident-level event metrics. No lead-time, downtime-reduction, or actual work-order cost claim is supported by current code.

## Architecture proposal (future work only)

```text
Sensor / PLC / approved historian
          ↓  (read-only, IT/OT approved)
MQTT or existing plant connector
          ↓
Timestamp / unit / ID validation
          ↓
PostgreSQL or existing time-series store
          ↓
Python feature + model service (RF baseline first)
          ↓
Reviewable risk API / alerts
          ↓
Grafana / existing operational dashboard
          ↓
Human maintenance confirmation → feedback log
```

Operational safeguards: read-only acquisition, approved network boundary, no autonomous machine actuation, secrets management, audit logs, backfill handling, measured latency, rollback and fail-safe behavior.

## English Q&A for an AI/Automation Engineering interviewer

**Q: Why choose Random Forest rather than GNN?**  
*A:* "Random Forest is a strong, interpretable-enough tabular baseline. I would only try GNNs when sensor or equipment relationships provide measurable gains in detection, localization, or robustness. A more complex model must justify its maintenance cost."

**Q: Why did the cost-focused threshold produce a lower F1?**  
*A:* "Because F1 weighs precision and recall symmetrically through their harmonic mean, while the assumed cost function assigns different costs to missed warnings and false alerts. The cost-optimal operating point does not necessarily maximize F1."

**Q: What is your most important limitation?**  
*A:* "The evidence comes from a synthetic dataset whose generator includes detectable pre-failure signals. I cannot claim manufacturing value until it has been tested against authentic maintenance events, real sensor noise and a realistic alert-response process."

**Q: How would you prove the system reduces downtime?**  
*A:* "I would define incident-level KPIs with maintenance engineers, establish a comparable baseline, run an approved shadow or controlled pilot, and attribute changes carefully while accounting for load, equipment, staffing and process differences."

**Q: What if a model works on one machine but fails on another?**  
*A:* "I would examine sensor scaling, machine operating regimes, missing data, domain shift and fault labels; then compare global, calibrated and machine-specific approaches using held-out machines and time periods."

**Q: How does your Bitcoin GNN research transfer?**  
*A:* "I transfer the discipline: relationship-aware representation learning, temporal validation, leakage prevention, reproducibility and anomaly error analysis. I would rebuild the graph for sensor and equipment relationships rather than reuse the Bitcoin encoder."

**Q: What would you ask the team?**  
*A:* "Which equipment reliability problems are currently highest priority? What constitutes an actionable early warning, and how is success measured today? Does the role focus more on modeling, integration or long-term monitoring?"

## Interview integrity checklist

- Distinguish **implemented and tested**, **independent prior research**, **future proposal**, and **unverified business impact**.
- Never say the synthetic results are Flex plant measurements.
- Never promise an exact percentage of downtime reduction based on the demo.
- Avoid sharing employer/customer proprietary information or using unapproved production data.
- Keep the focus on operational decisions, trustworthy data and repeatable evidence.

Run the code and tests using the commands in [README.md](README.md).
