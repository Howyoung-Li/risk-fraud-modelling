# IEEE-CIS Fraud Risk Modelling Project (End-to-End Risk Modelling Pipeline)

[中文说明 → README.md](README.md)

## Goal
Build a **reproducible, explainable, and production-ready** fraud risk modelling pipeline using the IEEE-CIS Fraud Detection dataset (Transaction + Identity tables), delivering business-ready outputs: metrics, policy simulation, monitoring pack, and explanation artifacts.

---

## 1. Key Results (Out-of-Time Test Set)
- **Main model: LightGBM (CPU)** achieves on OOT Test:
  - **PR-AUC = 0.5724**
  - **ROC-AUC = 0.8962**
  - **KS = 0.6333**
- Review capacity view (Test set, Top-K review):
  - **Top 1% review**: Precision ≈ **0.904**, Recall ≈ **0.260**
  - **Top 3% review**: Precision ≈ **0.610**, Recall ≈ **0.526**
  - **Top 10% review**: Precision ≈ **0.244**, Recall ≈ **0.700**
- Full closed loop delivered:
  - **Model evaluation → threshold/cost trade-off → monitoring (PSI/Score Drift) → explainability (SHAP/Reason Codes)**

---

## 2. Data Snapshot
- Sample size:
  - `train_transaction`: 590,540 transactions
  - `train_identity`: 144,233 rows
- Label distribution: fraud rate (`isFraud`) = **3.499%**
- Time range:
  - `TransactionDT` ∈ [86,400, 15,811,131] (seconds)
  - Constructed `time_day` as a daily time index for OOT splitting
- Data join:
  - `train_transaction` LEFT JOIN `train_identity` on `TransactionID`
- Feature dimension:
  - Wide table after feature engineering: ~**498** columns
  - Model training uses ~**494** features (excluding ID/time/label fields, etc.)
- Missingness:
  - Identity-related fields have heavy missing values
  - Use unified missing handling + tree-model robustness; apply stable encoding for categorical variables
- Official dataset description (Kaggle):
  - https://www.kaggle.com/competitions/ieee-fraud-detection/data

---

## 3. Data Privacy (De-identified)
The IEEE-CIS dataset is public competition data. Features are anonymized/de-identified and do not contain directly identifiable personal information. The modelling focuses on learning transaction risk patterns from anonymous features rather than identifying real individuals.

---

## 4. UID Construction & Entity-level Aggregations
To simulate real-world fraud systems that leverage historical behavior of the “same entity” (user/device/account), this project builds an anonymous entity identifier `uid_hash` (hash of concatenated stable fields) and generates time-aware aggregated features:

- `uid_cnt`: frequency of this UID (activity / repeated transaction intensity)
- `uid_amt_mean`: average transaction amount for this UID (typical spending level)
- `uid_amt_std`: std of transaction amount for this UID (behavior volatility / anomaly signal)

Implementation notes:
- Aggregations are computed **in chronological order** within the training pipeline to respect time and reduce leakage.
- **`uid_hash` is NOT used as a model input feature**. It is only used for groupby aggregation and then removed to avoid ID-memorization and generalization risk.

---

## 5. Reproducibility
- Config file: `configs/config.yaml`
- Kaggle raw data is not uploaded to GitHub. To reproduce, place it under `data/raw/`.

### One-click run (from repo root)
```bash
conda env create -f environment.yml
conda activate riskmodel

python -m src.00_check_data
python -m src.01_make_dataset
python -m src.02_feature_engineering
python -m src.03_train_models
python -m src.04_evaluate
python -m src.05_explain_shap
python -m src.06_policy_simulation
python -m src.07_monitoring_pack
```
---

## 6. Validation: Leakage-aware Time-based Split (OOT)

This project uses **time-based Out-of-Time (OOT)** splitting as the primary evaluation method:

- Sort by `time_day` in chronological order
- Split into **train / valid / test** by time

**Why this matters (fraud setting):**
- Fraud detection has strong **time drift** and **entity repetition** (same card/device/email domain, etc.)
- Random splits often introduce leakage and overestimate performance
- OOT is closer to real online deployment behavior

**Core evaluation metrics:**
- PR-AUC, KS, Recall@TopK, Precision@TopK, Calibration

**Business objective:**
- Maximize fraud capture under **limited review capacity** (Top-K review)

---

## 7. Model Performance (Base Models + Stacking)

### 7.1 Base Models: LightGBM vs XGBoost (Valid / Test)

| Model | Split | PR-AUC | ROC-AUC | KS | Brier |
|------|------|--------:|--------:|---:|------:|
| LightGBM | Valid | 0.6919 | 0.9340 | 0.7337 | 0.01830 |
| **LightGBM** | **Test** | **0.5724** | **0.8962** | **0.6333** | **0.02138** |
| XGBoost | Valid | 0.6273 | 0.8864 | 0.6378 | 0.01818 |
| XGBoost | Test | 0.5281 | 0.8476 | 0.5532 | 0.02161 |

**Business interpretation:**
- LightGBM outperforms XGBoost on Valid/Test, especially on **PR-AUC** and **KS**, matching this feature set and missingness pattern well.
- The drop from Valid to Test PR-AUC is expected under strict OOT evaluation (time drift + harder cases later in time).

**Outputs:**
- `artifacts/metrics/holdout_metrics.json`
- `artifacts/metrics/topk_table.csv`

### 7.2 Stacking (2nd-layer learner using base model scores)

| Stacking | Test PR-AUC | Test ROC-AUC | Test KS |
|---------|------------:|-------------:|--------:|
| Linear (logit on scores) | 0.5576 | 0.8625 | 0.5789 |
| GBM Meta (gbm on scores) | 0.5674 | 0.8883 | 0.6221 |

**Notes:**
- Stacking does not beat the single LightGBM (PR-AUC 0.5724).
- Typical reasons: base models are highly correlated; limited meta-training samples; no strict OOF stacking.
- Next step: K-fold OOF scores + regularization/calibration to improve generalization.

---

## 8. Review Capacity Metrics (Precision@TopK / Recall@TopK) — Test Set

| Review Top-K | Precision@K | Recall@K |
|-------------:|------------:|---------:|
| 0.1% | 1.0000 | 0.0289 |
| 0.5% | 0.9481 | 0.1362 |
| **1%** | **0.9041** | **0.2598** |
| **3%** | **0.6102** | **0.5261** |
| 5% | 0.4172 | 0.5994 |
| 10% | 0.2436 | 0.7000 |

**Business interpretation:**
- With only **1% daily review capacity**, you can capture about **26% of fraud**, and ~**90%** of reviewed cases are fraud (high precision).
- Expanding to **3% review capacity** captures about **53% of fraud**, with precision around **61%**.

---

## 9. Visualizations

PR Curve (LightGBM Test)  
![](artifacts/plots/pr_curve_lightgbm_test.png)

Top-K Capture Curve (LightGBM Test)  
![](artifacts/plots/topk_capture_lightgbm_test.png)

Lift Curve (LightGBM Test)  
![](artifacts/plots/lift_curve_lightgbm_test.png)

Calibration (LightGBM Test)  
![](artifacts/plots/calibration_lightgbm_test.png)

---

## 10. Explainability: SHAP + Reason Codes

- **Global explanation:** SHAP summary for key drivers
- **Local explanation:** evidence for high-risk cases
- **Reason Codes:** Top-N positive contribution features per sample for ops/policy usage

SHAP Summary  
![](artifacts/shap/shap_summary.png)

Sample reason codes:
- `artifacts/outputs/reason_codes_sample.csv`

---

## 11. Policy Simulation: Threshold / Cost / Review-rate Trade-off

Convert model scores into actionable policies by scanning thresholds / review rates and computing expected cost:

- **FP friction cost** (review/ops cost, user friction)
- **FN loss** (fraud loss, amount-based)

**Outputs:**
- `artifacts/policy/policy_tradeoff_table.csv`
- `artifacts/policy/threshold.json`

Cost vs Threshold  
![](artifacts/policy/cost_vs_threshold.png)

Cost vs Review Rate  
![](artifacts/policy/cost_vs_review.png)

Dollar Metrics  
![](artifacts/policy/dollar_metrics_vs_review_rate.png)

---

## 12. Monitoring Pack: PSI + Score Drift

Deployment monitoring assets (weekly/monthly checks):

- `artifacts/monitoring/psi_report.csv`
- `artifacts/plots/psi_top_features.png`
- `artifacts/monitoring/score_drift.csv`
- `artifacts/plots/score_drift.png`

PSI Top Features  
![](artifacts/plots/psi_top_features.png)

Score Drift (rolling window)  
![](artifacts/plots/score_drift.png)
