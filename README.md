# Risk Fraud Modelling (IEEE-CIS) — 风控建模项目

## Overview
End-to-end fraud risk modelling pipeline on the IEEE-CIS Fraud Detection dataset (transaction + identity tables).
Focus: leakage-aware time split, industry metrics (PR-AUC / Recall@TopK), policy simulation, monitoring pack, and explainability (SHAP + reason codes).

## Key Results (Out-of-time holdout)
**Champion model:** LightGBM (CPU)

| Split | PR-AUC | ROC-AUC | KS |
|------|--------|---------|----|
| Valid | 0.xxx | 0.xxx | 0.xxx |
| Test  | 0.573 | 0.898 | 0.646 |

**Review-capacity performance (Test):**
- Top 0.5%: Precision ~0.94, Recall ~0.135  
- Top 1%: Precision ~0.90, Recall ~0.259  
- Top 3%: Precision ~0.61, Recall ~0.526  

(From `artifacts/metrics/topk_table.csv`)

## Plots
PR Curve (Test)  
![](artifacts/plots/pr_curve_lightgbm_test.png)

Top-K Capture (Test)  
![](artifacts/plots/topk_capture_lightgbm_test.png)

Lift Curve (Test)  
![](artifacts/plots/lift_curve_lightgbm_test.png)

SHAP Summary  
![](artifacts/shap/shap_summary.png)

## Policy Simulation (Risk Strategy)
Simulated review capacity and cost trade-offs:
- outputs: `artifacts/policy/policy_tradeoff_table.csv`, `threshold.json`
- plots: `artifacts/policy/cost_vs_threshold.png`, `cost_vs_review.png`

![](artifacts/policy/cost_vs_threshold.png)

## Monitoring Pack
Drift monitoring via PSI + score distribution drift:
- `artifacts/monitoring/psi_report.csv`
- `artifacts/plots/psi_top_features.png`
- `artifacts/plots/score_drift.png`

![](artifacts/plots/score_drift.png)

## Repo Structure
- `src/`: pipeline scripts
- `configs/`: config.yaml
- `artifacts/`: metrics, plots, policy, monitoring outputs (curated)

## Reproducibility (One-command steps)
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
