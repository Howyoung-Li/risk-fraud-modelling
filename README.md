# Risk Fraud Modelling (End-to-End)

End-to-end e-commerce fraud risk modelling pipeline (feature engineering, leakage-aware split, XGBoost baseline, evaluation, SHAP explainability).

## Quickstart

```bash
conda env create -f environment.yml
conda activate riskmodel

python -m src.00_check_data
python -m src.01_build_features
python -m src.03_train_gbdt
python -m src.04_evaluate
