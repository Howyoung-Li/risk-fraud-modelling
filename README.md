# Risk Fraud Modelling (End-to-End)

End-to-end e-commerce fraud risk modelling pipeline (feature engineering, leakage-aware split, XGBoost baseline, evaluation, SHAP explainability).

## Quickstart

```bash
conda env create -f environment.yml
conda activate riskmodel

python -m src.00_check_data
python -m src.01_make_dataset
python -m src.02_feature_engineering
python -m src.03_train_models
python -m src.04_evaluate
```
## Data Dictionary

### Transaction Table

- **TransactionDT**: Time delta from a given reference datetime (i.e., *not* an actual timestamp).
- **TransactionAMT**: Transaction payment amount (USD).
- **ProductCD**: Product code; identifies the product associated with each transaction.
- **card1 – card6**: Payment card attributes (e.g., card type, card category, issuing bank, country, etc.).
- **addr**: Address-related fields.
- **dist**: Distance-related feature(s).
- **P_emaildomain / R_emaildomain**: Purchaser and recipient email domains.
- **C1 – C14**: Counting features (e.g., number of addresses associated with the payment card). Exact meanings are masked.
- **D1 – D15**: Time delta features (e.g., days since a previous transaction). Exact meanings are masked.
- **M1 – M9**: Match indicators (e.g., consistency between name/address/card information). Exact meanings are masked.
- **Vxxx**: Vesta-engineered rich features, including ranking, counting, and other entity relationship signals.

**Categorical Features (Transaction Table):**  
`ProductCD`, `card1 – card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `M1 – M9`

---

### Identity Table

Variables in this table capture identity and network/digital signature information associated with transactions. This includes:
- Network connection information (e.g., IP, ISP, proxy)
- Digital signature information (e.g., user agent, browser/OS/version)

These variables are collected by Vesta’s fraud protection system and digital security partners. Field names are masked, and a pairwise dictionary is not provided due to privacy protection and contractual constraints.

**Categorical Features (Identity Table):**  
`DeviceType`, `DeviceInfo`, `id_12 – id_38`
