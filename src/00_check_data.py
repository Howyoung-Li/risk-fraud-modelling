"""
This script validates file presence + prints core stats +
writes data_profile.json and missingness_top.csv.
"""

from pathlib import Path
import pandas as pd
import json

from .utils.config import load_config
from .utils.io import write_json, ensure_dir

REQUIRED = [
    "train_transaction.csv",
    "train_identity.csv",
]

OPTIONAL = [
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]

def main():
    cfg = load_config()
    raw = cfg.raw_dir

    print(f"[check] raw_dir = {raw.resolve()}")
    missing = [f for f in REQUIRED if not (raw / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {raw}: {missing}")
    
    present_optional = [f for f in OPTIONAL if (raw / f).exists()]
    print(f"[check] Found optional files: {present_optional}")

    # Read only needed files for stats
    tt_path = raw / "train_transaction.csv"
    ti_path = raw / "train_identity.csv"

    tt = pd.read_csv(tt_path, usecols=[cfg.split.id_col, cfg.split.time_col, cfg.split.target_col])
    ti = pd.read_csv(ti_path, usecols=[cfg.split.id_col])

    n_tt = len(tt)
    n_ti = len(ti)
    fraud_rate = tt[cfg.split.target_col].mean()
    dt_min = tt[cfg.split.time_col].min()
    dt_max = tt[cfg.split.time_col].max()

    # missingness snapshot: read a small sample of full columns to estimate missingness
    tt_full =  pd.read_csv(tt_path, nrows=200000)
    miss = (tt_full.isnull().mean().sort_values(ascending=False) * 100).round(2).head(50)
    miss_df = miss.reset_index()
    miss_df.columns = ["feature", "missing_pct"]

    artifacts_dir = cfg.artifacts_dir/"data"
    ensure_dir(artifacts_dir)

    profile = {
        "train_transaction_rows": n_tt,
        "train_identity_rows": n_ti,
        "fraud_rate": fraud_rate,
        "transactionDT_min": int(dt_min),
        "transactionDT_max": int(dt_max),
        "optional_files_found": present_optional,
    }

    write_json(profile, artifacts_dir/"data_profile.json")
    miss_df.to_csv(artifacts_dir/"missingness_top.csv", index=False)

    print(f"[check] Data profile and missingness saved to {artifacts_dir}")
    print(f"[check] train_transaction rows: {n_tt}")
    print(f"[check] train_identity rows: {n_ti}")
    print(f"[check] Fraud rate: {fraud_rate:.4f}")
    print(f"[check] transactionDT range: {dt_min} to {dt_max}")

if __name__ == "__main__":
    main()
    
