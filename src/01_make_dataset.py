from pathlib import Path
import pandas as pd
import numpy as np

from .utils.config import load_config
from .utils.io import write_json, ensure_dir
from .utils.encoding import label_encode_union
from .utils.reduce_mem import reduce_mem_usage

def add_base_transforms(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])  # log(x+1)
    # 86400 seconds in a day
    df["time_day"] = (df[time_col] // 86400).astype("int32")
    return df

def infer_column_types(df: pd.DataFrame, id_col: str, time_col: str, target_col: str):
    feature_cols = [c for c in df.columns if c not in [id_col, time_col, target_col]]
    cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return feature_cols, cat_cols, num_cols

def main():
    cfg = load_config()

    ensure_dir(cfg.processed_dir)
    ensure_dir(cfg.artifacts_dir/"data")

    raw = cfg.raw_dir
    train_tr = pd.read_csv(raw/"train_transaction.csv")
    train_id = pd.read_csv(raw/"train_identity.csv")

    test_tr = test_id = None
    if cfg.use_test_for_encoding:
        tt = raw/"test_transaction.csv"
        ti = raw/"test_identity.csv"
        test_tr = pd.read_csv(tt)
        test_id = pd.read_csv(ti)

    if cfg.reduce_memory:
        train_tr = reduce_mem_usage(train_tr)
        train_id = reduce_mem_usage(train_id)
        if test_tr is not None:
            test_tr = reduce_mem_usage(test_tr)
            test_id = reduce_mem_usage(test_id)

    # Merge
    train =train_tr.merge(train_id, on=cfg.split.id_col, how="left")
    test = None 
    if test_tr is not None and test_id is not None:
        test = test_tr.merge(test_id, on=cfg.split.id_col, how="left")
    
    # Base transforms
    train = add_base_transforms(train, cfg.split.time_col)
    if test is not None:
        test = add_base_transforms(test, cfg.split.time_col)
    
    # Fill missing: numeric median, categorical "missing"
    feature_cols, cat_cols, num_cols = infer_column_types(
        train,cfg.split.id_col, cfg.split.time_col, cfg.split.target_col
    )

    # ensure categorical columns are obejct before encoding
    for c in cat_cols:
        train[c] = train[c].astype("object")
        if test is not None and c in test.columns:
            test[c] = test[c].astype("object")
    
    # fill numeric with median from TRAIN only
    medians = {}
    for c in num_cols:
        if c in [cfg.split.id_col, cfg.split.time_col, cfg.split.target_col]:
            continue
        med = float(pd.to_numeric(train[c], errors="coerce").median())
        medians[c] = med
        train[c] = pd.to_numeric(train[c],errors = "coerce").fillna(med)
        if test is not None and c in test.columns:
            test[c] = pd.to_numeric(test[c], errors="coerce").fillna(med)
    
    # fill categorical and label encode consistently
    enc_mappings = {}
    for c in cat_cols:
        tr_enc, te_enc, mapping = label_encode_union(
            train[c], test[c] if test is not None and c in test.columns else None,
            fill = cfg.categorical_fill
        )
        train[c] = tr_enc
        if test is not None and te_enc is not None:
            test[c] = te_enc
        enc_mappings[c] = {"n_categories": len(mapping)}

    # reduce memory again after fills and encodings
    if cfg.reduce_memory:
        train = reduce_mem_usage(train)
        if test is not None:
            test = reduce_mem_usage(test)
    
    out_path = cfg.processed_dir/"model_table.parquet"
    train.to_parquet(out_path, index=False)

    schema = {
        "n_rows": int(len(train)),
        "n_cols": int(train.shape[1]),
        "id_col": cfg.split.id_col,
        "time_col": cfg.split.time_col,
        "target_col": cfg.split.target_col,
        "categorical_encoded_columns": cat_cols,
        "numerical_columns": num_cols,
        "base_features_added": ["TransactionAmt_log", "time_day"],
        "encoding_summary": enc_mappings,
    }
    schema["base_features_added"] = [ x for x in schema["base_features_added"] if x in train.columns]

    write_json(schema, cfg.artifacts_dir/"data"/"schema.json")

    print(f"[make_dataset] Wrote processed data to {out_path}")
    print(f"[make_dataset] Wrote data schema to {cfg.artifacts_dir/'data'/'schema.json'}")
    print(f"[make_dataset] Dataset shape: {train.shape}")
    if test is not None:
        print(f"[make_dataset] Test shape: {test.shape}")


if __name__ == "__main__":
    main()
