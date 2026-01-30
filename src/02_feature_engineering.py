from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .utils.config import load_config
from .utils.io import ensure_dir, write_json
from .utils.reduce_mem import reduce_mem_usage

# --- Helper functions ---
def time_split_indices(
		df: pd.DataFrame,
		time_col: str,
		train_frac: float,
		valid_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Sort by time_col and split by row index proportions.
	Return Boolean masks for train, valid, test splits alighed with df index.
	"""
	order = np.argsort(df[time_col].to_numpy())
	n = len(df)
	n_train = int(n * train_frac)
	n_valid = int(n * valid_frac)

	train_idx = order[:n_train]
	valid_idx = order[n_train:n_train + n_valid]
	test_idx = order[n_train + n_valid:]

	train_mask = np.zeros(n, dtype=bool)
	valid_mask = np.zeros(n, dtype=bool)
	test_mask = np.zeros(n, dtype=bool)

	train_mask[train_idx] = True
	valid_mask[valid_idx] = True
	test_mask[test_idx] = True
	return train_mask, valid_mask, test_mask

def safe_str_series(s: pd.Series, fill: str = "missing") -> pd.Series:
	return s.astype("object").fillna(fill).astype("string")

def split_email_domain(domain: pd.Series, fill: str ="missing")-> Tuple[pd.Series, pd.Series]:
	"""
	domain like 'gmail.com' -> provider 'gmail', tld = 'com'
	if not '.', tld = 'missing'
	"""
	s = safe_str_series(domain, fill=fill)
	parts = s.str.split(".", n=1, expand=True)
	provider = parts[0].fillna(fill)
	if parts.shape[1] > 1:
		tld = parts[1].fillna(fill)
	else:
		tld = pd.Series([fill] * len(s), index=s.index)
	return provider, tld

def freq_encode_train_only(
	df: pd.DataFrame,
	col: str,
	train_mask: np.ndarray,
	fill_value = -1,
	prefix: str = "FE_",
) -> pd.Series:
	"""
	Frenquency encoding using TRAIN subset only (leakage safe).
	Return a float32 series aligned with df
	"""
	x =df[col]
	if x.dtype == "object":
		x_all = x.fillna("missing")
		x_tr = x_all[train_mask]
	else:
		x_all = x.fillna(fill_value)
		x_tr = x_all[train_mask]

	vc = x_tr.value_counts(dropna=False)
	denom = float(len(x_tr))
	# Convert to dict and ensure x_all is in compatible dtype for mapping
	mapping_dict = (vc/denom).to_dict()
	freq = x_all.astype("object").map(mapping_dict).fillna(0.0).astype("float32")
	freq.name = f"{prefix}{col}"
	return freq

def hash_uid(df: pd.DataFrame, cols: List[str]) -> pd.Series:
	"""
	Laptop-friendly UID: unit64 hash of selected columns.
	Avoisds creating huge UID strings.
	"""
	# Use pandas's hashing; stable within a run
	uid = pd.util.hash_pandas_object(df[cols],index = False).astype("uint64")
	uid.name = "uid_hush"
	return uid

def group_stats_train_only(
	df: pd.DataFrame,
	group_col: str,
	value_col: str,
	train_mask: np.ndarray,
	stats: List[str],
	prefix: str
) -> pd.DataFrame:
	"""
	Compute group stats on train only, then map to all rows.
	Returns a DataFrame with cols like '{prefix}_{stat}' aligned to df index.
	"""
	dtr = df.loc[train_mask,[group_col, value_col]].copy()
	# group_col could be unit64, ensure it is usable as a group key
	g = dtr.groupby(group_col, observed=False)[value_col].agg(stats)
	g.columns = [f"{prefix}_{s}" for s in g.columns]

	# map back
	out = df[[group_col]].join(g, on=group_col)
	return out.drop(columns=[group_col])

def zsafe_divide(num: pd.Series, den: pd.Series, eps: float = 1e-6) -> pd.Series:
	return (num/(den + eps)).astype("float32")

#----------------------------
# ------- Main Code ----------
#----------------------------
def main():
	cfg = load_config()

	id_col = cfg.split.id_col
	time_col = cfg.split.time_col
	target_col = cfg.split.target_col

	processed_dir = cfg.processed_dir
	raw_dir = cfg.raw_dir
	artifacts_dir = cfg.artifacts_dir

	ensure_dir(processed_dir)
	ensure_dir(artifacts_dir/"features")

	base_path = processed_dir/"model_table.parquet"
	if not base_path.exists():
		raise FileNotFoundError(f"Processed model table not found at {base_path}. Run 01_make_dataset.py first.")
	
	print(f"[fe] loading base table: {base_path}")
	df = pd.read_parquet(base_path)

	# time split masks
	train_mask, valid_mask, test_mask = time_split_indices(
		df, time_col,
		train_frac=cfg.split.train_frac,
		valid_frac=cfg.split.valid_frac,
	)

	print(f"[fe] rows = {len(df):,} train={train_mask.sum():,} valid={valid_mask.sum():,} test={test_mask.sum():,}")

	# --------------------- 
	# 1) bring back raw string cols needed for parsing (email domain)
	#    because 01_make_dataset.py label encoded original object cols
	# ---------------------
	needed_raw_cols = [id_col]
	email_cols = []
	for c in ["P_emaildomain", "R_emaildomain"]:
		if c in df.columns:
			email_cols.append(c)
			needed_raw_cols.append(c)

	if email_cols:
		train_tr_path =raw_dir/"train_transaction.csv"
		if not train_tr_path.exists():
			raise FileNotFoundError(f"Raw train transaction file not found at {train_tr_path}")
		
		print(f"[fe] re-reading raw email columns from {train_tr_path.name}: {email_cols}")
		raw_email = pd.read_csv(train_tr_path, usecols=needed_raw_cols)
		# Merge raw strings (suffix to avoid collisions)
		df = df.merge(raw_email, on=id_col, how = "left", suffixes=("", "_raw"))

		# Use *_raw if present, otherwise fall back
		for c in email_cols:
			raw_c = f"{c}_raw"
			src = raw_c if raw_c in df.columns else c
			provider, tld = split_email_domain(df[src],fill="missing")
			df[f"{c}_provider"] = provider
			df[f"{c}_tld"] = tld

		# drop raw merge columns to keep table clean
		drop_cols = [f"{c}_raw" for c in email_cols if f"{c}_raw" in df.columns]
		if drop_cols:
			df.drop(columns=drop_cols, inplace=True)

	# -----------------------   
	# 2) UID construction (hashed)
	#    common kaggle-style idea: use card1 + addr1 + (time_day - D1) buckets
	#    this script implements a robust, explainable hash-based approach
	# -----------------------
	if "time_day" not in df.columns:
		df["time_day"] = (df[time_col] // (24*3600)).astype("int32")

	
	# relative first day: time_day - D1 (D1 is like "days since first activity")
	# t - (t - t_0) = t_0
	if "D1" in df.columns:
		rel_day = (df["time_day"].astype("float32") - df["D1"].astype("float32")).astype("float32")
		# bucket to reduce noise and cardinality
		df["rel_day_bucket"] = np.floor(rel_day).replace([np.inf, -np.inf], np.nan).fillna(-999).astype("int16")
	else:
		df["rel_day_bucket"] = (-999).astype("int16")

	uid_cols = [c for c in ["card1", "addr1", "rel_day_bucket"] if c in df.columns]
	if len(uid_cols) >= 2:
		df["uid_hash"] = hash_uid(df, uid_cols) # hash for selected cols
	else:
		df["uid_hash"] = pd.Series(pd.util.hash_pandas_object(df[id_col], index=False).astype("uint64"), name="uid_hash")

	#-----------------------
	# 3) frenquency encoding (train only) for high ROI cols
	#----------------------
	# Include both raw_derived (provider, tld) and stable ID-like numeric columns
	freq_cols = []
	candiates = [
		"DeviceInfo", "DeviceType",
		"id_30", "id_31", "id_33",
		"P_emaildomain", "R_emaildomain",
		"P_emaildomain_provider", "P_emaildomain_tld",
		"R_emaildomain_provider", "R_emaildomain_tld",
		"card1", "card4", "card6",
		"addr1", "addr2",
		"ProductCD",
		"M1", "M2", "M3", "M4", "M5", "M6","M7","M8", "M9",
	]
	for c in candiates:
		if c in df.columns:
			freq_cols.append(c)
	
	# if provider/tld were created, they are object, if not, ignore.
	print(f"[fe] frequency encoding {len(freq_cols)} columns (train_only)")
	for c in freq_cols:
		df[f"FE_{c}"] = freq_encode_train_only(df,c, train_mask=train_mask, fill_value=-1, prefix="FE_")

	
	# -----------------------
	# 4) group stats (train only) + amount normalization
	# - by card1
	# - by card4
	# - by uid_hash
	# ----------------------
	if "TransactionAmt" in df.columns:
		df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"],errors="coerce").fillna(df["TransactionAmt"].median())

	if "TransactionAmt_log" not in df.columns and "TransactionAmt" in df.columns:
		df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype("float32"))
	
	fe_defs: List[Dict[str, str]] = []

	def add_group_block(group_col: str, label: str):
		if group_col not in df.columns:
			return
		
		# stats on raw and log amount
		for val_col in ["TransactionAmt", "TransactionAmt_log"]:
			if val_col not in df.columns:
				continue

			stats_df = group_stats_train_only(
				df, group_col=group_col, value_col=val_col, train_mask=train_mask,
				stats = ["count", "mean", "std"], prefix = f"{val_col}_{label}"
			)
			for col in stats_df.columns:
				df[col] = stats_df[col].astype("float32")

			# normalization ratios (use mean/std)
			mean_col = f"{val_col}_{label}_mean"
			std_col = f"{val_col}_{label}_std"

			if mean_col in df.columns:
				df[f"{val_col}_to_mean_{label}"] = zsafe_divide(
					df[val_col].astype("float32"), df[mean_col].astype("float32")
				)
				fe_defs.append({"feature": f"{val_col}_to_mean_{label}", "definition": f"{val_col} divided by TRAIN-only mean({val_col} groupbed by {label})"})
			if std_col in df.columns:
				df[f"{val_col}_to_std_{label}"] = zsafe_divide(
					df[val_col].astype("float32"), df[std_col].astype("float32")
				)
				fe_defs.append({"feature": f"{val_col}_to_std_{label}", "definition": f"{val_col} divided by TRAIN-only std({val_col} groupbed by {label})"})
		# count 
		cnt_col = f"TransactionAmt_{label}_cnt"
		if cnt_col not in df.columns:
			fe_defs.append({"feature": cnt_col, "definition": f"TRAIN-only transaction count grouped by {group_col}"})
		
	add_group_block("card1", "card1")
	add_group_block("card4", "card4")
	add_group_block("uid_hash", "uid")

		#--------------------------------
		# 5) Encode newly created object features (email provider/tld) for 
		# LGBM compatibility. To keep model table numeric, label encode if present.
		# ---------------------------------
	obj_cols = [c for c in df.columns if df[c].dtype=="object"]
	if obj_cols:
		print(f"label encoding {len(obj_cols)} new object columns (post feature)")
		for c in obj_cols:
			s = safe_str_series(df[c],fill="missing")
			# factorize -> int codes
			codes,_ = pd.factorize(s,sort=True)
			df[c] = codes.astype("int32")
			fe_defs.append({"feature": c, "definition": "Label-encoded categorical feature"})
	# explicitly definitions for uid-related primitives
	fe_defs.append({"feature": "uid_hash", "definition": f"hashed UID proxy from columns: {uid_cols}"})
	fe_defs.append({"feature": "rel_day_bucket", "definition": "Bucketed (time_day - D1)/7, missing = -999"})

	# Ensure no string columns remain (LightGBM requires numeric/bool)
	string_cols = [c for c in df.columns if str(df[c].dtype) in ("string", "object")]
	for c in string_cols:
		s = df[c].astype("object").fillna("missing").astype(str)
		df[c] = pd.factorize(s, sort=True)[0].astype("int32")

	# reduce memory after creation:
	if getattr(cfg, "reduce_memory", True):
		df = reduce_mem_usage(df,verbose=True)
		
	# save engineered table
	out_path = processed_dir/"model_table_fe.parquet"
	print(f"[fe] writing: {out_path}")
	df.to_parquet(out_path,index=False)

	# save feature summary
	fe_summary = pd.DataFrame(fe_defs).drop_duplicates()
	fe_summary_path = artifacts_dir/"features"/"fe_summary.csv"
	fe_summary.to_csv(fe_summary_path, index=False)

	print(f"[fe] wrote: {out_path}")
	print(f"[fe] wrote: {fe_summary_path}")
	print(f"[fe] final shape: {df.shape}")

if __name__ == "__main__":
	main()