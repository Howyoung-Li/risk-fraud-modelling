from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb
import shap

from .utils.config import load_config
from .utils.io import ensure_dir

def load_feature_table(processed_path: Path, id_col: str, time_col: str, target_col: str):
	df = pd.read_parquet(processed_path)
	drop_cols = [c for c in [id_col, time_col, target_col, "uid_hash"] if c in df.columns]
	feature_cols = [c for c in df.columns if c not in drop_cols]
	return df, feature_cols

def pick_local_cases(scored: pd.DataFrame, target_col: str, n: int):
	scored = scored.copy()
	scored = scored.sort_values("score", ascending=False)
	tp = scored[scored[target_col]==1].head(n)
	fp = scored[scored[target_col]==0].head(n)
	return tp, fp

def main():
	cfg = load_config()

	artifacts_dir = cfg.artifacts_dir
	model_dir =artifacts_dir/"models"
	out_dir = artifacts_dir/"outputs"
	shap_dir = artifacts_dir/"shap"
	ensure_dir(shap_dir)
	ensure_dir(out_dir)

	id_col = cfg.split.id_col
	time_col = cfg.split.time_col
	target_col = cfg.split.target_col

	processed_path = cfg.processed_dir/"model_table_fe.parquet"
	scored_valid_path = out_dir/"scored_valid.csv"

	if not processed_path.exists():
		raise FileNotFoundError(f"missing: {processed_path}")
	if not scored_valid_path.exists():
		raise FileNotFoundError(f"missing: {scored_valid_path}")
	
	# load data
	df, feature_cols = load_feature_table(processed_path, id_col, time_col, target_col)
	scored_valid = pd.read_csv(scored_valid_path)

	# join scored_valid -> features (by TransactionID)
	merged = scored_valid[[id_col, target_col, "score"]].merge(
		df[[id_col] + feature_cols],
		on=id_col,
		how="left",
		validate="one_to_one",
	)

	# sample for SHAP
	model_name = cfg.explain.model_name
	sample_n = cfg.explain.shap_sample_size
	local_n = cfg.explain.local_cases
	topn = cfg.explain.reason_codes_top_n
	seed = cfg.explain.random_seed

	# Build X for explain
	X = merged[feature_cols]
	# SHAP requires no object/string columns
	bad =[c for c in feature_cols if str(X[c].dtype) in ("object", "string")]
	if bad:
		raise ValueError(f"Non-numeric columns remain (encode first): {bad[:10]} (count ={len(bad)})")
	
	# Fill NaN for SHAP explanation consistency
	X_filled = X.copy()
	for c in feature_cols:
		if X_filled[c].isna().any():
			X_filled[c]=X_filled[c].fillna(X_filled[c].median())
	
	rng = np.random.default_rng(seed)
	if len(X_filled) > sample_n:
		idx = rng.choice(len(X_filled), size=sample_n, replace=False)
		X_s = X_filled.iloc[idx]
		y_s = merged[target_col].iloc[idx].to_numpy()
	else:
		X_s = X_filled
		y_s = merged[target_col].to_numpy()

	# load model + SHAP
	if model_name =="lightgbm":
		model_path = model_dir/"lgbm_model.txt"
		if not model_path.exists():
			raise FileNotFoundError(f"missing: {model_path}")
		booster = lgb.Booster(model_file=str(model_path))
		explainer = shap.TreeExplainer(booster)

		shap_values = explainer.shap_values(X_s)
		# LightGBM binary sometimes returns list; take postive class
		if isinstance(shap_values, list):
			shap_values = shap_values[1]
	elif model_name == "xgboost":
		import xgboost as xgb

		model_path = model_dir/"xgb_model.json"
		if not model_path.exists():
			raise FileNotFoundError(f"missing: {model_path}")
		booster = xgb.Booster()
		booster.load_model(str(model_path))
		explainer = shap.TreeExplainer(booster)
		shap_values = explainer.shap_values(X_s)

	else:
		raise ValueError(f"Unsupported model_name={model_name}, use 'lightgbm' or 'xgboost' ")

	# Global shap summary (bar of mean |shap|)
	mean_abs = np.mean(np.abs(shap_values), axis=0)
	imp = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
	imp.to_csv(shap_dir/"shap_importance.csv", index=False)

	topk = imp.head(30)
	plt.figure(figsize=(8,10))
	plt.barh(topk["feature"][::-1], topk["mean_abs_shap"][::-1])
	plt.title(f"SHAP Global Importance (Top 30) - {model_name}")
	plt.tight_layout()
	plt.savefig(shap_dir/"shap_summary.png",dpi=150)
	plt.close()

	# local explanations + reason codes
	# # pick cases from full valid scored set and map to features
	tp, fp = pick_local_cases(scored_valid[[id_col,target_col, "score"]], target_col, local_n)
	local_ids = pd.concat([tp,fp],axis=0)[id_col].tolist()

	local = scored_valid[[id_col, target_col, "score"]].merge(
		df[[id_col] + feature_cols],
		on=id_col,
		how="left",
		validate="one_to_one"
	)
	local_X = local[local[id_col].isin(local_ids)][feature_cols].copy()
	# fill NA for explanation rendering
	for c in feature_cols:
		if local_X[c].isna().any():
			local_X[c] = local_X[c].fillna(local_X[c].median())

	# compute SHAP for locals
	local_shap = explainer.shap_values(local_X)
	if isinstance(local_shap,list):
		local_shap = local_shap[1]

	local_rows = local[local[id_col].isin(local_ids)][[id_col,target_col,"score"]].reset_index(drop=True)
	reason_rows = []
	for i in range(len(local_rows)):
		sv = local_shap[i]
		# take top positive contributions as reasons
		pos_idx = np.argsort(-sv) # descending by contribution
		picked = []
		for j in pos_idx:
			if sv[j] <=0:
				continue
			picked.append((feature_cols[j], float(sv[j])))
			if len(picked) >= topn:
				break
		row = {
			id_col: int(local_rows.loc[i, id_col]),
			target_col: int(local_rows.loc[i, target_col]),
			"score": float(local_rows.loc[i, "score"]),
		} 
		for k, (fname, contrib) in enumerate(picked, start=1):
			row[f"reason_{k}"] = fname
			row[f"contrib_{k}"] = contrib
		reason_rows.append(row)

		# save a simple local bar plot
		feats = [p[0] for p in picked][::-1]
		vals = [p[1] for p in picked][::-1]
		plt.figure(figsize=(8,4))
		plt.barh(feats, vals)
		plt.title(f"Local Reason Codes (Top + SHAP) - ID={row[id_col]} y ={row[target_col]} score={row['score']:.4f}")
		plt.tight_layout()
		plt.savefig(shap_dir/f"local_cases_{row[id_col]}.png", dpi=150)
		plt.close()
	pd.DataFrame(reason_rows).to_csv(out_dir/"reason_codes_sample.csv", index=False)

	meta = {
		"model_name": model_name,
		"shap_sample_size": int(len(X_s)),
		"local_cases": int(len(local_ids)),
		"topn_reason_codes": topn,
		"n_features": int(len(feature_cols)),
	}
	with open(shap_dir/"shap_meta.json","w",encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	print(f"[shap] wrote: {shap_dir/'shap_summary.png'}")
	print(f"[shap] wrote: {out_dir/'reason_codes_sample.csv'}")

if __name__=="__main__":
	main()
