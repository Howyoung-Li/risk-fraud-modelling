from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from .utils.config import load_config
from .utils.io import ensure_dir
from .utils.split import time_split_masks
from .utils.metrics import auc_roc, auc_pr

def compute_scale_pos_weight(y: np.ndarray) -> float:
	pos = float((y==1).sum())
	neg = float((y==0).sum())
	if pos == 0:
		return 1.0
	return neg/pos

def main():
	cfg = load_config()

	processed_dir = cfg.processed_dir
	artifacts_dir = cfg.artifacts_dir
	ensure_dir(artifacts_dir/"models")
	ensure_dir(artifacts_dir/"metrics")
	ensure_dir(artifacts_dir/"plots")
	ensure_dir(artifacts_dir/"outputs")

	id_col = cfg.split.id_col
	time_col = cfg.split.time_col
	target_col = cfg.split.target_col

	fe_path = processed_dir / "model_table_fe.parquet"
	if not fe_path.exists():
		raise FileNotFoundError(f"Missing engineered table: {fe_path}")
	
	print(f"[train] loading: {fe_path}")
	df = pd.read_parquet(fe_path)

	# split
	train_mask, valid_mask, test_mask = time_split_masks(
		df, time_col=time_col, train_frac=cfg.split.train_frac,
		valid_frac=cfg.split.valid_frac
	)
	print(f"[train] rows = {len(df):,} train={train_mask.sum():,}, valid={valid_mask.sum():,} test={test_mask.sum():,}")

	# Features: drop identifiers and label/time
	drop_cols = {id_col, time_col, target_col}
	feature_cols = [c for c in df.columns if c not in drop_cols]
	print(f"[train] n_features:{len(feature_cols)}")

	X_train = df.loc[train_mask, feature_cols]
	y_train = df.loc[train_mask, target_col].astype(int).to_numpy()

	X_valid = df.loc[valid_mask, feature_cols]
	y_valid = df.loc[valid_mask, target_col].astype(int).to_numpy()

	# scale_pos_weight
	params = dict(cfg.lightgbm.params) if cfg.lightgbm.params is not None else {}
	if cfg.class_imbalance.use_scale_pos_weight:
		spw = cfg.class_imbalance.scale_pos_weight
		if spw is None:
			spw = compute_scale_pos_weight(y_train)
		params["scale_pos_weight"] = float(spw)
		print(f"[train] scale_pos_weight = {params['scale_pos_weight']:.4f}")
		
	# Sensible CPU defaults if not set
	params.setdefault("verbosity", -1)
	params.setdefault("n_jobs", -1)
	params.setdefault("metric","auc")
	
	# LightGBM datasets
	dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
	dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=True)
	
	evals_results = {}
	early_stopping = int(cfg.lightgbm.early_stopping_rounds) if cfg.lightgbm is not None else 200
	
	print("[train] training LightGBM (CPU)...")
	model = lgb.train(
		params=params,
		train_set=dtrain,
		num_boost_round=int(params.get("n_estimators",5000)),
		valid_sets = [dtrain,dvalid],
		valid_names=["train","valid"],
		callbacks=[
			lgb.early_stopping(stopping_rounds=early_stopping,verbose=True),
			lgb.log_evaluation(period=50),
			lgb.record_evaluation(evals_results),
		]
	)

	# predict on valid
	valid_pred = model.predict(X_valid, num_iteration=model.best_iteration)
	pr = auc_pr(y_valid, valid_pred)
	roc = auc_roc(y_valid, valid_pred)
	print(f"[train] valid PR-AUC={pr:6f} ROC-AUC={roc:6f} best_iter = {model.best_iteration}")

	# save model
	model_path = artifacts_dir/"models"/"lgbm_model.txt"
	model.save_model(str(model_path))
	print(f"[train] saved model: {model_path}")

	# save training curve plot
	# lightGBM returns 'auc' values if metrics is auc
	train_metrics = list(evals_results["train"].keys())[0] if evals_results else None
	if train_metrics:
		plt.figure()
		plt.plot(evals_results["train"][train_metrics], label="train")
		plt.plot(evals_results["valid"][train_metrics], label="valid")
		plt.xlabel("Iteration")
		plt.ylabel(train_metrics)
		plt.title("LightGBM training curve")
		plt.legend()
		plot_path = artifacts_dir/"plots"/"training_curve.png"
		plt.tight_layout()
		plt.savefig(plot_path,dpi=150)
		plt.close()
		print(f"[train] saved plot: {plot_path}")
	
	# save holdout predictions (valid and test for later use)
	amt_col ="TransactionAmt"
	out_valid = df.loc[valid_mask, [id_col,time_col,target_col, amt_col]].copy()
	out_valid["score"] = valid_pred.astype("float32")
	scored_path = artifacts_dir/"outputs"/"scored_valid.csv"
	out_valid.to_csv(scored_path,index=False)
	print(f"[train] saved scored_valid: {scored_path}")

	X_test = df.loc[test_mask, feature_cols]
	y_test = df.loc[test_mask, target_col].astype(int).to_numpy()
	test_pred = model.predict(X_test, num_iteration = model.best_iteration)

	out_test = df.loc[test_mask,[id_col, time_col, target_col, amt_col]].copy()
	out_test["score"] = test_pred.astype("float32")
	scored_test_path = artifacts_dir/"outputs"/"scored_test.csv"
	out_test.to_csv(scored_test_path, index=False)
	print(f"[train] saved scored_test: {scored_test_path}")


	# save quick metrics

	metrics = {
		"valid_pr_auc": pr,
		"valid_roc_auc": roc,
		"best_iteration": int(model.best_iteration),
		"n_features": len(feature_cols),
		"params": params,
	}
	metrics_path = artifacts_dir/"metrics"/"train_summary.json"
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics,f,indent=2)
		print(f"[train] saved train_summary: {metrics_path}")

if __name__ == "__main__":
	main()