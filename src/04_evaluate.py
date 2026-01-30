"""
Currently only model lightgbm was implemented before evaluation
Any base model outputs for other models that generate:

artifacts/outputs/scored_valid_{model}.csv
artifacts/outputs/scored_test_{model}.csv

will be included in this script automatically.
Also, it will try stacking if >= 2 base models are available.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from .utils.config import load_config
from .utils.io import ensure_dir
from .utils.metrics import auc_pr, auc_roc, ks_stat, topk_table, calibration_bins, brier
from .utils.plots import save_pr_curve, save_lift_curve, save_calibration_plot, save_topk_capture

def load_scored(path: Path, target_col: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
	df = pd.read_csv(path)
	y = df[target_col].astype(int).to_numpy()
	s = df["score"].astype(float).to_numpy()
	return y, s, df

def discover_more_files(outputs_dir: Path, model_name: str)->Dict[str,Path]:
	"""
	Returns dict with keys: 'valid', 'test' if files exist.
	Name rules:
	  - default lightGBM: scored_valid.csv /scored_test.csv
	  - other models: scored_valid_{model}.csv / scored_test_{model}.csv
	"""
	files = {}
	if model_name == "lightgbm":
		v = outputs_dir/"scored_valid.csv"
		t = outputs_dir/"scored_test.csv"
	else:
		v = outputs_dir/f"scored_valid_{model_name}.csv"
		t = outputs_dir/f"scored_test_{model_name}.csv"
	
	if v.exists():
		files["valid"] = v
	if t.exists():
		files["test"] = t
	return files

def evaluate_one(y_true, y_score, topk_percents, make_calibration=True, cal_bins=20):
	out = {
		"pr_auc": auc_pr(y_true, y_score),
		"roc_auc": auc_roc(y_true, y_score),
		"ks": ks_stat(y_true, y_score),
		"brier": brier(y_true, y_score),
	}
	out["topk"] = topk_table(y_true, y_score, topk_percents)
	if make_calibration:
		centers,emp, pred,cnt = calibration_bins(y_true, y_score, n_bins=cal_bins)
		out["calibration"] = {
			"bin_centers": centers.tolist(),
			"empirical_rate": np.nan_to_num(emp,nan=np.nan).tolist(),
			"predicted_rate": np.nan_to_num(pred, nan=np.nan).tolist(),
			"bin_count": cnt.tolist(),
		}
	return out

def stack_linear(train_scores: np.ndarray, y: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
	"""
	Meta-model 1: logistic regression on base model scores.
	"""
	lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
	lr.fit(train_scores, y)
	return lr.predict_proba(test_scores)[:,1]

def stack_gbm(train_scores: np.ndarray, y: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
	"""
	Meta-model 2: LightGBM on base model scores (very small model).
	"""
	dtrain = lgb.Dataset(train_scores, label=y)
	params = {
		"objective": "binary",
		"learning_rate": 0.05,
		"num_leaves": 16,
		"min_data_in_leaf": 50,
		"feature_fraction": 1.0,
		"bagging_fraction": 1.0,
		"verbosity": -1,
		"metric": "auc",
	}
	model = lgb.train(params,dtrain, num_boost_round=500)
	return model.predict(test_scores)

def main():
	cfg = load_config()
	artifacts_dir = cfg.artifacts_dir
	outputs_dir =cfg.artifacts_dir/"outputs"
	metrics_dir = artifacts_dir/"metrics"
	plots_dir = artifacts_dir/"plots"

	ensure_dir(metrics_dir)
	ensure_dir(plots_dir)

	target_col = cfg.split.target_col
	topk_percents = cfg.evaluation.topk_percents
	make_cal = cfg.evaluation.make_calibration
	cal_bins = cfg.evaluation.calibration_bins
	amount_col = cfg.evaluation.amount_col

	# ---------------------
	# 1) Evaluate base models from outputs directory
	# ----------------------
	if hasattr(cfg,"model_run"):
		model_names = cfg.model_run
	else:
		# fallback
		model_names = ["lightgbm"]

	# Map run list names to file naming
	# logit, xgboost, lightgbm, torch_mlp
	normalized = []
	for m in model_names:
		normalized.append(m)
	
	results = {"base_models": {}, "stacking": {}}
	available_for_stacking = {} # list of (model_name, valid_scores, test_scores)
	valid_scores_matrix = []
	test_scores_matrix = []
	base_order = []

	for m in normalized:
		files = discover_more_files(outputs_dir, m)
		if "valid" not in files:
			continue
	
		yv, sv, _ = load_scored(files["valid"], target_col)
		res_v = evaluate_one(yv, sv, topk_percents, make_calibration=make_cal,cal_bins=cal_bins)

		results["base_models"][m] = {"valid": res_v}

		# plots for valid
		save_pr_curve(yv, sv, plots_dir/f"pr_curve_{m}_valid.png")
		save_lift_curve(yv, sv, plots_dir/f"lift_curve_{m}_valid.png")
		save_topk_capture(yv,sv,topk_percents, plots_dir/f"topk_capture_{m}_valid.png")

		if make_cal and "calibration" in res_v:
			c = res_v["calibration"]
			save_calibration_plot(
				np.array(c["bin_centers"]),
				np.array(c["empirical_rate"],dtype=float),
				np.array(c["predicted_rate"],dtype=float),
				plots_dir/f"calibration_{m}_valid.png",
			)
		
		# test evaluation if present
		if "test" in files:
			yt, st, _ = load_scored(files["test"],target_col)
			res_t = evaluate_one(yt, st, topk_percents, make_calibration=make_cal, cal_bins=cal_bins)
			results["base_models"][m]["test"] = res_t
			
		# plots for test
		save_pr_curve(yt, st, plots_dir/f"pr_curve_{m}_test.png")
		save_lift_curve(yt, st, plots_dir/f"lift_curve_{m}_test.png")
		save_topk_capture(yt,st,topk_percents, plots_dir/f"topk_capture_{m}_test.png")

		if make_cal and "calibration" in res_t:
			c = res_t["calibration"]
			save_calibration_plot(
				np.array(c["bin_centers"]),
				np.array(c["empirical_rate"],dtype=float),
				np.array(c["predicted_rate"],dtype=float),
				plots_dir/f"calibration_{m}_test.png",
			)

			# stacking candiates require both valid and test present
			valid_scores_matrix.append(sv)
			test_scores_matrix.append(st)
			base_order.append(m)

	# save base metrices to JSON
	out_json = metrics_dir /"holdout_metrics.json"
	with open(out_json,"w", encoding="utf-8") as f:
		json.dump(results,f,indent=2)
	print(f"[eval] wrote: {out_json}")

	# Write topk table csv for lightgbm valid sample
	if "lightgbm" in results["base_models"] and "valid" in results["base_models"]["lightgbm"]:
		topk = results["base_models"]["lightgbm"]["valid"]["topk"]
		pd.DataFrame(topk).to_csv(metrics_dir/"topk_table.csv", index=False)
		print(f"[eval] wrote: {metrics_dir/'topk_table.csv'}")

	# ------------------------
	# 2) Stacking (when base >=2 with valid + test scores)
	# Meta model trains on VALID and evaluates on TEST
	# ------------------------
	if len(base_order)>=2:
		V = np.vstack(valid_scores_matrix).T # shape (n_valid, n_models)
		T = np.vstack(test_scores_matrix).T # shape (n_test, n_models)

		# Need y_valid and y_test from any base file, use first model's files
		first = base_order[0]
		files = discover_more_files(outputs_dir,first)
		yv,_,_ = load_scored(files["valid"],target_col)
		yt,_,_ = load_scored(files["test"], target_col)

		# linear stacking
		stacked_lr = stack_linear(V,yv,T)
		res_lr = evaluate_one(yt, stacked_lr, topk_percents, make_calibration=make_cal,cal_bins=cal_bins)
		results["stacking"]["linear_logit_on_scores"] = {
			"base_order": base_order,
			"test": res_lr,
		}
		save_pr_curve(yt, stacked_lr, plots_dir/"pr_curve_stack_linear_test.png")
		save_lift_curve(yt, stacked_lr, plots_dir/"lift_curve_stack_linear_test.png")
		save_topk_capture(yt, stacked_lr, topk_percents, plots_dir/"topk_capture_stack_linear_test.png")

		# GBM stacking
		stacked_gbm = stack_gbm(V,yv,T)
		res_gbm = evaluate_one(yt, stacked_gbm, topk_percents, make_calibration=make_cal,cal_bins=cal_bins)
		results["stacking"]["gbm_logit_on_scores"] = {
			"base_order": base_order,
			"test": res_gbm,
		}
		save_pr_curve(yt, stacked_gbm, plots_dir/"pr_curve_stack_gbm_test.png")
		save_lift_curve(yt, stacked_gbm, plots_dir/"lift_curve_stack_gbm_test.png")
		save_topk_capture(yt, stacked_gbm, topk_percents, plots_dir/"topk_capture_stack_gbm_test.png")

		# save stacking metrices
		with open(out_json, "w", encoding = "utf-8") as f:
			json.dump(results,f,indent=2)

		print("[eval] stacking completed and metrics updated.")
	else:
		print("[eval] stacking skipped (need >=2 base models with BOTH valid and test scores)")
		print("[eval] next: train xgboost/logit/torch_mlp to produce scored_valid/test_<model>.csv")
	
if __name__ == "__main__":
	main()