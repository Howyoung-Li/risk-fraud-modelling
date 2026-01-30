from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils.config import load_config
from .utils.io import ensure_dir

import matplotlib.pyplot as plt

def expected_cost(y, score, amt, thr, fp_cost, fn_cost_multiplier):
	pred = (score>=thr).astype(int)
	fp = (pred==1) & (y==0)
	fn = (pred==0) & (y==1)
	cost = fp_cost * fp.sum() + fn_cost_multiplier * float((amt[fn]).sum())
	return float(cost)

def main():
	cfg = load_config()

	artifacts_dir = cfg.artifacts_dir
	out_dir = artifacts_dir/"outputs"
	policy_dir = artifacts_dir/"policy"
	ensure_dir(policy_dir)

	id_col = cfg.split.id_col
	target_col = cfg.split.target_col

	scored_path = out_dir/"scored_test.csv" # lightgbm champion
	if not scored_path.exists():
		raise FileNotFoundError(f"missing: {scored_path}")
	
	scored = pd.read_csv(scored_path)

	# Need TransactionAmt
	amount_col= cfg.evaluation.amount_col
	if amount_col not in scored.columns:
		table_path = cfg.processed_dir/"model_table_fe.parquet"
		df = pd.read_parquet(table_path,columns=[id_col,amount_col])
		scored = scored.merge(df, on=id_col,how="left", validate="one_to_one")

	# Config
	review_cap = 1.0
	fp_cost = 10.0
	fn_mult = 1.0
	optimize = "expected_cost"
	min_recall = 0.60
	grid_n = 400

	y = scored[target_col].astype(int).to_numpy()
	s = scored["score"].astype(float).to_numpy()
	amt = scored[amount_col].astype(float).to_numpy()

	# review-capacity threshold (top x%)
	n = len(s)
	k = max(1, int(round(n * (review_cap/100.0))))
	thr_cap = float(np.sort(s)[-k]) # smallest score in top k
	
	# Grid search thresholds over score quantiles (fast + robust)
	qs = np.linspace(0.0, 1.0, grid_n)
	thrs = np.quantile(s, qs)
	thrs = np.unique(thrs)

	total_pos = y.sum()
	total_fraud_amt = float((amt * y).sum()) if total_pos > 0 else 0.0

	rows = []
	for thr in thrs:
		pred = (s>=thr).astype(int)
		tp = int(((pred==1) & (y==1)).sum())
		fp = int(((pred==1) & (y==0)).sum())
		fn = int(((pred==0) & (y==1)).sum())

		reviewed = int(pred.sum())
		precision = tp / reviewed if reviewed > 0 else 0.0
		recall = tp / total_pos if total_pos > 0 else 0.0

		fraud_amt_captured = float((amt[(pred==1) & (y==1)]).sum())
		reviewed_amt = float(amt[pred==1].sum())
		dollar_recall = fraud_amt_captured/ total_fraud_amt if total_fraud_amt > 0 else 0.0
		dollar_precision = fraud_amt_captured/reviewed_amt if reviewed_amt > 0 else 0.0

		cost = fp_cost * fp + fn_mult * float(amt[(pred==0) & (y==1)].sum())

		rows.append({
			"threshold": float(thr),
			"reviewed_n": reviewed,
			"reviewed_percent": 100.0 * reviewed/n,
			"precision": float(precision),
			"recall": float(recall),
			"dollar_precision": float(dollar_precision),
			"dollar_recall": float(dollar_recall),
			"expected_cost": float(cost),
		})

	table = pd.DataFrame(rows).sort_values("threshold")
	table.to_csv(policy_dir/ "policy_tradeoff_table.csv", index=False)

	# Recall

	# choose threshold
	if optimize == "expected_cost":
		best = table.loc[table["expected_cost"].idxmin()]
	elif optimize == "constraint_recall":
		feas = table[table["recall"] >= min_recall]
		best = feas.loc[feas["expected_cost"].idxmin()] if len(feas) else table.loc[table["expected_cost"].idxmin()]
	else:
		raise ValueError(f"Unknown optimize= {optimize}")
		
	# provide two thresholds: capacity-based and cost-based
	thresholds = {
		"capacity_threshold": thr_cap,
		"cost_opt_threshold": float(best["threshold"]),
		"capacity_review_percent": review_cap,
		"selected_optimize": optimize,
		"selected_min_recall": min_recall if optimize=="constraint_recall" else None,

	}

	# Plot
	# ensure sorted for plotting lines
	table_plot = table.sort_values("reviewed_percent").reset_index(drop=True)

	cap_thr = thr_cap
	opt_thr = float(best["threshold"])
	cap_review = review_cap
	opt_review = float(best["reviewed_percent"])

	# cost vs reviewed %
	plt.figure(figsize=(7,4))
	plt.plot(table_plot["reviewed_percent"], table_plot["expected_cost"])
	plt.axvline(cap_review, linestyle="--")
	plt.axvline(opt_review, linestyle="--")
	plt.xlabel("Reviewed (%)")
	plt.ylabel ("Expected cost")
	plt.title("Expected cost vs review rate")
	plt.tight_layout()
	plt.savefig(policy_dir/"cost_vs_review.png", dpi=150)
	plt.close()

	# Expected cost vs threshold
	table_thr = table.sort_values("threshold").reset_index(drop=True)
	plt.figure(figsize=(7,4))
	plt.plot(table_thr["threshold"], table_thr["expected_cost"])
	plt.axvline(cap_thr, linestyle="--")
	plt.axvline(opt_thr, linestyle="--")
	plt.xlabel("Threshold")
	plt.ylabel("Expected cost")
	plt.title("Expected cost vs threshold")
	plt.tight_layout()
	plt.savefig(policy_dir/"cost_vs_threshold.png", dpi=150)
	plt.close()

	# Dollar metrics vs reviewed %
	plt.figure(figsize=(7,4))
	plt.plot(table_plot["reviewed_percent"], table_plot["dollar_recall"], label="dollar recall")
	plt.plot(table_plot["reviewed_percent"], table_plot["dollar_precision"], label="dollar precision")
	plt.axvline(cap_review, linestyle="--")
	plt.axvline(opt_review, linestyle="--")
	plt.xlabel("reviewed (%)")
	plt.ylabel("Metric")
	plt.title("Dollar metrics vs review rate")
	plt.legend()
	plt.tight_layout()
	plt.savefig(policy_dir/ "dollar_metrics_vs_review_rate.png", dpi=150)
	plt.close()

	# score distribution by class + overlay thresholds
	s0 = s[y==0]
	s1 = s[y==1]
	bins = 50

	plt.figure(figsize=(7,4))
	plt.hist(s0, bins=bins, density=True, alpha=0.6, label="y=0 (legit)")
	plt.hist(s1,bins=bins, density=True, alpha=0.6, label="y=1 (fraud)")
	plt. axvline(cap_thr, linestyle="--", label=f"Capacity thr ({cap_review:.1f}%)")
	plt.axvline(opt_thr, linestyle="--", label="Cost-opt thr")
	plt.xlabel("Score")
	plt.ylabel("Density")
	plt.title("Score distribution by class")
	plt.legend()
	plt.tight_layout()
	plt.savefig(policy_dir/ "score_distribution_by_class.png", dpi=150)
	plt.close()

	with open(policy_dir /"threshold.json", "w", encoding="utf-8") as f:
		json.dump(thresholds, f, indent=2)

	print(f"[policy] wrote: {policy_dir/'policy_tradeoff_table.csv'}")
	print(f"[policy] wrote: {policy_dir/'threshold.json'}")
	print(f"[policy] capacity threshold (top {review_cap}%: {thr_cap:.6f})")
	print(f"[policy] cost-opt threshold: {float(best['threshold']):.6f} | reviewed={best['reviewed_percent']:.3f}% | recall={best['recall']:.3f} | precision={best['precision']:.3f}")

if __name__=="__main__":
	main()