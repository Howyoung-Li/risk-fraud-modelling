from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt

from .utils.config import load_config
from .utils.io import ensure_dir

def psi(expected, actual, bins=10, eps=1e-6):
	expected = np.asarray(expected,dtype=float)
	actual = np.asarray(actual, dtype=float)

	qs = np.linspace(0,1, bins + 1)
	cuts = np.quantile(expected, qs)
	cuts = np.unique(cuts)
	if len(cuts) < 3: # degenerate
		return 0.0
	
	e_hist, _ = np.histogram(expected,bins=cuts)
	a_hist, _ = np.histogram(actual, bins=cuts)

	e = e_hist / (e_hist.sum() + eps)
	a = a_hist / (a_hist.sum() + eps)

	e = np.clip(e, eps, None)
	a = np.clip(a,eps,None)

	return float(np.sum((a - e) * np.log(a/e)))

def main():
	cfg = load_config()

	artifacts_dir = cfg.artifacts_dir
	out_dir = artifacts_dir/"outputs"
	mon_dir = artifacts_dir/"monitoring"
	plots_dir = artifacts_dir/"plots"
	ensure_dir(mon_dir)
	ensure_dir(plots_dir)

	id_col = cfg.split.id_col
	time_col = cfg.split.time_col

	scored_path = out_dir/"scored_test.csv"
	if not scored_path.exists():
		raise FileNotFoundError(f"missing: {scored_path}")
	scored = pd.read_csv(scored_path)

	table_path = cfg.processed_dir/"model_table_fe.parquet"
	df = pd.read_parquet(table_path)

	# merge time_day and candidate features onto scored
	if "time_day" not in df.columns:
		raise ValueError("time_day not found in model_table_fe.parquet. Make sure it exists.")
	
	# choose features for PSI (numeric only)
	drop_cols = [c for c in [id_col, time_col, cfg.split.target_col, "uid_hash"] if c in df.columns]
	feat_cols = [c for c in df.columns if c not in drop_cols]
	numeric_feats = [c for c in feat_cols if is_numeric_dtype(df[c])]
	numeric_feats = [c for c in numeric_feats if c != "time_day"]
	psi_thr = 0.25
	windows = [7,14,30]
	topk = 30

	# Merge
	keep_cols = [id_col, "time_day"] + numeric_feats
	joined = scored[[id_col, "score"]].merge(df[keep_cols], on=id_col, how="left", validate="one_to_one")

	# Enforce time_day numeric scalar and drop missing merges
	joined["time_day"] = pd.to_numeric(joined["time_day"], errors="coerce")
	joined = joined.dropna(subset=["time_day"]).copy()
	joined["time_day"] = joined["time_day"].astype(int)

	# define reference window as earliest window in test set
	tmin = int(joined["time_day"].min())
	tmax = int(joined["time_day"].max())

	cand = [c for c in numeric_feats if c not in ("time_day",)]
	tmp = joined[cand].apply(pd.to_numeric, errors="coerce").astype("float32")
	var = tmp.var(numeric_only=True).sort_values(ascending=False)
	psi_features = var.head(topk).index.tolist()

	# score drift plot data
	drift_rows = []
	psi_rows = []

	ref_start = tmin
	ref_end = tmin + max(windows)
	ref = joined[(joined["time_day"] >= ref_start) & (joined["time_day"] < ref_end)].copy()
	if len(ref) < 1000:
		# fallback: first 10% of data by time_day
		cutoff = np.quantile(joined["time_day"], 0.1)
		ref = joined[joined["time_day"] <= cutoff].copy()
		
	ref_score = ref["score"].to_numpy()

	# For each window size, cuompute PSI on score and on features for later windows
	for w in windows:
		start = tmin
		step = 1
		while start + w <= tmax + 1:
			win = joined[(joined["time_day"] >= start) & (joined["time_day"] < start + w)]
			if len(win) < 500:
				start += step
				continue

			score_psi = psi(ref_score, win["score"].to_numpy(), bins=10)
			drift_rows.append({
				"window_days": w,
				"start_day": start,
				"end_day": start + w,
				"n": int(len(win)),
				"score_mean": float(win["score"].mean()),
				"score_std": float(win["score"].std()),
				"score_psi": float(score_psi),
				"score_psi_alert": bool(score_psi >= psi_thr),
			})

			# Feature PSI
			for f in psi_features:
				a = win[f].to_numpy()
				e = ref[f].to_numpy()
				# handle NaN
				e = np.nan_to_num(e, nan = np.nanmedian(e))
				a = np.nan_to_num(a, nan = np.nanmedian(e))
				p = psi(e, a, bins=10)
				psi_rows.append({
					"window_days": w,
					"start_day": start,
					"end_day": start + w,
					"feature": f,
					"psi": float(p),
					"psi_alert": bool(p >= psi_thr),
				})

			start += w
	drift_df =pd.DataFrame(drift_rows)
	psi_df = pd.DataFrame(psi_rows)

	drift_df.to_csv(mon_dir / "score_drift.csv", index=False)
	psi_df.to_csv(mon_dir / "psi_report.csv", index=False)

	# Plot: score PSI over time for smallest window
	if len(drift_df):
		w0 = min(windows)
		sub = drift_df[drift_df["window_days"] == w0].sort_values("start_day")
		plt.figure(figsize=(10,4))
		plt.plot(sub["start_day"], sub["score_psi"])
		plt.axhline(psi_thr, linestyle="--")
		plt.xlabel("Start Day")
		plt.ylabel("Score PSI")
		plt.title(f"Score PSI drift (window={w0} days)")
		plt.tight_layout()
		plt.savefig(plots_dir/"score_drift.png", dpi=150)
		plt.close()
	
	# Plot: top PSI features (max PSI across windows)
	if len(psi_df):
		agg = psi_df.groupby("feature")["psi"].max().sort_values(ascending=False).head(10)
		plt.figure(figsize=(8,10))
		plt.barh(agg.index[::-1], agg.values[::-1])
		plt.title("Top PSI Features (max over windows)")
		plt.tight_layout()
		plt.savefig(plots_dir/"psi_top_features.png", dpi=150)
		plt.close()


	spec = {
		"reference_window": {"start_day": int(ref_start), "end_day": int(ref_end), "n": int(len(ref))},
		"psi_threshold": psi_thr,
		"window_days": windows,
		"top_features_for_psi": topk,
		"notes": "PSI computed using quantile bins from reference distribution."
	}
	with open(mon_dir/"monitoring_spec.json", "w", encoding="utf-8") as f:
		json.dump(spec,f,indent=2)

	print(f"[monitor] wrote: {mon_dir/'psi_report.csv'}")
	print(f"[monitor] wrote: {plots_dir/'psi_top_features.png'}")
	print(f"[monitor] wrote: {plots_dir / 'score_drift.png'}")

if __name__ == "__main__":
	main()