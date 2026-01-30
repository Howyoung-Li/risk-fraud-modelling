from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

def save_pr_curve(y_true, y_score, out_path):
	precision, recall, _ = precision_recall_curve(y_true, y_score)
	plt.figure()
	plt.plot(recall, precision)
	plt.xlabel("Recall")
	plt.ylabel("Precison")
	plt.title("Precison-Recall curve")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()

def save_lift_curve(y_true, y_score, out_path):
	"""
	Gains/lift style: x= population fraction, y = cumulative recall (fraud captured)
	"""
	y_true = np.asarray(y_true).astype(int)
	y_score = np.asarray(y_score).astype(float)

	order = np.argsort(-y_score)
	y_sorted = y_true[order]
	cum_pos = np.cumsum(y_sorted) # returns a vector
	total_pos = y_true.sum()

	x= np.arange(1, len(y_true) + 1) / len(y_true) # fractions of sample included
	y = cum_pos/total_pos if total_pos > 0 else cum_pos * 0.0

	plt.figure()
	plt.plot(x,y, label = "Model")
	plt.plot([0,1], [0,1], linestyle = "--", label="Random")
	plt.xlabel("polulation fraction")
	plt.ylabel("cumulative fraud captured (recall)")
	plt.title("Gains / Lift Curve")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()

def save_topk_capture(y_true, y_score, topk_percents, out_path):
	y_true = np.asarray(y_true).astype(int)
	y_score = np.asarray(y_score).astype(float)
	n = len(y_true)
	order = np.argsort(-y_score)
	y_sorted = y_true[order]
	total_pos = y_sorted.sum()

	xs = []
	ys = []
	for p in topk_percents:
		k = max(1, int(np.ceil(n * (p/100.0))))
		tp = int(y_sorted[:k].sum())
		recall = tp/total_pos if total_pos > 0 else 0.0
		xs.append(p)
		ys.append(recall)

		plt.figure()
		plt.plot(xs,ys, marker = "o")
		plt.xlabel("Top-K Reviewed (%)")
		plt.ylabel("Recall (Fraud captured)")
		plt.tight_layout()
		plt.savefig(out_path, dpi=150)
		plt.close()

def save_calibration_plot(bin_centers, emp_rate, pred_rate,out_path):
	plt.figure()
	plt.plot(bin_centers, emp_rate, marker="o", label = "Empirical")
	plt.plot(bin_centers, pred_rate, marker ="o", label="Predicted")
	plt.plot([0,1],[0,1], linestyle="--", label="Perfect")
	plt.xlabel("Predicted probability")
	plt.ylabel("Observed Fraud rate")
	plt.title("Calibration Curve")
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()
