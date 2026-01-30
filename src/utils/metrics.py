"""Minimal for training log"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
	roc_auc_score,
	average_precision_score,
	roc_curve,
	precision_recall_curve,
	brier_score_loss,
)

def auc_roc(y_true, y_score):
	return float(roc_auc_score(y_true, y_score))

def auc_pr(y_true, y_score):
	return float(average_precision_score(y_true, y_score))

def ks_stat(y_true, y_score) -> float:
	fpr, tpr, _ = roc_curve(y_true, y_score)
	return float(np.max(tpr - fpr))

def topk_table(y_true, y_score, topk_percents,amount=None):
	"""
	returns a list of dict rows with Recall@TopK, Precison@TopK, FraudRate@TopK.
	topk_percents are percentages like [0.1, 0.5, 1, 3, 5, 10]
	"""
	y_true = np.asarray(y_true).astype(int)
	y_score = np.asarray(y_score).astype(float)
	
	n = len(y_true)
	order = np.argsort(-y_score) # descending
	y_sorted = y_true[order]

	amt_sorted = None
	if amount is not None:
		amt = np.asarray(amount).astype(float)
		amt_sorted = amt[order]

	total_pos = y_true.sum()
	total_fraud_amt = None
	if amt_sorted is not None:
		total_fraud_amt = float((amt_sorted * y_sorted).sum())

	out=[]
	for p in topk_percents:
		k = max(1, int(np.ceil(n * (p /100.0))))
		top_y = y_sorted[:k]
		tp = int(top_y.sum())
		precision = tp/k
		recall = tp/total_pos if total_pos > 0 else 0.0

		row ={
			"topk_percent": float(p),
			"k": int(k),
			"precision_at_k": float(precision),
			"recall_at_k": float(recall),
		}

		if amt_sorted is not None:
			top_amt =amt_sorted[:k]
			fraud_amt_in_top = float((top_amt * top_y).sum())
			reviewed_amt = float((top_amt).sum())

			dollar_recall = fraud_amt_in_top/total_fraud_amt if fraud_amt_in_top and total_fraud_amt > 0 else 0.0
			dollar_precision = fraud_amt_in_top/ reviewed_amt if reviewed_amt > 0 else 0.0

			row.update({
				"fraud_amount_in_topk": fraud_amt_in_top,
				"reviewed_amount_topk": reviewed_amt,
				"dollar_recall_at_k": float(dollar_recall),
				"dollar_precision_at_k": float(dollar_precision),
			})

		out.append(row)
	return out

def calibration_bins(y_true, y_prob, n_bins: int=20):
	"""
	Returns bin centers, empirical fraud rate, mean predicted prob per bin.
	"""
	y_true = np.asarray(y_true).astype(int)
	y_prob = np.asarray(y_prob).astype(float)

	bins = np.linspace(0.0,1.0, n_bins + 1)
	idx = np.digitize(y_prob, bins) - 1
	idx = np.clip(idx,0, n_bins -1)

	emp = np.zeros(n_bins, dtype=float)
	pred = np.zeros(n_bins, dtype=float)
	cnt = np.zeros(n_bins, dtype=int)

	for b in range(n_bins):
		m = (idx ==b)
		cnt[b] = int(m.sum())
		if cnt[b] > 0:
			emp[b] = float(y_true[m].mean())
			pred[b] = float(y_prob[m].mean())
		else:
			emp[b] = np.nan
			pred[b] = np.nan

	centers = (bins[:-1] + bins[1:])/2.0
	return centers, emp, pred, cnt

def brier(y_true, y_prob) -> float:
	return float(brier_score_loss(y_true, y_prob))

