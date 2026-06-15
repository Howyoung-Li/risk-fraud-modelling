from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .artifact_store import ArtifactStore


@dataclass(frozen=True)
class ToolResult:
	name: str
	summary: str
	data: dict[str, Any]
	evidence: list[str]


class RiskToolRegistry:
	"""Deterministic tools that expose model assets to the risk analyst agent."""

	def __init__(self, store: ArtifactStore | None = None) -> None:
		self.store = store or ArtifactStore()

	def model_performance(self) -> ToolResult:
		metrics = self.store.read_json("metrics", "holdout_metrics.json")
		topk = self.store.read_csv("metrics", "topk_table.csv")
		lgbm = metrics.get("base_models", {}).get("lightgbm", metrics.get("lightgbm", {}))
		test = lgbm.get("test", lgbm.get("Test", {}))
		top_rows = topk.to_dict(orient="records")
		pr_auc = float(test.get("pr_auc", test.get("PR-AUC", 0.0)))
		roc_auc = float(test.get("roc_auc", test.get("ROC-AUC", 0.0)))
		ks = float(test.get("ks", test.get("KS", 0.0)))
		summary = (
			"LightGBM is the champion OOT model. "
			f"Test PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}, KS={ks:.4f}."
		)
		return ToolResult(
			name="model_performance",
			summary=summary,
			data={"holdout_metrics": metrics, "topk_table": top_rows},
			evidence=[
				self.store.evidence_path("metrics", "holdout_metrics.json"),
				self.store.evidence_path("metrics", "topk_table.csv"),
			],
		)

	def review_capacity(self, percent: float = 1.0) -> ToolResult:
		topk = self.store.read_csv("metrics", "topk_table.csv")
		idx = (topk["topk_percent"].astype(float) - percent).abs().idxmin()
		row = topk.loc[idx].to_dict()
		summary = (
			f"At top {row['topk_percent']:.1f}% manual review capacity, "
			f"precision is {row['precision_at_k']:.2%} and recall is {row['recall_at_k']:.2%}."
		)
		return ToolResult(
			name="review_capacity",
			summary=summary,
			data={"selected_capacity": row},
			evidence=[self.store.evidence_path("metrics", "topk_table.csv")],
		)

	def policy_simulation(self) -> ToolResult:
		summary = self.store.read_csv("policy", "decision_policy_summary.csv", required=False)
		thresholds = self.store.read_json("policy", "threshold.json", default={})
		tradeoff = self.store.read_csv("policy", "policy_tradeoff_table.csv")
		best_idx = tradeoff["expected_cost"].astype(float).idxmin()
		best = tradeoff.loc[best_idx].to_dict()
		decision = summary.iloc[0].to_dict() if len(summary) else {}
		text = (
			"Three-way policy produces approve/manual_review/decline actions. "
			f"Cost-min threshold reviews {best['reviewed_percent']:.2f}% with "
			f"precision {best['precision']:.2%}, recall {best['recall']:.2%}, "
			f"and expected cost {best['expected_cost']:.2f}."
		)
		return ToolResult(
			name="policy_simulation",
			summary=text,
			data={"best_threshold": best, "three_way_policy": decision, "thresholds": thresholds},
			evidence=[
				self.store.evidence_path("policy", "policy_tradeoff_table.csv"),
				self.store.evidence_path("policy", "decision_policy_summary.csv"),
				self.store.evidence_path("policy", "threshold.json"),
			],
		)

	def explain_case(self, transaction_id: int | None = None) -> ToolResult:
		reason_path = self.store.path("outputs", "reason_codes_sample.csv")
		if reason_path.exists():
			reasons = self.store.read_csv("outputs", "reason_codes_sample.csv")
			evidence = self.store.evidence_path("outputs", "reason_codes_sample.csv")
		else:
			reasons = self.store.read_csv("agent", "case_reason_codes_sample.csv")
			evidence = self.store.evidence_path("agent", "case_reason_codes_sample.csv")
		if transaction_id is not None and "TransactionID" in reasons.columns:
			matched = reasons[reasons["TransactionID"].astype(str) == str(transaction_id)]
			row = (matched.iloc[0] if len(matched) else reasons.iloc[0]).to_dict()
		else:
			row = reasons.sort_values("score", ascending=False).iloc[0].to_dict()

		reason_items = []
		for i in range(1, 6):
			reason = row.get(f"reason_{i}")
			contrib = row.get(f"contrib_{i}")
			if pd.notna(reason) and pd.notna(contrib):
				reason_items.append({"feature": reason, "contribution": float(contrib)})

		summary = (
			f"Transaction {int(row['TransactionID'])} has score {float(row['score']):.6f}. "
			"Top positive reason codes are "
			+ ", ".join(f"{item['feature']} ({item['contribution']:.2f})" for item in reason_items[:3])
			+ "."
		)
		return ToolResult(
			name="explain_case",
			summary=summary,
			data={"case": row, "reason_codes": reason_items},
			evidence=[evidence],
		)

	def monitor_drift(self) -> ToolResult:
		psi = self.store.read_csv("monitoring", "psi_report.csv")
		score = self.store.read_csv("monitoring", "score_drift.csv")
		top_psi = (
			psi.sort_values("psi", ascending=False)
			.head(10)[["window_days", "feature", "psi", "psi_alert"]]
			.to_dict(orient="records")
		)
		latest_score = score.sort_values(["window_days", "end_day"]).tail(1).iloc[0].to_dict()
		alert_count = int(psi["psi_alert"].astype(bool).sum()) if "psi_alert" in psi else 0
		summary = (
			f"Monitoring found {alert_count} PSI feature alerts. "
			f"Latest score window mean is {float(latest_score['score_mean']):.4f} "
			f"with score PSI {float(latest_score['score_psi']):.4f}."
		)
		return ToolResult(
			name="monitor_drift",
			summary=summary,
			data={"top_psi_features": top_psi, "latest_score_drift": latest_score, "psi_alert_count": alert_count},
			evidence=[
				self.store.evidence_path("monitoring", "psi_report.csv"),
				self.store.evidence_path("monitoring", "score_drift.csv"),
			],
		)

	def generate_risk_brief(self) -> ToolResult:
		perf = self.model_performance()
		policy = self.policy_simulation()
		drift = self.monitor_drift()
		case = self.explain_case()
		summary = "\n".join([perf.summary, policy.summary, drift.summary, case.summary])
		return ToolResult(
			name="generate_risk_brief",
			summary=summary,
			data={
				"performance": perf.data,
				"policy": policy.data,
				"monitoring": drift.data,
				"example_case": case.data,
			},
			evidence=sorted(set(perf.evidence + policy.evidence + drift.evidence + case.evidence)),
		)
