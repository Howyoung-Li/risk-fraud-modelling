from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .agent import RiskAgent


@dataclass(frozen=True)
class EvalCase:
	name: str
	question: str
	expected_intent: str
	required_evidence: str


DEFAULT_EVAL_CASES = [
	EvalCase(
		name="model_metrics",
		question="What is the OOT model performance?",
		expected_intent="model_performance",
		required_evidence="holdout_metrics.json",
	),
	EvalCase(
		name="policy_tradeoff",
		question="How should we set review and decline thresholds?",
		expected_intent="policy_simulation",
		required_evidence="policy_tradeoff_table.csv",
	),
	EvalCase(
		name="review_capacity_top_1",
		question="At top 1% review capacity, what are precision and recall?",
		expected_intent="review_capacity",
		required_evidence="topk_table.csv",
	),
	EvalCase(
		name="review_capacity_chinese",
		question="如果只审核 top 3% 交易，召回和命中率是多少？",
		expected_intent="review_capacity",
		required_evidence="topk_table.csv",
	),
	EvalCase(
		name="case_explanation",
		question="Explain transaction 3402093 with reason codes.",
		expected_intent="explain_case",
		required_evidence="reason_codes_sample.csv",
	),
	EvalCase(
		name="case_explanation_chinese",
		question="解释一下交易 3402093 的风险原因码",
		expected_intent="explain_case",
		required_evidence="reason_codes_sample.csv",
	),
	EvalCase(
		name="drift_monitoring",
		question="Is there any PSI or score drift alert?",
		expected_intent="monitor_drift",
		required_evidence="psi_report.csv",
	),
	EvalCase(
		name="drift_monitoring_chinese",
		question="模型最近有没有漂移或者 PSI 告警？",
		expected_intent="monitor_drift",
		required_evidence="score_drift.csv",
	),
	EvalCase(
		name="risk_brief",
		question="Generate a concise risk monitoring report.",
		expected_intent="generate_risk_brief",
		required_evidence="score_drift.csv",
	),
]


def run_eval_harness(agent: RiskAgent | None = None, cases: Iterable[EvalCase] = DEFAULT_EVAL_CASES) -> dict:
	agent = agent or RiskAgent()
	rows = []
	for case in cases:
		response = agent.answer(case.question)
		evidence_text = " ".join(response.evidence)
		passed = response.intent == case.expected_intent and case.required_evidence in evidence_text
		rows.append(
			{
				**asdict(case),
				"actual_intent": response.intent,
				"tools_called": response.tools_called,
				"evidence": response.evidence,
				"passed": passed,
			}
		)
	total = len(rows)
	passed = sum(1 for row in rows if row["passed"])
	return {
		"total_cases": total,
		"passed_cases": passed,
		"pass_rate": passed / total if total else 0.0,
		"cases": rows,
	}
