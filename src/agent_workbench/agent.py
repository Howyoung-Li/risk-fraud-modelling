from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .tools import RiskToolRegistry, ToolResult


@dataclass(frozen=True)
class RiskAgentResponse:
	question: str
	intent: str
	answer: str
	tools_called: list[str]
	evidence: list[str]
	data: dict[str, Any]


class RiskAgent:
	"""A deterministic agent shell for interview-safe fraud risk analysis demos."""

	def __init__(self, tools: RiskToolRegistry | None = None) -> None:
		self.tools = tools or RiskToolRegistry()

	def answer(self, question: str) -> RiskAgentResponse:
		intent = self._route(question)
		result = self._call_tool(intent, question)
		answer = self._format_answer(question, result)
		return RiskAgentResponse(
			question=question,
			intent=intent,
			answer=answer,
			tools_called=[result.name],
			evidence=result.evidence,
			data=result.data,
		)

	def _route(self, question: str) -> str:
		q = question.lower()
		if any(token in q for token in ["brief", "report", "summary", "报告", "总结", "复盘"]):
			return "generate_risk_brief"
		if any(token in q for token in ["shap", "reason", "case", "transaction", "解释", "原因码", "单"]):
			return "explain_case"
		if any(token in q for token in ["psi", "drift", "monitor", "监控", "漂移", "稳定"]):
			return "monitor_drift"
		if any(token in q for token in ["policy", "threshold", "cost", "decline", "review", "策略", "阈值", "审核", "成本"]):
			return "policy_simulation"
		return "model_performance"

	def _call_tool(self, intent: str, question: str) -> ToolResult:
		if intent == "explain_case":
			matched = re.search(r"\b(\d{6,})\b", question)
			transaction_id = int(matched.group(1)) if matched else None
			return self.tools.explain_case(transaction_id)
		if intent == "monitor_drift":
			return self.tools.monitor_drift()
		if intent == "policy_simulation":
			return self.tools.policy_simulation()
		if intent == "generate_risk_brief":
			return self.tools.generate_risk_brief()
		return self.tools.model_performance()

	def _format_answer(self, question: str, result: ToolResult) -> str:
		evidence = "; ".join(result.evidence)
		return f"{result.summary}\nEvidence: {evidence}"
