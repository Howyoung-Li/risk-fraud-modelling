from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd

from src.agent_workbench import RiskAgent
from src.agent_workbench.eval_harness import run_eval_harness


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> str | None:
	if not src.exists():
		return None
	dst.parent.mkdir(parents=True, exist_ok=True)
	shutil.copy2(src, dst)
	return str(dst.relative_to(ROOT / "docs"))


def write_public_reason_sample() -> None:
	src = ROOT / "artifacts/outputs/reason_codes_sample.csv"
	dst = ROOT / "artifacts/agent/case_reason_codes_sample.csv"
	if not src.exists():
		return
	dst.parent.mkdir(parents=True, exist_ok=True)
	cols = [
		"TransactionID",
		"isFraud",
		"score",
		"reason_1",
		"contrib_1",
		"reason_2",
		"contrib_2",
		"reason_3",
		"contrib_3",
		"reason_4",
		"contrib_4",
		"reason_5",
		"contrib_5",
	]
	pd.read_csv(src, usecols=cols).head(20).to_csv(dst, index=False)


def serialize_response(response) -> dict:
	return {
		"question": response.question,
		"intent": response.intent,
		"answer": response.answer,
		"tools_called": response.tools_called,
		"evidence": response.evidence,
		"trace": response.trace,
	}


def main() -> None:
	write_public_reason_sample()
	agent = RiskAgent()
	questions = [
		"What is the OOT model performance?",
		"At top 1% review capacity, what are precision and recall?",
		"How should we set review and decline thresholds?",
		"Explain transaction 3402093 with reason codes.",
		"Is there any PSI or score drift alert?",
		"Generate a concise risk monitoring report.",
	]
	responses = [serialize_response(agent.answer(q)) for q in questions]
	eval_result = run_eval_harness(agent)

	assets = {
		"pr_curve": copy_asset(
			ROOT / "artifacts/plots/pr_curve_lightgbm_test.png",
			ROOT / "docs/assets/plots/pr_curve_lightgbm_test.png",
		),
		"topk_capture": copy_asset(
			ROOT / "artifacts/plots/topk_capture_lightgbm_test.png",
			ROOT / "docs/assets/plots/topk_capture_lightgbm_test.png",
		),
		"policy_cost": copy_asset(
			ROOT / "artifacts/policy/cost_vs_review.png",
			ROOT / "docs/assets/plots/cost_vs_review.png",
		),
		"score_drift": copy_asset(
			ROOT / "artifacts/plots/score_drift.png",
			ROOT / "docs/assets/plots/score_drift.png",
		),
		"shap_summary": copy_asset(
			ROOT / "artifacts/shap/shap_beeswarm_feature_value_red_high_blue_low.png",
			ROOT / "docs/assets/plots/shap_beeswarm_feature_value_red_high_blue_low.png",
		),
	}

	workbench = {
		"title": "Fraud Risk Agent Workbench",
		"subtitle": "Evidence-bound AI analyst layer over the IEEE-CIS fraud model pipeline.",
		"agent_responses": responses,
		"eval": eval_result,
		"assets": assets,
		"architecture": [
			"User question",
			"Intent router",
			"Risk tool registry",
			"Model, policy, SHAP, monitoring artifacts",
			"Evidence-bound answer",
			"Evaluation harness",
		],
	}
	write_json(ROOT / "artifacts/agent/workbench.json", workbench)
	write_json(ROOT / "docs/data/workbench.json", workbench)
	print("[agent] wrote artifacts/agent/workbench.json")
	print("[pages] wrote docs/data/workbench.json and copied chart assets")


if __name__ == "__main__":
	main()
