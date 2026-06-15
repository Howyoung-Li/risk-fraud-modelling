import unittest

from src.agent_workbench import RiskAgent
from src.agent_workbench.eval_harness import run_eval_harness


class RiskAgentWorkbenchTest(unittest.TestCase):
	def test_routes_policy_question_to_policy_tool(self):
		response = RiskAgent().answer("How should we set review and decline thresholds?")
		self.assertEqual(response.intent, "policy_simulation")
		self.assertIn("policy_tradeoff_table.csv", " ".join(response.evidence))

	def test_eval_harness_passes_default_cases(self):
		result = run_eval_harness(RiskAgent())
		self.assertEqual(result["passed_cases"], result["total_cases"])
		self.assertGreaterEqual(result["pass_rate"], 1.0)


if __name__ == "__main__":
	unittest.main()
