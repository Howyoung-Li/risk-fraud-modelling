import importlib
import importlib.util
import unittest

import numpy as np

for dependency in ["pandas", "matplotlib"]:
	if importlib.util.find_spec(dependency) is None:
		raise unittest.SkipTest(f"{dependency} is not installed in this interpreter")

monitoring = importlib.import_module("src.07_monitoring_pack")


class MonitoringUtilsTest(unittest.TestCase):
	def test_psi_counts_actual_values_outside_reference_range(self):
		expected = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
		actual = np.array([2.0, 2.1, 2.2, 2.3, 2.4])

		self.assertGreater(monitoring.psi(expected, actual, bins=5), 0.0)


if __name__ == "__main__":
	unittest.main()
