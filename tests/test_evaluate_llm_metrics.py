import math
import unittest

from scripts.evaluate_llm_metrics import aggregate_results, set_metrics


class TestEvaluateLLMMetrics(unittest.TestCase):
    def test_set_metrics_exact_match(self):
        metrics = set_metrics({"a"}, {"a"})
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_set_metrics_partial_overlap(self):
        metrics = set_metrics({"a", "b"}, {"a"})
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2.0 / 3.0)

    def test_set_metrics_no_overlap(self):
        metrics = set_metrics({"a"}, {"b"})
        self.assertAlmostEqual(metrics["precision"], 0.0)
        self.assertAlmostEqual(metrics["recall"], 0.0)
        self.assertAlmostEqual(metrics["f1"], 0.0)

    def test_set_metrics_empty(self):
        metrics = set_metrics(set(), set())
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["exact_match"], 1.0)

    def test_aggregate_results_micro_macro(self):
        rows = [
            {
                "expected_labels": ["a", "b"],
                "predicted_labels": ["a"],
                "response_time_sec": 1.0,
                "error": "",
                "precision": 1.0,
                "recall": 0.5,
                "f1": 2.0 / 3.0,
                "exact_match": 0.0,
                "cpu_avg": 0.0,
                "cpu_max": 0.0,
                "ram_avg_percent": 0.0,
                "ram_max_percent": 0.0,
                "ram_max_mb": 0.0,
                "ram_avg_percent_of_16gb": 0.0,
                "ram_max_percent_of_16gb": 0.0,
            },
            {
                "expected_labels": ["b"],
                "predicted_labels": ["b"],
                "response_time_sec": 2.0,
                "error": "",
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_match": 1.0,
                "cpu_avg": 0.0,
                "cpu_max": 0.0,
                "ram_avg_percent": 0.0,
                "ram_max_percent": 0.0,
                "ram_max_mb": 0.0,
                "ram_avg_percent_of_16gb": 0.0,
                "ram_max_percent_of_16gb": 0.0,
            },
        ]
        summary = aggregate_results(rows)
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        self.assertTrue(math.isfinite(summary["micro_f1"]))
        self.assertTrue(math.isfinite(summary["macro_f1"]))


if __name__ == "__main__":
    unittest.main()
