from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "run_sglang.py"
SPEC = importlib.util.spec_from_file_location("easyep_run_sglang", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_sglang = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_sglang)


class EvaluationClientTests(unittest.TestCase):
    def test_base_url_normalization(self):
        self.assertEqual(run_sglang.normalize_base_url("http://127.0.0.1:60000"), "http://127.0.0.1:60000/v1")
        self.assertEqual(run_sglang.normalize_base_url("http://host/v1/"), "http://host/v1")

    def test_duplicate_predictions_keep_independent_correctness(self):
        records = [
            {"job_id": "0:0", "output": "\\boxed{1}", "prediction": [{"solution": "same"}]},
            {"job_id": "0:1", "output": "\\boxed{2}", "prediction": [{"solution": "same"}]},
        ]

        class FakeEvaluator:
            def score(self, predictions, references):
                return [True, False]

        original = run_sglang.MATHEvaluator
        run_sglang.MATHEvaluator = FakeEvaluator
        try:
            scored, summary = run_sglang.score_results(records)
        finally:
            run_sglang.MATHEvaluator = original
        self.assertTrue(scored[0]["prediction"][0]["correctness"])
        self.assertFalse(scored[1]["prediction"][0]["correctness"])
        self.assertEqual(summary["correct"], 1)

    def test_resume_reader_ignores_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            path.write_text(
                '{"type":"sample","job_id":"0:0"}\n'
                '{"type":"summary","total":1}\n',
                encoding="utf-8",
            )
            self.assertEqual(set(run_sglang.read_completed(path)), {"0:0"})


if __name__ == "__main__":
    unittest.main()
