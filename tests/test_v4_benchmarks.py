from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "run_v4_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("easyep_v4_benchmarks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmarks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmarks
SPEC.loader.exec_module(benchmarks)


class V4BenchmarkTests(unittest.TestCase):
    def test_gpqa_option_extraction(self):
        self.assertEqual(benchmarks.last_option(r"Reasoning\n\boxed{C}"), "C")
        self.assertEqual(benchmarks.last_option("The answer is B."), "B")
        self.assertIsNone(benchmarks.last_option("No option was selected."))

    def test_agent_os_action_extraction(self):
        self.assertEqual(benchmarks.last_action("Think: x\nAct: bash\n```bash\nls\n```"), "bash")
        self.assertEqual(benchmarks.last_action("Act: answer(42)"), "answer")
        self.assertIsNone(benchmarks.last_action("no action"))

    def test_livecodebench_sample_score(self):
        sample = benchmarks.Sample(
            prompt="", reference=None,
            metadata={"scoring": "python_samples", "platform": "atcoder", "samples": [("2\n", "4")]},
        )
        correct, detail = benchmarks.score_sample(
            "kuvecodebench", sample, "```python\nprint(int(input()) * 2)\n```", sys.executable, 3.0,
        )
        self.assertTrue(correct)
        self.assertEqual(detail["samples"][0]["actual"], "4")

    def test_livecodebench_sample_output_excludes_explanation(self):
        question = """
Sample Input 1

0 0
4 0
0 3

Sample Output 1

Yes

The triangle ABC is a right triangle.
"""
        self.assertEqual(
            benchmarks.extract_samples(question),
            [("0 0\n4 0\n0 3\n", "Yes")],
        )

    def test_livecodebench_multiline_output_excludes_explanation(self):
        question = """
Sample Input 1

1

Sample Output 1

7
3
13

Let us explain the first query.
"""
        self.assertEqual(benchmarks.extract_samples(question), [("1\n", "7\n3\n13")])

    def test_livecodebench_unscored_without_runnable_cases(self):
        sample = benchmarks.Sample(
            prompt="", reference=None,
            metadata={"scoring": "python_samples", "platform": "leetcode", "samples": []},
        )
        correct, detail = benchmarks.score_sample("kuvecodebench", sample, "```python\npass\n```", sys.executable, 3.0)
        self.assertIsNone(correct)
        self.assertEqual(detail["reason"], "no_runnable_leetcode_examples")

    def test_leetcode_method_samples_and_score(self):
        starter = "class Solution:\n    def add(self, nums: List[int], amount: int) -> int:\n        pass\n"
        question = """
Example 1:

Input: nums = [2,3], amount = 4
Output: 9
"""
        method, cases = benchmarks.extract_leetcode_samples(question, starter)
        self.assertEqual(method, "add")
        self.assertEqual(cases, [([[2, 3], 4], 9)])
        sample = benchmarks.Sample(
            prompt="", reference=None,
            metadata={"platform": "leetcode", "method": method, "leetcode_cases": cases},
        )
        correct, detail = benchmarks.score_sample(
            "kuvecodebench", sample,
            "```python\nclass Solution:\n    def add(self, nums, amount):\n        return sum(nums) + amount\n```",
            sys.executable, 3.0,
        )
        self.assertTrue(correct)
        self.assertEqual(detail["examples"][0]["actual"], 9)

    def test_local_arrow_adapters_have_expected_sizes(self):
        expected = {"agent_os": 26, "gpqa": 249, "kuvecodebench": 101}
        for name, count in expected.items():
            samples = benchmarks.build_samples(name, benchmarks.load_arrow_rows(benchmarks.DATASETS[name]))
            self.assertEqual(len(samples), count)
        self.assertEqual(
            benchmarks.build_samples("gpqa", benchmarks.load_arrow_rows(benchmarks.DATASETS["gpqa"]))[0].reference,
            "A",
        )
        kuve = benchmarks.build_samples("kuvecodebench", benchmarks.load_arrow_rows(benchmarks.DATASETS["kuvecodebench"]))
        self.assertEqual(sum(sample.metadata["platform"] == "leetcode" for sample in kuve), 48)
        self.assertTrue(all(sample.metadata["leetcode_cases"] for sample in kuve if sample.metadata["platform"] == "leetcode"))


if __name__ == "__main__":
    unittest.main()
