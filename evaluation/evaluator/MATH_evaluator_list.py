class MATHEvaluator:
    def __init__(self):
        try:
            from symeval import EvaluatorMathBatch
        except ImportError:
            try:
                from math_verify import parse, verify
            except ImportError as exc:
                raise RuntimeError(
                    "Math scoring requires either the original `symeval` package or the "
                    "maintained `math-verify` package (`pip install -r requirements-eval.txt`). "
                    "Generation results are checkpointed; install one evaluator and rerun "
                    "the same command to score/resume."
                ) from exc
            self.backend = "math_verify"
            self.parse = parse
            self.verify = verify
            self.evaluator = None
        else:
            self.backend = "symeval"
            self.evaluator = EvaluatorMathBatch()

    def extract_answer_math(self, s):
        ans = s.split("boxed")
        if len(ans) == 1:
            return s
        ans = ans[-1]
        if len(ans) == 0:
            return ""
        try:
            if ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
        except (IndexError, TypeError):
            return ""
        return a

    def score(self, pred_ans, real_ans):
        if self.backend == "math_verify":
            scores = []
            for prediction, reference in zip(pred_ans, real_ans):
                try:
                    scores.append(bool(self.verify(self.parse(reference), self.parse(prediction))))
                except Exception:
                    scores.append(False)
            return scores
        answers = [self.extract_answer_math(a) for a in real_ans]
        preds = [self.extract_answer_math(a) for a in pred_ans]
        scores = self.evaluator.batch_eq(ref_answers=answers, pred_answers=preds)
        return scores
