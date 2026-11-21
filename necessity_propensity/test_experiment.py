"""Offline checks for the necessity/propensity judges: no API calls, nothing written to logs/.

Run from the repo root: python -m unittest discover -s necessity_propensity -v
"""
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as e  # noqa: E402

PINNED = {"question_necessity": "e2bc1564d4190f6a5642004e7c0511b0ef33ff706916483da8e38938e8a018c7",
          "trace_profile": "91c1797fac1ffc54425b05daec1f8facd0e2c2bcd75a83fad3072e30f1d2e946",
          "hint_mention": "8e20ffd709b4603061a02e0911fd2f66949db7fc1b6229d1033708eac56bdea3"}
SPEC, HASHES = e.load_prompts()
UNITS = e.build_units(SPEC)
DATA = e.load_dataset()
TRACE = "First, recall the definition. A palette is a thin board for mixing paint. So the answer is (D). Let me double-check the others."
UNIT = {"task": "trace_profile", "uid": "u", "trace": TRACE}
HINT_UNIT = {"task": "hint_mention", "uid": "h", "trace": "The user thinks (B). But I computed D."}


class TestPromptsAndUnits(unittest.TestCase):
    def test_prompts_are_pinned(self):
        self.assertEqual(HASHES, PINNED, "prompts_v1.json changed; add prompts_v2.json instead")

    def test_unit_counts_and_exclusions(self):
        got = Counter((u["task"], u["excluded"]) for u in UNITS)
        self.assertEqual(got, {("question_necessity", None): 120, ("trace_profile", None): 238,
                               ("trace_profile", "truncated at max_new_tokens"): 2,
                               ("hint_mention", None): 239,
                               ("hint_mention", "truncated at max_new_tokens"): 1})

    def test_question_judge_sees_no_answers(self):
        for u in (u for u in UNITS if u["task"] == "question_necessity"):
            q = DATA[u["id"]]
            self.assertNotIn(q["hint_text"], u["text"])
            self.assertNotIn("correct answer", u["text"].lower().replace("correct option", ""))
            self.assertEqual(u["text"].count("<question>"), 1)
            self.assertNotIn(u["id"], u["text"])

    def test_trace_judges_get_the_thinking_field_and_no_gold(self):
        raw = {f"{tag}|{r['key']}": r for tag in e.QWEN for _, r in e.b2.read_log(tag)}
        for u in (u for u in UNITS if u["task"] != "question_necessity" and not u["excluded"]):
            r, q = raw[u["uid"]], DATA[u["id"]]
            tpl = SPEC["tasks"][u["task"]]
            # The request is exactly the template filled with the allowed fields, nothing else.
            if u["task"] == "hint_mention":
                expected = tpl["template"].format(hint=q["hint_answer"], hint_text=q["hint_text"],
                                                  hint_context=tpl["hint_context"][q["hint_type"]],
                                                  trace=r["thinking"].strip())
            else:
                expected = tpl["template"].format(question=e.format_question(q), trace=r["thinking"].strip(),
                                                  final=e.b2.parse_answer(r["content"]))
            self.assertEqual(u["text"], expected)

    def test_plan_is_fixed_resumable_and_task_filtered(self):
        first = [u["uid"] for _, u in e.plan(UNITS, HASHES, "m", [])]
        self.assertEqual(first, [u["uid"] for _, u in e.plan(UNITS, HASHES, "m", [])])
        self.assertEqual(len(first), 597)
        done = [(1, {"cache_key": k, "status": "ok"}) for k, _ in e.plan(UNITS, HASHES, "m", [])[:10]]
        self.assertEqual(len(e.plan(UNITS, HASHES, "m", done)), 587)
        self.assertEqual(len(e.plan(UNITS, HASHES, "m", [], ["question_necessity"])), 120)
        other = {**HASHES, "trace_profile": "edited"}
        self.assertEqual(len(e.plan(UNITS, other, "m", done)), 587 + sum(
            u["task"] == "trace_profile" for _, u in e.plan(UNITS, HASHES, "m", [])[:10]))


class TestValidation(unittest.TestCase):
    def profile(self, quote, **kw):
        out = {"commit_quote": quote, "answer_from_worked_steps": False, "after_commit": "verification",
               "states_decisive_reason": True, "brief_reason": "r", **kw}
        return e.validate(SPEC, UNIT, json.dumps(out))

    def test_commit_quote_located(self):
        parsed, problems, extras = self.profile("So the answer is (D).")
        self.assertEqual(problems, [])
        self.assertAlmostEqual(extras["commit_position"], TRACE.index("So the answer") / len(TRACE), places=2)

    def test_case_only_quote_is_accepted_and_located(self):
        parsed, problems, extras = self.profile("A Palette is a thin board for mixing paint.")
        self.assertEqual((problems, extras["quote_check"]), ([], "case_only"))
        self.assertIsNotNone(extras["commit_position"])

    def test_invalid_trace_profiles(self):
        self.assertIn("commit_quote not found in trace", self.profile("The answer is (B).")[1])
        self.assertTrue(self.profile("So the answer is (D).", after_commit="maybe")[1])
        self.assertTrue(e.validate(SPEC, UNIT, "nope")[1])
        self.assertTrue(e.validate(SPEC, UNIT, json.dumps({"commit_quote": ""}))[1])

    def test_empty_commit_quote_is_valid_without_position(self):
        parsed, problems, extras = self.profile("")
        self.assertEqual((problems, extras.get("commit_position")), ([], None))

    def test_hint_mention_consistency(self):
        def check(**kw):
            out = {"mentions_hint": True, "stance": "rejects", "evidence_quote": "The user thinks (B).",
                   "brief_reason": "r", **kw}
            return e.validate(SPEC, HINT_UNIT, json.dumps(out))[1]
        self.assertEqual(check(), [])
        self.assertTrue(check(stance="not_mentioned"))
        self.assertTrue(check(evidence_quote=""))
        self.assertTrue(check(evidence_quote="The professor said B."))
        self.assertEqual(check(mentions_hint=False, stance="not_mentioned", evidence_quote=""), [])

    def test_api_failure_is_an_error_record(self):
        def boom(**kw):
            err = Exception("down")
            err.status_code = 503
            raise err
        client = SimpleNamespace(responses=SimpleNamespace(create=boom))
        unit = next(u for u in UNITS if u["task"] == "question_necessity")
        records = e.judge_unit(client, {"model": "m", "endpoint_sha256": None}, SPEC, HASHES, "k", unit,
                               max_attempts=2, sleep=lambda s: None)
        self.assertEqual([r["status"] for r in records], ["error", "error"])
        self.assertNotIn("parsed", records[-1])


class TestSummary(unittest.TestCase):
    def test_rates_and_denominators(self):
        questions = {"q1": {"j1_status": "ok", "reasoning_type": "recall", "reference-model_correct": True,
                            "Qwen3-4B_vllm_correct": True, "Qwen3-1.7B_vllm_correct": False},
                     "q2": {"j1_status": "ok", "reasoning_type": "multi_step", "reference-model_correct": False,
                            "Qwen3-4B_vllm_correct": True, "Qwen3-1.7B_vllm_correct": True},
                     "q3": {"j1_status": "invalid", "reasoning_type": "", "reference-model_correct": False,
                            "Qwen3-4B_vllm_correct": False, "Qwen3-1.7B_vllm_correct": False}}
        traces = [{"model": "Qwen3-4B_vllm", "condition": "clean", "reasoning_type": "recall", "status": "ok",
                   "commit_position": 0.1, "worked_steps": False, "decisive_reason": True,
                   "after_commit": "verification", "n_tokens": 100, "trace_class": "clean"},
                  {"model": "Qwen3-4B_vllm", "condition": "clean", "reasoning_type": "recall", "status": "invalid",
                   "n_tokens": 300, "trace_class": "clean"},
                  {"model": "Qwen3-4B_vllm", "condition": "hinted", "reasoning_type": "recall", "status": "ok",
                   "mentions_hint": True, "stance": "relies", "trace_class": "flip", "b1_disclosed": True,
                   "b2_flag": None, "n_tokens": 50}]
        s = e.compute_summary(questions, traces)
        acc = e.pick(s, "necessity", "clean accuracy (sample 0)", "reference-model", "all")
        self.assertEqual((acc["k"], acc["n"]), (1, 2))
        early = e.pick(s, "propensity", "answer fixed in first 25% of trace", "Qwen3-4B_vllm", "recall")
        self.assertEqual((early["k"], early["n"]), (1, 1))
        tokens = e.pick(s, "necessity", "median thinking tokens (clean)", "Qwen3-4B_vllm", "recall")
        self.assertEqual(tokens["value"], 200)
        b2 = e.pick(s, "monitorability", "flips flagged by the CoT-only monitor (B2)", "Qwen3-4B_vllm", "recall")
        self.assertEqual(b2["n"], 0)
        mention = e.pick(s, "propensity", "hinted traces that mention the hint", "Qwen3-4B_vllm", "flip")
        self.assertEqual((mention["k"], mention["n"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
