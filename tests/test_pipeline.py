import json
import tempfile
import unittest
from pathlib import Path

from vgec.config import validate_config
from vgec.pipeline import Pipeline, validate_candidate
from vgec.prompts import EXAMPLES, classify_prompt


GOOD = "Bác thợ đặt một chiếc ghế gỗ cạnh hiên."
BAD = "Bác thợ đặt một chiếc gỗ ghế cạnh hiên."


def candidate():
    return {"input": BAD, "error_span": "gỗ ghế", "fixed_span": "ghế gỗ", "explanation": "Sai vị trí chất liệu."}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, *args, **kwargs):
        return next(self.responses)

    def close(self):
        pass


def verdict(label):
    return {"label": label, "confidence": 0.95, "reason": "Bằng chứng ngữ pháp."}


AUDIT = {"grammatical_fixed": True, "meaning_preserved": True, "minimal": True,
         "phrase_only": True, "function_only": False, "punctuation_only": False,
         "natural_error": True, "error_count": 1, "reason": "Lỗi nội bộ cụm."}


class PipelineTests(unittest.TestCase):
    def config(self, directory, **changes):
        return validate_config({"run_dir": directory, "count": 1, "round_size": 1, "max_rounds": 2, **changes})

    def test_blind_verifier_never_receives_candidate_answer(self):
        payload = json.loads(classify_prompt("Câu bí mật.")[1]["content"].split("\n", 1)[1])
        self.assertEqual(set(payload), {"Input", "boundary_examples"})
        self.assertEqual(payload["Input"], "Câu bí mật.")

    def test_verifier_does_not_receive_its_input_as_labeled_fewshot(self):
        example = EXAMPLES[3]
        payload = json.loads(classify_prompt(example["Input"])[1]["content"].split("\n", 1)[1])
        self.assertNotIn(example, payload["boundary_examples"])

    def test_holdout_fewshot_is_removed_from_generation_context(self):
        with tempfile.TemporaryDirectory() as directory:
            example = EXAMPLES[3]
            path = Path(directory) / "dev.jsonl"
            path.write_text(json.dumps({"ID": 1, "Input": example["Input"], "Output": {
                "Error_Type_Id": 2, "Fixed Sentence": [example["fixed"]]}}), encoding="utf-8")
            pipeline = Pipeline(self.config(str(Path(directory) / "run"), holdout_paths=[str(path)]))
            self.assertNotIn(example, pipeline.examples)

    def test_rejects_nonlocal_and_function_only_repairs(self):
        bad = candidate()
        bad["fixed_span"] = "bàn nhựa"
        with self.assertRaises(ValueError):
            validate_candidate(bad, GOOD)
        with self.assertRaises(ValueError):
            validate_candidate({"input": "Cuốn sách thuộc của tôi.", "error_span": "thuộc của",
                                "fixed_span": "thuộc", "explanation": "thừa của"}, "Cuốn sách thuộc tôi.")

    def test_full_generate_verify_export_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(directory)
            models = iter([ScriptedModel([{"sentence": GOOD}, candidate()]),
                           ScriptedModel([verdict(2), verdict(0), AUDIT])])
            pipeline = Pipeline(config, factory=lambda _: next(models))
            pipeline.run()
            rows = [json.loads(line) for line in (Path(directory) / "dataset.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["Output"], {"Error_Type_Id": 2, "Fixed Sentence": [GOOD]})
            self.assertEqual(rows[0]["ID"], 120001)
            Pipeline(config, factory=lambda _: self.fail("Resume must not call models")).run()

    def test_wrong_label_is_quarantined_and_short_export_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            models = iter([ScriptedModel([{"sentence": GOOD}, candidate()]), ScriptedModel([verdict(3)])])
            pipeline = Pipeline(self.config(directory), factory=lambda _: next(models))
            pipeline.generate_round()
            pipeline.verify()
            with self.assertRaises(ValueError):
                pipeline.export()
            self.assertFalse((Path(directory) / "dataset.jsonl").exists())
            self.assertEqual(pipeline.state["decisions"][0]["status"], "rejected")

    def test_changed_config_cannot_resume_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            Pipeline(self.config(directory))
            with self.assertRaises(ValueError):
                Pipeline(self.config(directory, min_confidence=0.7))

    def test_orphan_final_json_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "dataset.json"
            target.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                Pipeline(self.config(directory))
            self.assertEqual(target.read_text(), "[]")

    def test_export_checks_all_artifacts_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(self.config(directory))
            pipeline.state["candidates"] = [{"cid": "a", "input": BAD, "fixed": GOOD, "subtype": "noun_order"}]
            pipeline.state["decisions"] = [{"cid": "a", "status": "accepted"}]
            audit_path = Path(directory) / "audit.jsonl"
            audit_path.write_text('{"user_modified":true}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                pipeline.export()
            self.assertFalse((Path(directory) / "dataset.jsonl").exists())

    def test_verifier_cannot_equal_generator(self):
        with self.assertRaises(ValueError):
            validate_config({"verifier": {"model": "Qwen/Qwen3.5-4B"}})

    def test_short_seed_corpus_is_verified_before_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_path = Path(directory) / "seeds.jsonl"
            seed_path.write_text(json.dumps({"sentence": GOOD}), encoding="utf-8")
            config = self.config(str(Path(directory) / "run"), seed_path=str(seed_path), round_size=60)
            models = iter([ScriptedModel([candidate()]), ScriptedModel([verdict(2), verdict(0), AUDIT])])
            pipeline = Pipeline(config, factory=lambda _: next(models))
            self.assertTrue(pipeline.run().exists())

    def test_insufficient_seed_corpus_stops_without_looping(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_path = Path(directory) / "seeds.jsonl"
            seed_path.write_text(json.dumps({"sentence": GOOD}), encoding="utf-8")
            config = self.config(str(Path(directory) / "run"), seed_path=str(seed_path), round_size=60, count=2)
            models = iter([ScriptedModel([candidate()]), ScriptedModel([verdict(2), verdict(0), AUDIT])])
            pipeline = Pipeline(config, factory=lambda _: next(models))
            with self.assertRaisesRegex(ValueError, "Seed corpus exhausted"):
                pipeline.run()
            self.assertEqual(len(pipeline.accepted()), 1)

    def test_low_confidence_and_function_only_audit_never_pass(self):
        for responses in ([{**verdict(2), "confidence": 0.4}],
                          [verdict(2), verdict(0), {**AUDIT, "function_only": True}]):
            with self.subTest(responses=responses), tempfile.TemporaryDirectory() as directory:
                models = iter([ScriptedModel([{"sentence": GOOD}, candidate()]), ScriptedModel(responses)])
                pipeline = Pipeline(self.config(directory), factory=lambda _: next(models))
                pipeline.generate_round()
                pipeline.verify()
                self.assertEqual(pipeline.accepted(), [])

    def test_300_record_export_has_exact_schema_ids_and_single_corrections(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline(self.config(directory, count=300))
            # Synthetic fixtures test the export contract, not language quality.
            for i in range(300):
                pipeline.state["candidates"].append({"cid": str(i), "input": f"fixture input {i}",
                                                       "fixed": f"fixture fixed {i}", "subtype": "noun_order"})
                pipeline.state["decisions"].append({"cid": str(i), "status": "accepted"})
            rows = [json.loads(line) for line in pipeline.export().read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["ID"] for r in rows], list(range(120001, 120301)))
            self.assertTrue(all(set(r) == {"ID", "Input", "Output"} for r in rows))
            self.assertTrue(all(type(r["Output"]["Error_Type_Id"]) is int and
                                len(r["Output"]["Fixed Sentence"]) == 1 for r in rows))

    def test_holdout_sentence_never_enters_candidate_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            heldout = Path(directory) / "dev.jsonl"
            heldout.write_text(json.dumps({"ID": 5, "Input": GOOD, "Output": {
                "Error_Type_Id": 0, "Fixed Sentence": [GOOD]}}, ensure_ascii=False), encoding="utf-8")
            pipeline = Pipeline(self.config(str(Path(directory) / "run"), holdout_paths=[str(heldout)]),
                                factory=lambda _: ScriptedModel([{"sentence": GOOD}]))
            pipeline.generate_round()
            self.assertEqual(pipeline.state["candidates"], [])
