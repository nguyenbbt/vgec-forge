import json
import sys
import tempfile
import unittest
from pathlib import Path

from vgec.evaluation import evaluate, macro_f1, mine_hard_examples


def gold_row(i, label=2):
    source = f"Mẫu lỗi {i}."
    return {"ID": i, "Input": source, "Output": {
        "Error_Type_Id": label, "Fixed Sentence": [source if label == 0 else f"Mẫu đã sửa {i}."]}}


class EvaluationTests(unittest.TestCase):
    def test_macro_f1_uses_fixed_label_universe_and_confusion(self):
        score = macro_f1([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
        self.assertEqual(score["macro_f1"], 1.0)
        self.assertEqual(macro_f1([2], [2])["macro_f1"], 0.2)
        self.assertEqual(macro_f1([0, 0, 1], [0, 1, 1])["confusion_matrix"][0], [1, 1, 0, 0, 0])

    def test_alignment_by_id_and_missing_prediction_fails(self):
        gold = [gold_row(1, 0), gold_row(2)]
        predictions = [{"ID": 2, "label": 2}, {"ID": 1, "label": 0}]
        self.assertEqual(evaluate(gold, predictions)["classifier"]["macro_f1"], 0.4)
        with self.assertRaises(ValueError):
            evaluate(gold, predictions[:1])
        with self.assertRaises(ValueError):
            evaluate(gold, predictions + predictions[:1])

    def test_gec_is_not_claimed_without_official_scorer(self):
        report = evaluate([gold_row(1)], [{"ID": 1, "label": 2, "fixed": "Mẫu đã sửa 1."}])
        self.assertIsNone(report["gec_official"])

    def test_external_scorer_receives_gold_order_and_no_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "scorer.py"
            script.write_text('import json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()\nassert p == ["first", "second"]\nprint(json.dumps({"F0.5": 0.75}))\n', encoding="utf-8")
            reference = root / "gold.m2"
            reference.write_text("native benchmark format", encoding="utf-8")
            scorer = {"name": "fixture scorer", "metric_key": "F0.5", "gold_reference": str(reference),
                      "command": [sys.executable, str(script), "{predictions}", "{gold}"]}
            report = evaluate([gold_row(1), gold_row(2)], [
                {"ID": 2, "label": 2, "fixed": "second"}, {"ID": 1, "label": 2, "fixed": "first"}], scorer)
            self.assertEqual(report["gec_official"]["metrics"]["F0.5"], 0.75)

    def test_mining_keeps_gold_and_requires_review(self):
        rows = mine_hard_examples([gold_row(1)], [{"ID": 1, "label": 3, "confidence": 0.98}])
        self.assertEqual(rows[0]["record"]["Output"]["Error_Type_Id"], 2)
        self.assertFalse(rows[0]["human_verified"])
        self.assertEqual(rows[0]["split"], "train")

    def test_mining_rejects_holdout_contamination(self):
        with self.assertRaises(ValueError):
            mine_hard_examples([gold_row(1)], [{"ID": 1, "label": 3}], holdout=[gold_row(1)])
