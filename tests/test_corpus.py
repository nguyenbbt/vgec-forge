import json
import tempfile
import unittest
from pathlib import Path

from vgec.data import read_records, references, validate_gold_record, validate_record
from vgec.evaluation import evaluate, assert_no_overlap
from vgec.pipeline import Pipeline
from vgec.config import validate_config


class CorpusTests(unittest.TestCase):
    def test_json_array_preserves_empty_and_multiple_references(self):
        rows = [
            {"ID": 1, "Input": " Tôi về... ", "Output": {"Error_Type_Id": 0, "Fixed Sentence": []}},
            {"ID": 2, "Input": "Câu lỗi.", "Output": {"Error_Type_Id": 2, "Fixed Sentence": ["Đáp án một.", "Đáp án hai."]}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8-sig")
            loaded = read_records(path)
        self.assertEqual(loaded, rows)
        self.assertEqual(references(loaded[0]), [" Tôi về... "])
        self.assertEqual(len(references(loaded[1])), 2)
        for row in rows:
            validate_gold_record(row)
            with self.assertRaises(ValueError):
                validate_record(row)
        self.assertEqual(evaluate(rows, [{"ID": 1, "label": 0}, {"ID": 2, "label": 2}])["count"], 2)

    def test_holdout_blocks_all_references_not_only_first(self):
        holdout = {"ID": 1, "Input": "Câu nguồn hoàn toàn khác.", "Output": {
            "Error_Type_Id": 2, "Fixed Sentence": ["Bản sửa đầu tiên.", "Bác thợ đặt chiếc ghế gỗ bên hiên."]}}
        train = {"ID": 2, "Input": "Bác thợ đặt chiếc ghế gỗ bên hiên.", "Output": {
            "Error_Type_Id": 0, "Fixed Sentence": []}}
        with self.assertRaises(ValueError):
            assert_no_overlap([train], [holdout])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holdout.json"
            path.write_text(json.dumps([holdout]), encoding="utf-8")
            pipeline = Pipeline(validate_config({"run_dir": str(Path(directory)/"run"), "holdout_paths": [str(path)]}))
            self.assertTrue(pipeline.excluded.contains(holdout["Output"]["Fixed Sentence"][1]))

    def test_error_gold_still_requires_a_reference(self):
        with self.assertRaises(ValueError):
            validate_gold_record({"ID": 1, "Input": "Câu lỗi.", "Output": {"Error_Type_Id": 2, "Fixed Sentence": []}})
