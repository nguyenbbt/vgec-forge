import json
import tempfile
import unittest
from pathlib import Path

from vgec.data import Deduplicator, read_jsonl, validate_record, write_jsonl, validate_dataset


def record(i=120001):
    return {"ID": i, "Input": "Tôi mua một chiếc gỗ bàn nhỏ.",
            "Output": {"Error_Type_Id": 2, "Fixed Sentence": ["Tôi mua một chiếc bàn gỗ nhỏ."]}}


class DataTests(unittest.TestCase):
    def test_rejects_extra_keys_and_non_integer_labels(self):
        for label in (True, "2", 2.0):
            row = record()
            row["Output"]["Error_Type_Id"] = label
            with self.assertRaises(ValueError):
                validate_record(row)
        row = record()
        row["reason"] = "extra"
        with self.assertRaises(ValueError):
            validate_record(row)

    def test_correction_cannot_be_empty_or_multiple(self):
        for fixed in ([], [""], ["a", "b"], "a"):
            row = record()
            row["Output"]["Fixed Sentence"] = fixed
            with self.assertRaises(ValueError):
                validate_record(row)

    def test_zero_preserves_exact_text(self):
        row = record()
        row["Input"] = " Tôi về...  "
        row["Output"] = {"Error_Type_Id": 0, "Fixed Sentence": ["Tôi về..."]}
        with self.assertRaises(ValueError):
            validate_record(row)
        row["Output"]["Fixed Sentence"] = [row["Input"]]
        validate_record(row)

    def test_export_ids_are_exact_and_ordered(self):
        with self.assertRaises(ValueError):
            validate_dataset([record(), record(120003)], count=2)
        with self.assertRaises(ValueError):
            validate_dataset([record()], count=300)

    def test_unicode_and_near_duplicate(self):
        dedup = Deduplicator(0.9)
        dedup.add("Tôi đọc một cuốn sách về lịch sử Việt Nam.")
        self.assertTrue(dedup.contains("  TÔI đọc một cuốn sách về lịch sử Việt Nam. "))
        self.assertTrue(dedup.contains("Tôi đọc một cuốn sách về lịch sử Việt Nam!"))
        self.assertFalse(dedup.contains("Bác sĩ kiểm tra bệnh nhân trước ca phẫu thuật."))

    def test_jsonl_roundtrip_quotes_and_unicode(self):
        row = record()
        row["Input"] += ' "trích dẫn"'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            write_jsonl(path, [row])
            self.assertEqual(read_jsonl(path), [row])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), row)


if __name__ == "__main__":
    unittest.main()
