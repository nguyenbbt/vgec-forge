import json
import unittest
from pathlib import Path

from vgec.modal_support import config_for_count, run_name


class ModalSettingsTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.base = json.loads((root / "configs/modal.json").read_text(encoding="utf-8"))

    def test_count_changes_target_round_and_checkpoint_without_mutating_base(self):
        small = config_for_count(self.base, 10)
        large = config_for_count(self.base, 75)
        self.assertEqual((small["count"], small["round_size"]), (10, 20))
        self.assertEqual((large["count"], large["round_size"]), (75, 60))
        self.assertNotEqual(small["run_dir"], large["run_dir"])
        self.assertEqual(self.base["count"], 10)

    def test_reject_invalid_count(self):
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                run_name(count)


if __name__ == "__main__":
    unittest.main()
