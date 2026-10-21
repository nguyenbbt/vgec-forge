import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('download_runtime', Path(__file__).resolve().parents[1]/'scripts/download-runtime.py')
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class SetupTests(unittest.TestCase):
    def exercise(self, initial, downloaded):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        wheel = root/'tmp/wheels/fake-1-py3-none-any.whl'
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(initial)
        info = {'urls': [{'filename': wheel.name, 'url': 'https://example.invalid/wheel',
                          'size': 4, 'digests': {'sha256': hashlib.sha256(b'good').hexdigest()}}]}
        commands = []

        def command(args, **kwargs):
            if '/json' in args[-1]:
                return SimpleNamespace(stdout=json.dumps(info))
            commands.append(args)
            wheel.write_bytes(downloaded)
            return SimpleNamespace(returncode=0)

        return root, wheel, commands, command

    def test_complete_corrupt_wheel_restarts_instead_of_resuming(self):
        root, wheel, commands, command = self.exercise(b'evil', b'good')
        with patch.object(runtime, 'ROOT', root), patch.object(runtime.subprocess, 'run', side_effect=command):
            runtime.download(('fake', '1'))
        self.assertNotIn('-C', commands[0])
        self.assertEqual(wheel.read_bytes(), b'good')
        self.assertTrue(list(wheel.parent.glob('*.corrupt-*')))

    def test_checksum_failure_quarantines_partial_corruption_for_next_retry(self):
        root, wheel, commands, command = self.exercise(b'ev', b'evil')
        with patch.object(runtime, 'ROOT', root), patch.object(runtime.subprocess, 'run', side_effect=command):
            with self.assertRaisesRegex(ValueError, 'Checksum'):
                runtime.download(('fake', '1'))
        self.assertIn('-C', commands[0])
        self.assertFalse(wheel.exists())
