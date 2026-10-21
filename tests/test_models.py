import json
import unittest
from unittest.mock import patch

from vgec.models import APIBackend, ModelError, parse_object, TransformersBackend
from types import SimpleNamespace
from unittest.mock import MagicMock


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        return self.payload


class ModelTests(unittest.TestCase):
    def test_accepts_json_and_fence_but_not_explanation_or_duplicate_keys(self):
        self.assertEqual(parse_object('```json\n{"a": 1}\n```'), {"a": 1})
        for value in ('explanation {"a":1}', '{"a":1,"a":2}', '[1]', '{"a":NaN}'):
            with self.assertRaises(ValueError):
                parse_object(value)

    def test_api_payload_and_content(self):
        with patch('vgec.models.urlopen', return_value=Response({"choices": [{
            "finish_reason": "stop", "message": {"content": '{"ok": true}'}}]})) as call:
            result = APIBackend({"model": "Qwen/Qwen3.5-4B", "base_url": "http://localhost:8000/v1"}).complete(
                [{"role": "user", "content": "test"}], temperature=0.7, seed=42)
        self.assertEqual(result, {"ok": True})
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["model"], "Qwen/Qwen3.5-4B")

    def test_truncated_response_is_never_accepted(self):
        with patch('vgec.models.urlopen', return_value=Response({"choices": [{
            "finish_reason": "length", "message": {"content": '{"ok": true}'}}]})):
            with self.assertRaises(ModelError):
                APIBackend({"model": "q", "base_url": "http://localhost:8000/v1", "retries": 0}).complete([])

    def test_no_plaintext_remote_key_or_endpoint_credentials(self):
        for url in ('http://remote.example/v1', 'https://user:password@example.com/v1'):
            with self.assertRaises(ValueError):
                APIBackend({"model": "q", "base_url": url})

    def test_local_checkpoint_architecture_comes_from_config(self):
        torch = MagicMock()
        hf = MagicMock()
        hf.AutoConfig.from_pretrained.return_value = SimpleNamespace(model_type="qwen3_5")
        with patch.dict('sys.modules', {'torch': torch, 'transformers': hf}):
            model = TransformersBackend({"model": "D:/models/generator", "load_in_4bit": False})
            model._load()
        hf.Qwen3_5ForConditionalGeneration.from_pretrained.assert_called_once()
        hf.AutoModelForCausalLM.from_pretrained.assert_not_called()
