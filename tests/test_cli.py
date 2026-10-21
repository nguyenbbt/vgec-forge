import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-m", "vgec", *args], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", timeout=30)

    def test_help_and_doctor_work_without_ml_dependencies(self):
        self.assertEqual(self.invoke("--help").returncode, 0)
        result = self.invoke("doctor", "--config", "configs/api.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["generator"], "Qwen/Qwen3.5-4B")

    def test_actual_http_pipeline_and_id_contract(self):
        good = "Người thợ mang một chiếc ghế gỗ ra hiên."
        bad = "Người thợ mang một chiếc gỗ ghế ra hiên."
        responses = iter([
            {"sentence": good},
            {"input": bad, "error_span": "gỗ ghế", "fixed_span": "ghế gỗ", "explanation": "Sai vị trí chất liệu."},
            {"label": 2, "confidence": 0.99, "reason": "Cấu tạo cụm sai."},
            {"label": 0, "confidence": 0.99, "reason": "Câu đúng."},
            {"grammatical_fixed": True, "meaning_preserved": True, "minimal": True, "phrase_only": True,
             "function_only": False, "punctuation_only": False, "natural_error": True, "error_count": 1, "reason": "Một lỗi cụm."},
        ])
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                    "content": json.dumps(next(responses), ensure_ascii=False)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                url = f"http://127.0.0.1:{server.server_port}/v1"
                config = {"run_dir": "run", "count": 1, "round_size": 1, "max_rounds": 1,
                          "generator": {"base_url": url}, "verifier": {"base_url": url}}
                path = root / "config.json"
                path.write_text(json.dumps(config), encoding="utf-8")
                result = self.invoke("run", "--config", str(path))
                self.assertEqual(result.returncode, 0, result.stderr)
                output = root / "run/dataset.jsonl"
                checked = self.invoke("validate", str(output), "--count", "1")
                self.assertEqual(checked.returncode, 0, checked.stderr)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["Output"]["Fixed Sentence"], [good])
                self.assertEqual(len(received), 5)
                self.assertEqual(received[2]["model"], "Qwen/Qwen3-4B-Instruct-2507")
                payload = json.loads(received[2]["messages"][1]["content"].split("\n", 1)[1])
                self.assertNotIn("proposed_correction", payload)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_validate_incomplete_dataset_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            self.assertNotEqual(self.invoke("validate", str(path)).returncode, 0)
