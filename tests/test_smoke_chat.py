#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    payload = {"choices": [{"message": {"content": "323"}}]}

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def do_POST(self) -> None:  # noqa: N802
        raw = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _serve(payload: dict) -> tuple[HTTPServer, str]:
    _Handler.payload = payload
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}/v1/chat/completions"


class SmokeChatTests(unittest.TestCase):
    def test_cli_writes_json_out_and_prints_content(self) -> None:
        httpd, url = _serve({"choices": [{"message": {"content": "323"}}]})
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "smoke.json"
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "smoke_chat.py"),
                        "--url",
                        url,
                        "--json-out",
                        str(out),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(proc.stdout.strip(), "323")
                payload = json.loads(out.read_text())
                self.assertEqual(
                    payload["choices"][0]["message"]["content"].strip(),
                    proc.stdout.strip(),
                )
        finally:
            httpd.shutdown()

    def test_cli_fails_on_empty_thinking_content(self) -> None:
        httpd, url = _serve({"choices": [{"message": {"content": "   "}}]})
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "smoke_chat.py"), "--url", url],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("empty content", proc.stderr)
        finally:
            httpd.shutdown()
