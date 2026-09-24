#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Handler(BaseHTTPRequestHandler):
    payload = {"choices": [{"message": {"content": "red"}}]}
    status = 200
    err_body = b""

    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def do_POST(self) -> None:  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.server.last_body = json.loads(raw.decode())  # type: ignore[attr-defined]
        if self.status >= 400:
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.err_body)))
            self.end_headers()
            self.wfile.write(self.err_body)
            return
        out = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def _serve(
    payload: dict, *, status: int = 200, err_body: bytes = b""
) -> tuple[HTTPServer, str]:
    _Handler.payload = payload
    _Handler.status = status
    _Handler.err_body = err_body
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    httpd.last_body = None  # type: ignore[attr-defined]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    return httpd, f"http://{host}:{port}/v1/chat/completions"


class SmokeVisionTests(unittest.TestCase):
    def test_cli_sends_image_url_and_prints_content(self) -> None:
        httpd, url = _serve({"choices": [{"message": {"content": "red"}}]})
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "smoke_vision.py"), "--url", url],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("red", proc.stdout)
            body = httpd.last_body  # type: ignore[attr-defined]
            self.assertEqual(body["model"], "deepseek-ai/DeepSeek-V4.1-Flash")
            content = body["messages"][0]["content"]
            kinds = {part["type"] for part in content}
            self.assertEqual(kinds, {"text", "image_url"})
            url_part = next(p for p in content if p["type"] == "image_url")
            self.assertTrue(
                url_part["image_url"]["url"].startswith("data:image/png;base64,")
            )
            src = (ROOT / "smoke_vision.py").read_text()
            self.assertNotIn("from PIL", src)
            self.assertNotIn("import PIL", src)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_cli_fails_on_wrong_color(self) -> None:
        httpd, url = _serve({"choices": [{"message": {"content": "Gray"}}]})
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "smoke_vision.py"), "--url", url],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("wrong_color", proc.stderr)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_fixture_is_64x64_solid_red_png(self) -> None:
        sys.path.insert(0, str(ROOT))
        import smoke_vision

        png = base64.b64decode(smoke_vision.RED_PNG_B64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        width, height, depth, ctype = struct.unpack(">IIBB", png[16:26])
        self.assertEqual((width, height, depth, ctype), (64, 64, 8, 2))
        idat_len = struct.unpack(">I", png[33:37])[0]
        self.assertEqual(png[37:41], b"IDAT")
        raw = zlib.decompress(png[41 : 41 + idat_len])
        self.assertEqual(raw, (b"\x00" + b"\xff\x00\x00" * 64) * 64)

    def test_cli_fails_on_not_multimodal(self) -> None:
        err = b'{"error":{"message":"is not a multimodal model"}}'
        httpd, url = _serve({}, status=400, err_body=err)
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "smoke_vision.py"), "--url", url],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("is not a multimodal model", proc.stderr)
            self.assertIn("not_multimodal", proc.stderr)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
