#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "tools/measure_lail_prose.py"
    spec = importlib.util.spec_from_file_location("measure_lail_prose", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LailProseHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load()

    def test_frozen_protocol_is_512_tokens_temperature_0_2(self) -> None:
        self.assertEqual(self.mod.MAX_TOKENS, 512)
        self.assertEqual(self.mod.TEMPERATURE, 0.2)
        body = self.mod.completion_body("deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["min_tokens"], 512)
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(body["temperature"], 0.2)
        self.assertTrue(body["stream"])
        kwargs = body["chat_template_kwargs"]
        self.assertFalse(kwargs["thinking"])
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_lail_tok_s_is_completion_tokens_over_post_ttft_wall(self) -> None:
        # 512 completion tokens over 16 s of post-TTFT wall → 32 tok/s.
        self.assertAlmostEqual(self.mod.lail_decode_tok_s(512, 16.0), 32.0)
        self.assertEqual(self.mod.lail_decode_tok_s(512, 0.0), 0.0)
        self.assertEqual(self.mod.lail_decode_tok_s(0, 16.0), 0.0)

    def test_stream_one_fixture_matches_lail_arithmetic(self) -> None:
        decode_s = 0.2
        first_delay = 0.05
        payload = {
            "max_tokens": self.mod.MAX_TOKENS,
            "min_tokens": self.mod.MAX_TOKENS,
            "temperature": self.mod.TEMPERATURE,
            "ignore_eos": True,
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A003
                return

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode())
                for key, value in payload.items():
                    if req.get(key) != value:
                        self.send_error(400, f"bad {key}")
                        return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"A"}}]}\n\n')
                self.wfile.flush()
                threading.Event().wait(decode_s)
                usage = {
                    "choices": [{"delta": {}}],
                    "usage": {
                        "prompt_tokens": 40,
                        "completion_tokens": 512,
                    },
                }
                self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
            row = self.mod.stream_one(url, "deepseek-ai/DeepSeek-V4.1-Flash")
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(row["completion_tokens"], 512)
        self.assertAlmostEqual(row["lail_tok_s"], 512 / row["decode_s"], places=6)
        self.assertGreater(row["decode_s"], decode_s - 1.0)
        self.assertLess(row["ttft_s"], first_delay + 1.0)
        self.assertEqual(
            row["lail_tok_s"],
            self.mod.lail_decode_tok_s(row["completion_tokens"], row["decode_s"]),
        )
        self.assertFalse(row["collapsed"])
        self.assertEqual(row["preview"], "A")

    def test_collapse_sample_is_detected(self) -> None:
        sample = (ROOT / "tests/fixtures/prose-collapse.txt").read_text()
        self.assertLess(self.mod.prose_type_token_ratio(sample), 0.12)
        self.assertLess(self.mod.prose_ngram_diversity(sample), 0.12)
        self.assertTrue(self.mod.prose_collapsed(sample))
        essay = (
            "The KV cache is the product, not a leftover after util. "
            "Unified memory on a Spark changes how a long system prompt "
            "is shared across tabs. Decode throughput and time-to-first-token "
            "are different numbers, and the essay has to keep moving through "
            "new sentences instead of repeating the same nine words. "
            "Batch size, expert count, and the captured graph all change "
            "the wall time in ways a looping prompt tail cannot."
        )
        self.assertFalse(self.mod.prose_collapsed(essay))
        self.assertGreater(self.mod.prose_type_token_ratio(essay), 0.12)


if __name__ == "__main__":
    unittest.main()
