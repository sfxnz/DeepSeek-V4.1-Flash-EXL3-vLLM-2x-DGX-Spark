#!/usr/bin/env python3
"""Offline tests for tests/quality_eval.py: scoring, parsing, Wilson CI, gates,
vendored data and main() wiring against a canned local HTTP server."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import quality_eval as q  # noqa: E402


class PureLogicTests(unittest.TestCase):
    def test_wilson_known_values(self) -> None:
        lo, hi = q.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.27754, places=4)
        lo, hi = q.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.23659, places=4)
        self.assertAlmostEqual(hi, 0.76341, places=4)
        self.assertEqual(q.wilson(0, 0), (0.0, 1.0))
        self.assertEqual(q.rate(18, 20)["rate"], 0.9)

    def test_token_logprobs_skips_bos_and_prefix(self) -> None:
        ids = [0, 11, 12, 13]
        plp = [None, {"11": {"logprob": -1.0}},
               {"12": {"logprob": -2.0}, "99": {"logprob": -0.1}},
               {"13": {"logprob": -3.0}}]
        self.assertEqual(q.token_logprobs(plp, ids, 1), [-1.0, -2.0, -3.0])
        self.assertEqual(q.token_logprobs(plp, ids, 2), [-2.0, -3.0])

    def test_percentile(self) -> None:
        self.assertEqual(q.percentile([3, 1, 2], 0.5), 2)
        self.assertAlmostEqual(q.percentile([0, 10], 0.99), 9.9)
        self.assertEqual(q.percentile([], 0.5), 0.0)

    def test_gsm_extraction(self) -> None:
        self.assertEqual(q.extract_gsm_answer("3+4=7\nAnswer: 1,200"), "1200")
        self.assertEqual(q.extract_gsm_answer("so 18.\n**Answer:** $18"), "18")
        self.assertEqual(q.extract_gsm_answer("I get 5 then 12"), "12")
        self.assertIsNone(q.extract_gsm_answer("no digits"))
        self.assertTrue(q.gsm_correct("Answer: 18.0", "18"))
        self.assertFalse(q.gsm_correct("Answer: 17", "18"))
        self.assertTrue(q.gsm_correct("Answer: -3", "-3"))

    def test_letter_extraction(self) -> None:
        self.assertEqual(q.extract_letter("B"), "B")
        self.assertEqual(q.extract_letter("C. 24"), "C")
        self.assertEqual(q.extract_letter("The answer is D"), "D")
        self.assertIsNone(q.extract_letter("none"))

    def test_args_match(self) -> None:
        want = {"amount": 250, "from": "USD", "tags": ["a", "b"],
                "settings": {"newsletter": False}}
        self.assertTrue(q.args_match({"amount": 250.0, "from": "usd ",
                                      "tags": ["A", "b"],
                                      "settings": {"newsletter": False}}, want))
        self.assertFalse(q.args_match({"amount": 250, "from": "USD",
                                       "tags": ["b", "a"],
                                       "settings": {"newsletter": False}}, want))
        # bool must not match 0/1
        self.assertFalse(q.args_match({"amount": 250, "from": "USD", "tags": ["a", "b"],
                                       "settings": {"newsletter": 0}}, want))
        # extra key fails unless ignored
        self.assertFalse(q.args_match({"x": 1, "y": 2}, {"x": 1}))
        self.assertTrue(q.args_match({"x": 1, "y": 2}, {"x": 1}, ignore=["y"]))
        self.assertFalse(q.args_match({"x": "1"}, {"x": 1}))

    def test_score_tool_items(self) -> None:
        item = {"id": "t", "expect": {"name": "f", "args": {"a": 1}}}
        call = lambda name, args: {"tool_calls": [{"function": {"name": name, "arguments": args}}]}  # noqa: E731
        self.assertTrue(q.score_tool_item(item, call("f", '{"a": 1}'))["exact"])
        self.assertFalse(q.score_tool_item(item, call("g", '{"a": 1}'))["exact"])
        bad = q.score_tool_item(item, call("f", '{"a": 1'))
        self.assertFalse(bad["json_valid"])
        self.assertFalse(bad["exact"])
        miss = q.score_tool_item(item, {"content": "no"})
        self.assertFalse(miss["exact"])
        self.assertEqual(miss["n_calls"], 0)
        neg = {"id": "n", "expect": None}
        self.assertTrue(q.score_tool_item(neg, {"content": "Ottawa"})["no_call_ok"])
        self.assertFalse(q.score_tool_item(neg, call("f", "{}"))["no_call_ok"])
        s = q.summarize_tools([q.score_tool_item(item, call("f", '{"a": 1}')),
                               bad, miss,
                               q.score_tool_item(neg, {"content": "x"})])
        self.assertEqual((s["json_valid"]["k"], s["json_valid"]["n"]), (1, 2))
        self.assertEqual((s["exact_args"]["k"], s["exact_args"]["n"]), (1, 3))
        self.assertEqual((s["no_call"]["k"], s["no_call"]["n"]), (1, 1))

    def test_divergence_and_hazard(self) -> None:
        self.assertIsNone(q.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(q.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(q.first_divergence([1, 2], [1, 2, 3]), 2)
        h = q.hazard([([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [1, 5, 3, 4])])
        self.assertEqual(h["diverged"], 1)
        self.assertEqual(h["first_div"], [None, 1])
        self.assertAlmostEqual(h["hazard"], 1 / (4 + 2), places=5)

    def test_filler_is_deterministic_novel_text(self) -> None:
        a = q.filler_paragraphs(5, 3)
        self.assertEqual(a, q.filler_paragraphs(5, 3))
        self.assertNotEqual(a, q.filler_paragraphs(6, 3))
        self.assertNotIn("vLLM", " ".join(a))
        name, code = q.needle_code(1001)
        self.assertEqual((name, code), q.needle_code(1001))
        self.assertRegex(code, r"^[A-Z]{3}-\d{4}-[A-Z]{2}$")
        paras = [f"p{i}" for i in range(10)]
        doc = q.needle_doc(paras, 0.5, name, code, "HDR").split("\n\n")
        self.assertEqual(doc[0], "HDR")
        self.assertIn(code, doc[6])
        self.assertEqual(len(doc), 12)


def _base() -> dict:
    return {"components": {
        "nll": {"mean_nll": 0.2450, "repeat_abs_delta": 0.0025},
        "decode": {"median_abs_dlogprob": 0.015, "gen_nll_prefill": 0.30},
        "selfcons": {"aa": {"hazard": 0.017}},
        "tools": {"exact_args": q.rate(21, 22), "json_valid": q.rate(22, 22),
                  "no_call": q.rate(8, 8)},
        "gsm8k": {"acc": q.rate(90, 100)},
        "needle": {"cells": {"8192@0.1": {"found": True}, "8192@0.5": {"found": True}}},
    }}


class GateTests(unittest.TestCase):
    def _gate(self, rows, name):
        return next(r for r in rows if r["gate"] == name)

    def test_absolute_gates_without_baseline(self) -> None:
        cur = {"components": {"vision": {"pass": False, "content": "Gray"},
                              "c2": {"pass": True, "answers": ["323", "252"]},
                              "nll": {"error": "RuntimeError: HTTP 500"}}}
        rows = q.gates(cur, None)
        self.assertFalse(self._gate(rows, "vision")["pass"])
        self.assertTrue(self._gate(rows, "c2")["pass"])
        self.assertFalse(self._gate(rows, "nll.error")["pass"])
        self.assertFalse(any(r["gate"] == "nll.mean_nll" for r in rows))

    def test_nll_gate_uses_floor_and_noise(self) -> None:
        base = _base()
        ok = q.gates({"components": {"nll": {"mean_nll": 0.2549}}}, base)
        self.assertTrue(self._gate(ok, "nll.mean_nll")["pass"])
        bad = q.gates({"components": {"nll": {"mean_nll": 0.2551}}}, base)
        self.assertFalse(self._gate(bad, "nll.mean_nll")["pass"])
        base["components"]["nll"]["repeat_abs_delta"] = 0.01   # noisy baseline -> 0.03
        wide = q.gates({"components": {"nll": {"mean_nll": 0.2700}}}, base)
        self.assertTrue(self._gate(wide, "nll.mean_nll")["pass"])
        self.assertAlmostEqual(self._gate(wide, "nll.mean_nll")["limit"], 0.275)

    def test_decode_gates(self) -> None:
        cur = {"components": {"decode": {"median_abs_dlogprob": 0.06, "gen_nll_prefill": 0.35}}}
        rows = q.gates(cur, _base())
        self.assertTrue(self._gate(rows, "decode.median_abs_dlogprob")["pass"])
        self.assertTrue(self._gate(rows, "decode.gen_nll_prefill")["pass"])
        cur["components"]["decode"] = {"median_abs_dlogprob": 0.5, "gen_nll_prefill": 2.0}
        rows = q.gates(cur, _base())
        self.assertFalse(self._gate(rows, "decode.median_abs_dlogprob")["pass"])
        self.assertFalse(self._gate(rows, "decode.gen_nll_prefill")["pass"])

    def test_rate_gate_is_wilson_upper_vs_base(self) -> None:
        cur = {"components": {"gsm8k": {"acc": q.rate(85, 100)}}}
        self.assertTrue(self._gate(q.gates(cur, _base()), "gsm8k.acc")["pass"])
        cur = {"components": {"gsm8k": {"acc": q.rate(80, 100)}}}
        self.assertFalse(self._gate(q.gates(cur, _base()), "gsm8k.acc")["pass"])

    def test_needle_and_selfcons_gates(self) -> None:
        cur = {"components": {
            "needle": {"cells": {"8192@0.1": {"found": True}, "8192@0.5": {"found": False},
                                 "131072@0.5": {"found": True}}},
            "selfcons": {"aa": {"hazard": 0.01}, "golden": {"hazard": 0.03}}}}
        rows = q.gates(cur, _base())
        needle = self._gate(rows, "needle.found")
        self.assertEqual((needle["value"], needle["limit"], needle["pass"]), (1, 2, False))
        self.assertTrue(self._gate(rows, "selfcons.golden_hazard")["pass"])
        cur["components"]["selfcons"]["golden"]["hazard"] = 0.05
        self.assertFalse(self._gate(q.gates(cur, _base()), "selfcons.golden_hazard")["pass"])

    def test_add_golden(self) -> None:
        sc = {"runs": [[[1, 2, 3]], [[1, 2, 4]]]}
        q.add_golden(sc, {"components": {"selfcons": {"runs": [[[1, 2, 3]], [[9]]]}}})
        self.assertEqual(sc["golden"]["first_div"], [None, 2])
        sc2 = {"runs": [[[1]]]}
        q.add_golden(sc2, None)
        self.assertNotIn("golden", sc2)


class VendoredDataTests(unittest.TestCase):
    def test_sets(self) -> None:
        nll = q._jsonl("nll_passages.jsonl")
        self.assertEqual(len(nll), 40)
        self.assertTrue(all(500 <= p["tokens"] <= 512 for p in nll))
        self.assertEqual(len({p["id"] for p in nll}), 40)
        gsm = q._jsonl("gsm8k_100.jsonl")
        self.assertEqual([g["idx"] for g in gsm], list(range(0, 1300, 13)))
        self.assertTrue(all(g["answer"].lstrip("-").isdigit() for g in gsm))
        mmlu = q._jsonl("mmlu_228.jsonl")
        self.assertEqual(len(mmlu), 228)
        self.assertEqual(len({m["subject"] for m in mmlu}), 57)
        self.assertTrue(all(m["answer"] in "ABCD" and len(m["choices"]) == 4 for m in mmlu))
        spec = json.loads((q.DATA / "tools30.json").read_text())
        items = spec["items"]
        self.assertEqual(len(items), 30)
        self.assertEqual(sum(i["expect"] is None for i in items), 8)
        for it in items:
            for t in it["tools"]:
                self.assertIn(t, spec["tools"])
            if it["expect"]:
                self.assertIn(it["expect"]["name"], it["tools"])
        lic = (q.DATA / "README.md").read_text()
        self.assertIn("Copyright (c) 2021 OpenAI", lic)
        self.assertIn("Copyright (c) 2020 Dan Hendrycks", lic)


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: ARG002
        return

    def _send(self, obj, ctype="application/json") -> None:
        out = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/metrics":
            self._send('vllm:prefix_cache_hits_total{engine="0"} 5.0\n', "text/plain")
        else:
            self._send({"data": [{"id": "fake-model", "root": "/snap/x", "max_model_len": 4096}]})

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["model"] == "fake-model"
        if self.path == "/v1/completions":
            assert body["prompt"].startswith(q.BOS_TEXT)
            ids = [0] + list(range(100, 140))
            plp = [None] + [{str(i): {"logprob": -0.5}} for i in ids[1:]]
            self._send({"choices": [{"prompt_token_ids": ids, "prompt_logprobs": plp}]})
            return
        content = body["messages"][0]["content"]
        assert body["chat_template_kwargs"] == q.CHAT_KWARGS
        if isinstance(content, list):
            text = "Red"
        else:
            text = "323" if "17*19" in content else "252"
        self._send({"choices": [{"message": {"content": text}}]})


class MainWiringTests(unittest.TestCase):
    def test_main_writes_json_and_gates(self) -> None:
        httpd = HTTPServer(("127.0.0.1", 0), _Fake)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        no_docker = mock.patch.object(q.subprocess, "run", side_effect=FileNotFoundError)
        try:
            with no_docker, tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "r.json"
                with redirect_stdout(io.StringIO()) as buf:
                    rc = q.main(["--url", url, "--only", "nll,vision,c2", "--out", str(out)])
                self.assertEqual(rc, 0, buf.getvalue())
                res = json.loads(out.read_text())
                nll = res["components"]["nll"]
                self.assertEqual(nll["tokens_scored"], 40 * (40 - q.NLL_SKIP))
                self.assertAlmostEqual(nll["mean_nll"], 0.5)
                self.assertNotIn("repeat_abs_delta", nll)   # quick scores once
                self.assertEqual(res["provenance"]["model"], "fake-model")
                self.assertEqual(res["provenance"]["image"], "unavailable (FileNotFoundError)")
                self.assertIn("QUALITY", buf.getvalue())
                base = Path(tmp) / "base.json"
                res["components"]["nll"]["mean_nll"] = 0.4
                base.write_text(json.dumps(res))
                with redirect_stdout(io.StringIO()):
                    rc = q.main(["--url", url, "--only", "nll", "--baseline", str(base)])
                self.assertEqual(rc, 1)   # 0.5 > 0.4 + 0.01
                # --result re-gates a saved run offline (no HTTP needed).
                with redirect_stdout(io.StringIO()):
                    rc_off = q.main(["--url", "http://127.0.0.1:9", "--result", str(out),
                                     "--baseline", str(base)])
                    rc_self = q.main(["--url", "http://127.0.0.1:9", "--result", str(base),
                                      "--baseline", str(base)])
                self.assertEqual((rc_off, rc_self), (1, 0))
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
