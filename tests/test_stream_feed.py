#!/usr/bin/env python3
"""g8_stream_feed drain-in-place on a synthetic loader (CPU only, no torch)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_site_gates import run_site  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEED = _load(ROOT / "docker/patch/g8_stream_feed.py", "g8_stream_feed")

# Stand-in for vl_model.py: the stock wrapper method around a stub loader.
FAKE_VL_MODEL = '''from __future__ import annotations
from typing import Iterable


class AutoWeightsLoader:
    probe = None

    def __init__(self, model):
        pass

    def load_weights(self, weights):
        return AutoWeightsLoader.probe(weights)


class _Mapper:
    def apply(self, weights):
        yield from weights


class Wrapper:
    hf_to_vllm_mapper = _Mapper()

''' + FEED.INSTALL_OLD + '''
        self._weights_finalized = True
        return loaded_params
'''


class _W:
    """A weight; weakref-able so the probe can see when it is freed."""

    __slots__ = ("__weakref__",)


def _probe(weights):
    """Consume like AutoWeightsLoader; after each item, count earlier items still alive."""
    names, refs, retained = [], [], []
    for name, w in weights:
        names.append(name)
        refs.append(weakref.ref(w))
        retained.append(sum(r() is not None for r in refs[:-1]))
    return names, retained


class StreamFeedTests(unittest.TestCase):
    N = 64

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "vl_model.py"
        path.write_text(FAKE_VL_MODEL)
        with mock.patch.dict(os.environ, {"DSV41_VL_MODEL_PATH": str(path)}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(FEED.install())
            self.assertTrue(FEED.install())  # idempotent
        self.assertIn(FEED.MARKER, path.read_text())
        self.vl = _load(path, "fake_vl_model")
        self.vl.AutoWeightsLoader.probe = staticmethod(_probe)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, **env: str):
        clean = {k: v for k, v in os.environ.items()
                 if k not in ("DSV41_STREAM_FEED", "DSV41_LOAD_PF_G8")}
        clean.update(env)
        # Reverse order so the wrapper's sort matters.
        feed = ((f"w{i:03d}", _W()) for i in reversed(range(self.N)))
        with mock.patch.dict(os.environ, clean, clear=True):
            return self.vl.Wrapper().load_weights(feed)

    def test_drain_retains_o1_consumed_items(self) -> None:
        for flag in ("DSV41_STREAM_FEED", "DSV41_LOAD_PF_G8"):
            with self.subTest(flag=flag):
                names, retained = self._run(**{flag: "1"})
                self.assertEqual(names, sorted(names))
                self.assertEqual(len(names), self.N)
                self.assertEqual(max(retained), 0)

    def test_off_is_stock_and_retains_everything(self) -> None:
        for env in ({}, {"DSV41_STREAM_FEED": "0", "DSV41_LOAD_PF_G8": "0"}):
            with self.subTest(env=env):
                names, retained = self._run(**env)
                self.assertEqual(names, sorted(names))
                self.assertEqual(retained, list(range(self.N)))


class StreamFeedWiringTests(unittest.TestCase):
    def test_forwarded_to_both_ranks_default_on(self) -> None:
        run = (ROOT / "run.sh").read_text()
        block = re.search(r"^FORWARD_ENVS=\((.*?)^\)", run, re.M | re.S)
        self.assertIsNotNone(block)
        self.assertIn("DSV41_STREAM_FEED=", block.group(1).split())
        self.assertIn('DSV41_STREAM_FEED="${DSV41_STREAM_FEED:-1}"', run.splitlines())  # round 34 (s4)

    def test_sitecustomize_installs_on_stream_feed_only_when_set(self) -> None:
        self.assertNotIn("STUB g8_stream_feed", run_site(["g8_stream_feed"]))
        self.assertNotIn("STUB g8_stream_feed", run_site(["g8_stream_feed"], DSV41_STREAM_FEED="0"))
        self.assertIn("STUB g8_stream_feed", run_site(["g8_stream_feed"], DSV41_STREAM_FEED="1"))


if __name__ == "__main__":
    unittest.main()
