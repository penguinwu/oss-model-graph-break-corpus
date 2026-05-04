"""Tests for sweep/results_loader + enforcement of single-reader rule.

The enforcement test (test_no_direct_identify_results_reader) greps the
repo for direct json.load() of identify_results.json. Anyone bypassing
the loader fails this test — keeps amendment-awareness universal.
"""
import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from sweep.results_loader import (
    load_amendments_metadata,
    load_effective_results,
    load_raw,
    load_results_list,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files allowed to read identify_results.json directly:
# - the loader itself (it IS the canonical reader)
# - writers (orchestrator, run_sweep, amend_sweep) — they write, not just read
# - tools/run_experiment.py (delegates to run_sweep)
ALLOWED_DIRECT_READERS = {
    "sweep/results_loader.py",
    "sweep/run_sweep.py",
    "sweep/orchestrator.py",
    "tools/run_experiment.py",
    "tools/amend_sweep.py",
    "sweep/test_results_loader.py",  # this file documents the rule
    # Mentions "identify_results.json" in a help-string only — no actual read
    "tools/analyze_explain.py",
    # Calls Path(...identify_results.json).exists() for presence check; no content read
    "tools/sweep_watchdog_check.py",
}


class LoaderBasicsTest(unittest.TestCase):
    """Round-trip: build a synthetic sweep, load it, verify shape."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sweep = Path(self.tmp)
        original_data = {
            "metadata": {"pass": "identify", "python": "torch=2.13.0+cu126"},
            "results": [
                {"name": "Foo", "mode": "eval", "status": "full_graph"},
                {"name": "Foo", "mode": "train", "status": "graph_break"},
                {"name": "Bar", "mode": "eval", "status": "eager_error"},
            ],
        }
        (self.sweep / "identify_results.json").write_text(json.dumps(original_data))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_no_amendments_returns_originals(self):
        eff = load_effective_results(self.sweep)
        self.assertEqual(len(eff), 3)
        self.assertEqual(eff[("Foo", "eval")]["status"], "full_graph")
        self.assertEqual(eff[("Foo", "eval")]["result_source"], "original")

    def test_amendment_supersedes_original(self):
        data = json.loads((self.sweep / "identify_results.json").read_text())
        data["amendments"] = [{
            "amendment_id": "2026-05-04T13-30Z-aria-fix",
            "applied_at": "2026-05-04T13:30:00Z",
            "fix_commit": "abc123",
            "fix_description": "test",
            "trigger": "test",
            "rows": [
                {"name": "Bar", "mode": "eval", "status": "graph_break"},
            ],
        }]
        (self.sweep / "identify_results.json").write_text(json.dumps(data))
        eff = load_effective_results(self.sweep)
        self.assertEqual(eff[("Bar", "eval")]["status"], "graph_break")
        self.assertEqual(eff[("Bar", "eval")]["result_source"],
                         "amended:2026-05-04T13-30Z-aria-fix")
        # Foo unchanged
        self.assertEqual(eff[("Foo", "eval")]["result_source"], "original")

    def test_later_amendment_supersedes_earlier(self):
        data = json.loads((self.sweep / "identify_results.json").read_text())
        data["amendments"] = [
            {"amendment_id": "first", "applied_at": "2026-05-04T13:00:00Z",
             "fix_commit": "a", "fix_description": "x", "trigger": "x",
             "rows": [{"name": "Foo", "mode": "eval", "status": "graph_break"}]},
            {"amendment_id": "second", "applied_at": "2026-05-04T14:00:00Z",
             "fix_commit": "b", "fix_description": "y", "trigger": "y",
             "rows": [{"name": "Foo", "mode": "eval", "status": "full_graph"}]},
        ]
        (self.sweep / "identify_results.json").write_text(json.dumps(data))
        eff = load_effective_results(self.sweep)
        self.assertEqual(eff[("Foo", "eval")]["status"], "full_graph")
        self.assertEqual(eff[("Foo", "eval")]["result_source"], "amended:second")

    def test_amendments_metadata(self):
        data = json.loads((self.sweep / "identify_results.json").read_text())
        data["amendments"] = [{
            "amendment_id": "test-id", "applied_at": "2026-05-04T13:30:00Z",
            "fix_commit": "abc", "fix_description": "x", "trigger": "y",
            "rows": [{"name": "Foo", "mode": "eval", "status": "full_graph"}],
        }]
        (self.sweep / "identify_results.json").write_text(json.dumps(data))
        meta = load_amendments_metadata(self.sweep)
        self.assertEqual(len(meta), 1)
        self.assertEqual(meta[0]["amendment_id"], "test-id")
        self.assertEqual(meta[0]["row_count"], 1)
        # Rows themselves should NOT be in metadata
        self.assertNotIn("rows", meta[0])


class EnforcementTest(unittest.TestCase):
    """Asserts no consumer reads identify_results.json directly."""

    def test_no_direct_identify_results_reader(self):
        """Detects ACTUAL reads (json.load/open) of identify_results.json.

        Allows mentions in docstrings/help-text/path construction — only flags
        files that actually parse the file content.
        """
        # Two patterns of a real read:
        #   1. json.load(open(...identify_results.json...))
        #   2. open(...identify_results.json...) followed by json.load
        # Both reduce to: file mentions identify_results.json AND calls json.load
        # on a path/handle derived from that string. We detect by scanning each
        # file's source: if any line containing "identify_results.json" is also
        # adjacent to a json.load/json.loads call, flag it.
        py_files = list((REPO_ROOT / "sweep").rglob("*.py")) + list((REPO_ROOT / "tools").rglob("*.py"))
        violations = []
        for path in py_files:
            rel = os.path.relpath(path, REPO_ROOT)
            if rel in ALLOWED_DIRECT_READERS:
                continue
            src = path.read_text()
            # Quick reject: file doesn't mention the canonical name at all
            if "identify_results.json" not in src and "identify_streaming.jsonl" not in src:
                continue
            # Tokenize-ish: find identify_results.json mentions outside of strings
            # ending in .json that are PASSED INTO json.load. Heuristic: look for
            # patterns indicating actual reads.
            real_read_patterns = [
                r"json\.load\s*\(\s*open\s*\([^)]*identify_(?:results\.json|streaming\.jsonl)",
                r"json\.loads\s*\([^)]*identify_(?:results\.json|streaming\.jsonl)\.read",
                r"open\s*\([^)]*identify_(?:results\.json|streaming\.jsonl)[^)]*\)\s*as\s+f\s*:[^j]*json\.load",
            ]
            multiline_check = re.search(
                r"open\s*\([^)]*identify_(?:results\.json|streaming\.jsonl)[^)]*\)\s*as\s+\w+\s*:.{0,200}?json\.load",
                src, flags=re.DOTALL)
            simple_match = any(re.search(p, src) for p in real_read_patterns)
            if simple_match or multiline_check:
                violations.append(rel)
        self.assertFalse(
            violations,
            f"\n  Files actually reading identify_results.json without going through "
            f"sweep/results_loader.py:\n    "
            + "\n    ".join(violations)
            + "\n  Either migrate to load_effective_results() or, if the file "
            "is a writer, add it to ALLOWED_DIRECT_READERS in this test."
        )


if __name__ == "__main__":
    unittest.main()
