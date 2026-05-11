#!/usr/bin/env python3
"""Tests for tools/dedup_source_lines.py.

Pins the per-source-line dedup gate behavior (Peng directive 2026-05-11
following the umbrella-#122-vs-#77/#78 overlap miss).

Run: PYTHONPATH=$(pwd) python3 tools/test_dedup_source_lines.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dedup_source_lines as dsl


class ExtractSourceLinesTests(unittest.TestCase):
    def test_extracts_transformers_paths(self):
        text = """
        Top sites:
          16× transformers/models/opt/modeling_opt.py:375
          12× transformers/models/bart/modeling_bart.py:660
        """
        out = dsl.extract_source_lines(text)
        self.assertEqual(out, [
            "transformers/models/opt/modeling_opt.py:375",
            "transformers/models/bart/modeling_bart.py:660",
        ])

    def test_extracts_diffusers_torch_sweep_tools_paths(self):
        text = (
            "diffusers/models/foo.py:1\n"
            "torch/nn/functional.py:5621\n"
            "sweep/worker.py:42\n"
            "tools/file_issues.py:100\n"
        )
        out = dsl.extract_source_lines(text)
        self.assertEqual(set(out), {
            "diffusers/models/foo.py:1",
            "torch/nn/functional.py:5621",
            "sweep/worker.py:42",
            "tools/file_issues.py:100",
        })

    def test_dedupes_repeated_paths(self):
        text = "transformers/models/opt/modeling_opt.py:375 ... again transformers/models/opt/modeling_opt.py:375"
        out = dsl.extract_source_lines(text)
        self.assertEqual(out, ["transformers/models/opt/modeling_opt.py:375"])

    def test_no_match_returns_empty(self):
        text = "Just prose, no source-line citations here."
        self.assertEqual(dsl.extract_source_lines(text), [])

    def test_ignores_non_repo_paths(self):
        # Don't match arbitrary site-packages paths
        text = "/home/user/.venv/lib/python3.12/site-packages/some/file.py:99"
        self.assertEqual(dsl.extract_source_lines(text), [])


class GrepInBodyTests(unittest.TestCase):
    def test_exact_match(self):
        body = "We see this at transformers/models/opt/modeling_opt.py:375 in the failing case."
        self.assertTrue(dsl.grep_in_body(body, "transformers/models/opt/modeling_opt.py:375"))

    def test_loose_path_match_different_line(self):
        # #77's body cites bart:536, draft cites bart:660 — same file, different line
        body = "Cite at transformers/models/bart/modeling_bart.py:536 stack."
        self.assertTrue(dsl.grep_in_body(body, "transformers/models/bart/modeling_bart.py:660"))

    def test_no_match(self):
        body = "Totally different content about Tensor.item() at unrelated path."
        self.assertFalse(dsl.grep_in_body(body, "transformers/models/opt/modeling_opt.py:375"))

    def test_empty_body(self):
        self.assertFalse(dsl.grep_in_body("", "any/path.py:1"))


class FindOverlappingIssuesTests(unittest.TestCase):
    def setUp(self):
        self.issue_77 = {
            "number": 77,
            "title": "[dynamo] HF transformers layerdrop idiom: torch.rand([])",
            "body": "Cite at transformers/models/bart/modeling_bart.py:536 stack frame.",
            "labels": [{"name": "for:dynamo-team"}],
        }
        self.issue_78 = {
            "number": 78,
            "title": "[dynamo] Data-dep branching on torch.all(mask==1) mamba",
            "body": "transformers/models/qwen3_5/modeling_qwen3_5.py:1310 (close to 1312).",
            "labels": [{"name": "for:dynamo-team"}],
        }
        self.issue_55 = {
            "number": 55,
            "title": "[dynamo] _local_scalar_dense from .tolist()",
            "body": "Affects GraniteMoeHybrid at modeling_granitemoehybrid.py:924.",
            "labels": [{"name": "for:dynamo-team"}],
        }
        self.issue_120 = {
            "number": 120,
            "title": "[corpus-tooling] Some unrelated tooling issue",
            "body": "transformers/models/opt/modeling_opt.py:375 cited even though title is corpus-tooling.",
            "labels": [{"name": "for:corpus-tooling"}],
        }

    def test_finds_overlap_with_dynamo_issues(self):
        # Draft cites bart:660; #77 cites bart:536 → loose-path match
        source_lines = ["transformers/models/bart/modeling_bart.py:660"]
        overlaps = dsl.find_overlapping_issues(
            source_lines, [self.issue_77, self.issue_78], dynamo_only=True
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["issue_num"], 77)
        self.assertEqual(overlaps[0]["match_type"], "loose-path")

    def test_dynamo_only_filter(self):
        # #120 has corpus-tooling title — must be excluded by dynamo_only filter
        source_lines = ["transformers/models/opt/modeling_opt.py:375"]
        overlaps = dsl.find_overlapping_issues(
            source_lines, [self.issue_120], dynamo_only=True
        )
        self.assertEqual(overlaps, [])

    def test_all_open_finds_corpus_tooling_too(self):
        source_lines = ["transformers/models/opt/modeling_opt.py:375"]
        overlaps = dsl.find_overlapping_issues(
            source_lines, [self.issue_120], dynamo_only=False
        )
        self.assertEqual(len(overlaps), 1)
        self.assertEqual(overlaps[0]["issue_num"], 120)

    def test_multiple_source_lines_match_one_issue(self):
        # Draft cites 2 paths both in #77's range → 2 overlap rows
        source_lines = [
            "transformers/models/bart/modeling_bart.py:660",
            "transformers/models/bart/modeling_bart.py:537",
        ]
        overlaps = dsl.find_overlapping_issues(
            source_lines, [self.issue_77], dynamo_only=True
        )
        self.assertEqual(len(overlaps), 2)
        self.assertTrue(all(o["issue_num"] == 77 for o in overlaps))


if __name__ == "__main__":
    unittest.main()
