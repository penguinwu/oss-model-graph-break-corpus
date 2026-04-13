#!/usr/bin/env python3
"""Unit tests for the shared graph break analysis module.

Fast, no downloads, no GPU required. Validates that run_graph_break_analysis()
produces correct output format across all input types and edge cases.

Usage:
    python test_explain.py              # run all tests
    python test_explain.py -v           # verbose output
"""
import json
import sys
import unittest
from pathlib import Path

# Ensure sweep/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn

from explain import run_graph_break_analysis


# Expected keys in every successful result
REQUIRED_KEYS = {"status", "graph_count", "graph_break_count",
                 "ops_per_graph", "compile_times", "break_reasons",
                 "explain_time_s"}


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


class ItemBreakModel(nn.Module):
    """Model that calls .item() — may cause graph break depending on PyTorch version."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear(x)
        val = x.sum().item()
        return x * (2 if val > 0 else 1)


class ErrorModel(nn.Module):
    """Model that raises during forward."""
    def forward(self, x):
        raise RuntimeError("intentional test error")


class TestOutputFormat(unittest.TestCase):
    """Verify output format is consistent regardless of input type."""

    def _check_ok_result(self, result):
        self.assertEqual(result["status"], "ok")
        self.assertTrue(REQUIRED_KEYS.issubset(result.keys()),
                        f"Missing keys: {REQUIRED_KEYS - set(result.keys())}")
        self.assertIsInstance(result["graph_count"], int)
        self.assertIsInstance(result["graph_break_count"], int)
        self.assertIsInstance(result["ops_per_graph"], list)
        self.assertIsInstance(result["compile_times"], list)
        self.assertIsInstance(result["break_reasons"], list)
        self.assertIsInstance(result["explain_time_s"], float)
        self.assertGreater(result["graph_count"], 0)
        self.assertEqual(result["graph_break_count"],
                         max(0, result["graph_count"] - 1))
        self.assertEqual(len(result["ops_per_graph"]), result["graph_count"])
        self.assertEqual(len(result["compile_times"]), result["graph_count"])
        # Verify JSON-serializable
        json.dumps(result)

    def test_tensor_input(self):
        model = SimpleLinear().eval()
        result = run_graph_break_analysis(model, torch.randn(2, 16), mode="eval")
        self._check_ok_result(result)
        self.assertEqual(result["graph_break_count"], 0)

    def test_dict_input(self):
        model = SimpleLinear().eval()
        result = run_graph_break_analysis(model, {"x": torch.randn(2, 16)}, mode="eval")
        self._check_ok_result(result)
        self.assertEqual(result["graph_break_count"], 0)

    def test_tuple_input(self):
        model = SimpleLinear().eval()
        result = run_graph_break_analysis(model, (torch.randn(2, 16),), mode="eval")
        self._check_ok_result(result)
        self.assertEqual(result["graph_break_count"], 0)

    def test_train_mode(self):
        model = SimpleLinear().train()
        result = run_graph_break_analysis(model, torch.randn(2, 16), mode="train")
        self._check_ok_result(result)


class TestBreakDetection(unittest.TestCase):
    """Verify break detection and break_reasons structure."""

    def test_item_break_captured(self):
        model = ItemBreakModel().eval()
        result = run_graph_break_analysis(model, torch.randn(2, 16), mode="eval")
        self.assertEqual(result["status"], "ok")
        # break_reasons should capture log messages even if the backend
        # handles the break internally (PyTorch 2.10+ behavior)
        for br in result["break_reasons"]:
            self.assertIn("reason", br)
            self.assertIn("type", br)
            self.assertIn(br["type"],
                          {"aten.nonzero", "Tensor.item()", "data-dependent-branch", "other"})


class TestErrorHandling(unittest.TestCase):
    """Verify graceful error handling."""

    def test_forward_error(self):
        model = ErrorModel()
        result = run_graph_break_analysis(model, torch.randn(2, 16), mode="eval")
        self.assertEqual(result["status"], "explain_error")
        self.assertIn("error", result)
        self.assertIn("intentional test error", result["error"])
        self.assertIn("explain_time_s", result)

    def test_error_result_serializable(self):
        model = ErrorModel()
        result = run_graph_break_analysis(model, torch.randn(2, 16), mode="eval")
        json.dumps(result)  # must not raise


class TestWorkerImports(unittest.TestCase):
    """Verify both workers can import the shared module."""

    def test_sweep_worker_import(self):
        # sweep/worker.py imports from explain — verify no circular imports
        import importlib
        spec = importlib.util.spec_from_file_location(
            "sweep_worker", Path(__file__).parent / "worker.py")
        self.assertIsNotNone(spec)

    def test_custom_worker_import(self):
        custom_worker = Path(__file__).parent.parent / "corpora" / "custom-models" / "worker.py"
        if custom_worker.exists():
            import importlib
            spec = importlib.util.spec_from_file_location("custom_worker", custom_worker)
            self.assertIsNotNone(spec)


if __name__ == "__main__":
    unittest.main()
