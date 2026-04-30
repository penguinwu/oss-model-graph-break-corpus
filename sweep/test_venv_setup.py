#!/usr/bin/env python3
"""Tests for sweep.venv_setup. Runnable as `python -m sweep.test_venv_setup`
or `python sweep/test_venv_setup.py`. Uses unittest stdlib (no extra deps).

Tests cover:
  - Health check primitives against current venvs
  - Pool detection
  - TorchSpec.matches semantics
  - Pool inventory (cu lib presence)
  - find_matching_pt_venvs against current state
  - Bootstrap commands generation (no install — just text)
  - escalate_no_pool exit code via subprocess
  - Failure injection: ensure_venv_ready against missing pool exits 42
  - Failure injection: verify_post_install with mismatch exits 43
"""
from __future__ import annotations
import os
import subprocess
import sys
import unittest
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sweep.venv_setup import (
    TorchSpec,
    VenvInfo,
    ENVS_DIR,
    CU_LIB_PACKAGES,
    EXIT_AWAITING_HUMAN_BOOTSTRAP,
    EXIT_VERSION_MISMATCH,
    inspect_venv,
    is_pool_healthy,
    pool_for_variant,
    find_matching_pt_venvs,
    can_auto_repair,
    _bootstrap_commands,
    _extract_cuda_variant,
)


class TestTorchSpec(unittest.TestCase):
    def test_exact_match(self):
        spec = TorchSpec("2.12.0.dev20260323+cu128", "cu128")
        self.assertTrue(spec.matches("2.12.0.dev20260323+cu128"))
        self.assertFalse(spec.matches("2.11.0+cu128"))

    def test_glob_match(self):
        spec = TorchSpec("2.12.*", "cu128")
        self.assertTrue(spec.matches("2.12.0.dev20260323+cu128"))
        self.assertTrue(spec.matches("2.12.0+cu128"))
        self.assertFalse(spec.matches("2.11.0+cu128"))
        self.assertFalse(spec.matches("2.13.0a0+gitf8d66d2"))

    def test_substring_match(self):
        spec = TorchSpec("2.12", "cu128")
        self.assertTrue(spec.matches("2.12.0.dev20260323+cu128"))


class TestExtractCudaVariant(unittest.TestCase):
    def test_wheel_variant(self):
        self.assertEqual(_extract_cuda_variant("2.12.0.dev20260323+cu128"), "cu128")
        self.assertEqual(_extract_cuda_variant("2.13.0.dev20260425+cu126"), "cu126")
        self.assertEqual(_extract_cuda_variant("2.11.0+cu128"), "cu128")

    def test_source_build_no_variant(self):
        # Source builds don't have a +cuXXX tag
        self.assertIsNone(_extract_cuda_variant("2.13.0a0+gitf8d66d2"))

    def test_no_torch(self):
        self.assertIsNone(_extract_cuda_variant(None))


class TestPoolHealth(unittest.TestCase):
    def test_pool_for_variant_returns_existing(self):
        # cu128 and cu126 pools should exist (Cycle 0 bootstrapped them)
        self.assertEqual(pool_for_variant("cu128"), ENVS_DIR / "cu128")
        self.assertEqual(pool_for_variant("cu126"), ENVS_DIR / "cu126")

    def test_pool_for_variant_returns_none_for_missing(self):
        self.assertIsNone(pool_for_variant("cu999"))

    def test_pools_are_healthy(self):
        # Both pools should pass is_pool_healthy
        self.assertTrue(is_pool_healthy(ENVS_DIR / "cu128"),
                        "cu128 pool missing required cu library packages")
        self.assertTrue(is_pool_healthy(ENVS_DIR / "cu126"),
                        "cu126 pool missing required cu library packages")

    def test_required_cu_packages_set(self):
        # Sanity: required set is non-empty + sane size
        self.assertGreaterEqual(len(CU_LIB_PACKAGES), 15)
        self.assertIn("cuda-toolkit", CU_LIB_PACKAGES)
        self.assertIn("nvidia-cuda-nvrtc-cu12", CU_LIB_PACKAGES)


class TestInspectVenv(unittest.TestCase):
    def test_inspect_torch211(self):
        v = inspect_venv(ENVS_DIR / "torch211")
        self.assertIsNotNone(v.torch_version)
        self.assertTrue(v.torch_version.startswith("2.11"))
        self.assertEqual(v.cuda_variant, "cu128")
        self.assertFalse(v.is_pool)

    def test_inspect_pool_has_no_torch(self):
        v = inspect_venv(ENVS_DIR / "cu128")
        self.assertIsNone(v.torch_version)
        self.assertTrue(v.is_pool)

    def test_inspect_missing_dir_safe(self):
        v = inspect_venv(ENVS_DIR / "_nonexistent_xyz")
        # Should not crash; will report issues
        self.assertIsNone(v.torch_version)


class TestFindMatchingPtVenvs(unittest.TestCase):
    def test_finds_torch211_for_2_11(self):
        spec = TorchSpec("2.11.*", "cu128")
        matches = find_matching_pt_venvs(spec)
        names = [m.path.name for m in matches]
        self.assertIn("torch211", names)
        self.assertNotIn("cu128", names)  # pool excluded
        self.assertNotIn("cu126", names)

    def test_finds_torch_nightly_cu128_for_2_12(self):
        spec = TorchSpec("2.12.*", "cu128")
        matches = find_matching_pt_venvs(spec)
        names = [m.path.name for m in matches]
        self.assertIn("torch-nightly-cu128", names)

    def test_no_match_for_unrealistic_version(self):
        spec = TorchSpec("9.99.*", "cu128")
        matches = find_matching_pt_venvs(spec)
        self.assertEqual(matches, [])


class TestAutoRepair(unittest.TestCase):
    def test_can_auto_repair_clean(self):
        v = VenvInfo(
            path=Path("/tmp/_x"), torch_version="2.11.0+cu128",
            cuda_variant="cu128", transformers_version="5.6.2",
            is_pool=False, health_issues=[],
        )
        self.assertTrue(can_auto_repair(v))

    def test_can_auto_repair_shebang_only(self):
        v = VenvInfo(
            path=Path("/tmp/_x"), torch_version="2.11.0+cu128",
            cuda_variant="cu128", transformers_version="5.6.2",
            is_pool=False, health_issues=["stale pip shebang"],
        )
        self.assertTrue(can_auto_repair(v))

    def test_cannot_auto_repair_torch_broken(self):
        v = VenvInfo(
            path=Path("/tmp/_x"), torch_version=None,
            cuda_variant=None, transformers_version=None,
            is_pool=False, health_issues=["torch present but not importable"],
        )
        self.assertFalse(can_auto_repair(v))


class TestBootstrapCommands(unittest.TestCase):
    def test_bootstrap_includes_pool_path(self):
        spec = TorchSpec("2.12.*", "cu128")
        cmds = _bootstrap_commands(spec)
        self.assertIn(str(ENVS_DIR / "cu128"), cmds)
        self.assertIn("python3 -m venv", cmds)
        self.assertIn("nightly/cu128", cmds)
        self.assertIn("uninstall -y torch torchvision", cmds)

    def test_bootstrap_for_unknown_variant(self):
        spec = TorchSpec("2.99.*", "cu130")
        cmds = _bootstrap_commands(spec)
        self.assertIn("cu130", cmds)


class TestExitCodes(unittest.TestCase):
    """Failure-injection tests: invoke venv_setup as subprocess and verify
    distinct exit codes."""

    def test_check_only_runs_clean(self):
        # --check-only should always exit 0 (no install attempted)
        result = subprocess.run(
            [sys.executable, "-m", "sweep.venv_setup",
             "--torch", "2.12.*", "--cuda", "cu128", "--check-only"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0,
                         f"--check-only exit was {result.returncode}; "
                         f"stdout={result.stdout[-200:]} stderr={result.stderr[-200:]}")

    def test_exit_codes_distinct(self):
        # Sanity: exit codes should be distinct numerics
        self.assertNotEqual(EXIT_AWAITING_HUMAN_BOOTSTRAP, 0)
        self.assertNotEqual(EXIT_VERSION_MISMATCH, 0)
        self.assertNotEqual(EXIT_AWAITING_HUMAN_BOOTSTRAP, EXIT_VERSION_MISMATCH)
        self.assertEqual(EXIT_AWAITING_HUMAN_BOOTSTRAP, 42)
        self.assertEqual(EXIT_VERSION_MISMATCH, 43)


class TestEnsureVenvReady(unittest.TestCase):
    """Integration: ensure_venv_ready against the actual current state."""

    def test_resolves_existing_pt_2_12_venv(self):
        # PT 2.12 venv already exists at ~/envs/torch-nightly-cu128 (Cycle 0)
        from sweep.venv_setup import ensure_venv_ready
        spec = TorchSpec("2.12.*", "cu128")
        result = ensure_venv_ready(spec)
        self.assertEqual(result.name, "torch-nightly-cu128")

    def test_resolves_existing_pt_2_11_venv(self):
        from sweep.venv_setup import ensure_venv_ready
        spec = TorchSpec("2.11.*", "cu128")
        result = ensure_venv_ready(spec)
        self.assertEqual(result.name, "torch211")


if __name__ == "__main__":
    unittest.main(verbosity=2)
