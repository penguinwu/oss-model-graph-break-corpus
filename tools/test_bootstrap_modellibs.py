#!/usr/bin/env python3
"""Tests for tools/bootstrap_modellibs.py.

Pins the tree_is_ready file-path check that prevents the empty-target
false-positive (encoded 2026-05-11 after live discovery during the
transformers 5.8.0 install — an empty modellibs target dir was reported
"ready" because the venv's site-packages had the same package version
and the import probe fell through to it).

Run: PYTHONPATH=$(pwd) python3 tools/test_bootstrap_modellibs.py
"""
from __future__ import annotations

import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap_modellibs as bm


class TreeIsReadyTests(unittest.TestCase):
    """Validate tree_is_ready dual-checks version + module file path."""

    def setUp(self):
        self._orig_tree_path = bm.tree_path

    def tearDown(self):
        bm.tree_path = self._orig_tree_path

    def _patched(self, fake_path: Path):
        bm.tree_path = lambda pkg, ver: fake_path

    def test_returns_false_when_path_does_not_exist(self):
        self._patched(Path("/nonexistent/path"))
        self.assertFalse(bm.tree_is_ready("transformers", "5.8.0", "/usr/bin/python3"))

    def test_returns_false_when_module_file_falls_through_to_site_packages(self):
        """Regression pin: empty target dir + venv's site-packages has same
        package + same version → tree_is_ready must return False."""
        with unittest.mock.patch("subprocess.run") as mock_run:
            # Simulate: probe imports transformers, gets version 5.8.0, but
            # the module file is from the venv's site-packages (NOT the target).
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout="5.8.0\n/home/user/envs/torch/lib/python3.12/site-packages/transformers/__init__.py\n",
                stderr="",
            )
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                self._patched(Path("/home/user/envs/modellibs/transformers-5.8.0"))
                result = bm.tree_is_ready("transformers", "5.8.0", "/fake/python")
                self.assertFalse(
                    result,
                    "tree_is_ready must REJECT when module loads from outside target_dir "
                    "(this is the empty-target-dir bug fix from 2026-05-11)",
                )

    def test_returns_true_when_module_file_under_target(self):
        """Positive path: module file IS under target_dir → ready."""
        target = "/home/user/envs/modellibs/transformers-5.8.0"
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=f"5.8.0\n{target}/transformers/__init__.py\n",
                stderr="",
            )
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                self._patched(Path(target))
                self.assertTrue(bm.tree_is_ready("transformers", "5.8.0", "/fake/python"))

    def test_returns_false_on_version_mismatch(self):
        target = "/home/user/envs/modellibs/transformers-5.8.0"
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0,
                stdout=f"5.7.1\n{target}/transformers/__init__.py\n",
                stderr="",
            )
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                self._patched(Path(target))
                self.assertFalse(bm.tree_is_ready("transformers", "5.8.0", "/fake/python"))

    def test_returns_false_on_probe_failure(self):
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=1,
                stdout="",
                stderr="ImportError: No module named transformers",
            )
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                self._patched(Path("/home/user/envs/modellibs/transformers-5.8.0"))
                self.assertFalse(bm.tree_is_ready("transformers", "5.8.0", "/fake/python"))

    def test_returns_false_on_malformed_probe_output(self):
        """Defensive: probe returns only one line (no module file) → reject."""
        with unittest.mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = unittest.mock.Mock(
                returncode=0, stdout="5.8.0\n", stderr="",
            )
            with unittest.mock.patch.object(Path, "exists", return_value=True):
                self._patched(Path("/home/user/envs/modellibs/transformers-5.8.0"))
                self.assertFalse(bm.tree_is_ready("transformers", "5.8.0", "/fake/python"))


if __name__ == "__main__":
    unittest.main()
