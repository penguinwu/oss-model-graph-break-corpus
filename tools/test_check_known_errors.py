#!/usr/bin/env python3
"""Tests for tools/check_known_errors.py.

Pins the bias-INFRA-FIX policy + applies_to_versions requirement.

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_check_known_errors.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "check_known_errors.py"
PYTHON = sys.executable


def _run(*args, fixture: dict | None = None):
    """Run check_known_errors.py with optional fixture path. Returns (rc, out, err)."""
    cmd = [PYTHON, str(TOOL)]
    tmp_path = None
    if fixture is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fixture, f)
            tmp_path = Path(f.name)
        cmd.append(str(tmp_path))
    cmd.extend(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if tmp_path:
        tmp_path.unlink(missing_ok=True)
    return r.returncode, r.stdout, r.stderr


def _eager_entry(model="ModelA", versions=None):
    return {
        "model": model, "modes": ["eval"], "status": "eager_error",
        "error_pattern": "boom",
        "applies_to_versions": versions or ["2.13"],
        "added": "2026-05-10", "reason": "test",
    }


def _create_entry(model="ModelB", versions=None, reason="test"):
    return {
        "model": model, "modes": ["eval"], "status": "create_error",
        "error_pattern": "ctor failed",
        "applies_to_versions": versions or ["2.13"],
        "added": "2026-05-10", "reason": reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_empty_entries_passes():
    rc, out, err = _run(fixture={"entries": []})
    assert rc == 0, f"empty entries should pass; got rc={rc}\n{err}"


def test_eager_error_with_versions_passes():
    rc, out, err = _run(fixture={"entries": [_eager_entry()]})
    assert rc == 0, f"eager_error with versions should pass; got rc={rc}\n{err}"


def test_create_error_with_short_reason_rejected():
    """Default: short `reason` field on create_error → REJECTED."""
    rc, out, err = _run(fixture={"entries": [_create_entry(reason="test")]})
    assert rc != 0, "create_error with reason='test' (4 chars) should be REJECTED"
    assert "create_error" in err and "reason" in err


def test_create_error_with_substantive_reason_passes():
    """create_error WITH substantive (≥30 char) reason field → PASSES (legacy + new)."""
    rc, out, err = _run(fixture={"entries": [_create_entry(
        reason="Upstream PyTorch decomp bug; tracked in pytorch/pytorch#12345 — Re-verify on 2.14+."
    )]})
    assert rc == 0, f"create_error with substantive reason should pass; got rc={rc}\n{err}"


def test_create_error_short_reason_with_override_passes():
    """Short reason BUT --allow-create-error-escape with --reason override → PASSES."""
    rc, out, err = _run(
        "--allow-create-error-escape",
        "--reason", "Investigated; will fix in next sweep cycle. Documented in retro Z.",
        fixture={"entries": [_create_entry(reason="x")]}
    )
    assert rc == 0, f"override should bypass short-reason check; got rc={rc}\n{err}"


def test_override_without_reason_fails():
    rc, out, err = _run("--allow-create-error-escape",
                        fixture={"entries": [_create_entry(reason="x"*40)]})
    assert rc != 0, "--allow-create-error-escape requires --reason"
    assert "--reason" in err


def test_short_cli_reason_fails():
    rc, out, err = _run("--allow-create-error-escape", "--reason", "too short",
                        fixture={"entries": [_create_entry(reason="x"*40)]})
    assert rc != 0, "short --reason should fail"


def test_missing_applies_to_versions_fails():
    """applies_to_versions is REQUIRED per audit_new_errors design rev 2 gap #6."""
    bad = _eager_entry()
    del bad["applies_to_versions"]
    rc, out, err = _run(fixture={"entries": [bad]})
    assert rc != 0, "missing applies_to_versions should fail"
    assert "applies_to_versions" in err


def test_invalid_status_fails():
    bad = _eager_entry()
    bad["status"] = "wrong_status"
    rc, out, err = _run(fixture={"entries": [bad]})
    assert rc != 0
    assert "invalid status" in err


def test_real_known_errors_file_passes():
    """The actual sweep/known_errors.json should pass (no policy violations as of today)."""
    rc, out, err = _run()  # no fixture → uses default path
    assert rc == 0, f"real known_errors.json should pass; got rc={rc}\n{err}"


def main() -> int:
    tests = [(n, fn) for n, fn in globals().items() if n.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
