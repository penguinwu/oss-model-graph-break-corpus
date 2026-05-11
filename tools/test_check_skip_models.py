#!/usr/bin/env python3
"""Tests for tools/check_skip_models.py + sweep/skip_models_loader.py.

Pins the dict-of-objects schema + the legacy-list rejection.

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_check_skip_models.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "check_skip_models.py"
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT / "sweep"))
from skip_models_loader import load_skip_models, load_skip_models_raw  # noqa: E402


def _run(*args, fixture=None):
    cmd = [PYTHON, str(TOOL)]
    tmp = None
    if fixture is not None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(fixture, f)
            tmp = Path(f.name)
        cmd.append(str(tmp))
    cmd.extend(args)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if tmp:
        tmp.unlink(missing_ok=True)
    return r.returncode, r.stdout, r.stderr


# ─────────────────────────────────────────────────────────────────────────────
# skip_models_loader tests
# ─────────────────────────────────────────────────────────────────────────────


def test_loader_handles_legacy_list_format():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(["A", "B", "C"], f); p = Path(f.name)
    try:
        assert load_skip_models(p) == {"A", "B", "C"}
    finally:
        p.unlink(missing_ok=True)


def test_loader_handles_dict_format():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"X": {"reason": "r"}, "Y": {"reason": "r"}}, f); p = Path(f.name)
    try:
        assert load_skip_models(p) == {"X", "Y"}
    finally:
        p.unlink(missing_ok=True)


def test_loader_missing_file_returns_empty():
    assert load_skip_models(Path("/tmp/does-not-exist.json")) == set()


def test_loader_real_file_loads():
    """The real sweep/skip_models.json loads (whichever format it's in today)."""
    names = load_skip_models()
    assert isinstance(names, set)
    assert len(names) > 0  # we know there are entries today


# ─────────────────────────────────────────────────────────────────────────────
# check_skip_models.py tests
# ─────────────────────────────────────────────────────────────────────────────


def _good_entry(reason="A documented reason explaining why this model is on the skip list (≥30 chars).",
                follow_up=None, added="2026-05-10"):
    return {"reason": reason, "follow_up_task": follow_up, "added": added}


def test_legacy_list_format_rejected_by_default():
    rc, out, err = _run(fixture=["A", "B"])
    assert rc == 2, f"legacy list should exit 2; got rc={rc}"
    assert "legacy" in err.lower() or "TEMPORARY" in err


def test_legacy_list_allowed_with_flag():
    rc, out, err = _run("--allow-legacy-format", fixture=["A", "B"])
    assert rc == 0


def test_dict_with_complete_entries_passes():
    rc, out, err = _run(fixture={
        "ModelA": _good_entry(),
        "ModelB": _good_entry(follow_up="Investigate then remove"),
    })
    assert rc == 0, f"valid dict should pass; got rc={rc}\n{err}"


def test_missing_required_fields_fails():
    rc, out, err = _run(fixture={"ModelA": {"reason": "x"*40}})  # no follow_up_task, no added
    assert rc != 0
    assert "follow_up_task" in err or "added" in err


def test_short_reason_fails():
    rc, out, err = _run(fixture={"ModelA": _good_entry(reason="too short")})
    assert rc != 0
    assert "reason" in err


def test_invalid_added_date_fails():
    rc, out, err = _run(fixture={"ModelA": _good_entry(added="yesterday")})
    assert rc != 0
    assert "added" in err


def test_stale_entry_warns_but_passes():
    """Entry from 200 days ago → warning, but valid schema → exit 0."""
    rc, out, err = _run(fixture={"OldModel": _good_entry(added="2025-10-23")})
    assert rc == 0  # warnings don't fail
    assert "STALE" in err or "stale" in err.lower()


def test_no_warn_stale_suppresses():
    rc, out, err = _run("--no-warn-stale", fixture={"OldModel": _good_entry(added="2025-10-23")})
    assert rc == 0
    assert "stale" not in err.lower()


def test_real_skip_models_json_state():
    """The current sweep/skip_models.json — either passes (after migration) or
    rejects with legacy format error. Either is acceptable; this test pins
    that the tool runs without crashing."""
    rc, out, err = _run()
    # Must be 0 (dict, valid) OR 2 (legacy list, needs migration). Not 1 (corrupt).
    assert rc in (0, 2), f"expected 0 (valid dict) or 2 (legacy list); got rc={rc}\n{err}"


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
