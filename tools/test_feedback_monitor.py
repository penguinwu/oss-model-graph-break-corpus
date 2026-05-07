#!/usr/bin/env python3
"""Tests for tools/feedback_monitor.py.

Backfills tests for the May-4 incident bug fix (separate replied_messages
from processed_messages) — per docs/testing.md, every bug fix gets a test.

Run: python3 tools/test_feedback_monitor.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent))
import feedback_monitor as fm  # noqa: E402


def _isolate(tmp: Path):
    """Point fm at temp paths so tests don't touch real state files."""
    fm.STATE_FILE = tmp / "state.json"
    fm.AUDIT_LOG = tmp / "audit.jsonl"


def test_load_state_old_format_migrates_with_replied_default():
    """Pre-fix state files lack 'replied_messages' — load_state must default it to []."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        old = {"last_check_epoch": 12345, "processed_messages": ["m1", "m2"]}
        fm.STATE_FILE.write_text(json.dumps(old))
        loaded = fm.load_state()
        assert "replied_messages" in loaded, "replied_messages key missing after load_state"
        assert loaded["replied_messages"] == [], f"expected [], got {loaded['replied_messages']}"
        assert loaded["processed_messages"] == ["m1", "m2"], "processed_messages not preserved"


def test_load_state_no_file_returns_default_with_both_fields():
    """When state file doesn't exist, both processed_messages and replied_messages default to []."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        loaded = fm.load_state()
        assert loaded["processed_messages"] == []
        assert loaded["replied_messages"] == []


def test_save_state_caps_replied_at_1000():
    """replied_messages is capped at 1000 to prevent unbounded growth."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        state = {
            "last_check_epoch": 0,
            "processed_messages": [f"p{i}" for i in range(600)],
            "replied_messages": [f"r{i}" for i in range(1500)],
        }
        fm.save_state(state)
        loaded = json.loads(fm.STATE_FILE.read_text())
        assert len(loaded["replied_messages"]) == 1000, f"got {len(loaded['replied_messages'])}"
        # Latest 1000 are kept (last 1000 of the original 1500)
        assert loaded["replied_messages"][0] == "r500"
        assert loaded["replied_messages"][-1] == "r1499"
        # processed cap stays at 500
        assert len(loaded["processed_messages"]) == 500


def test_mark_replied_appends_dedupes():
    """mark_replied appends new IDs and skips duplicates."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        fm.mark_replied(["A", "B", "A", "C"])
        loaded = fm.load_state()
        assert sorted(loaded["replied_messages"]) == ["A", "B", "C"], \
            f"got {loaded['replied_messages']}"
        # Calling again with overlapping IDs doesn't re-add them
        fm.mark_replied(["B", "D"])
        loaded = fm.load_state()
        assert sorted(loaded["replied_messages"]) == ["A", "B", "C", "D"]


def test_list_needs_reply_filters_replied():
    """list_needs_reply skips IDs that have been mark_replied'd."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        # Audit log: 2 needs_answer + 1 needs_action + 1 unrelated
        fm.AUDIT_LOG.write_text("\n".join([
            json.dumps({"action": "needs_answer", "message_id": "m1", "text": "?"}),
            json.dumps({"action": "needs_answer", "message_id": "m2", "text": "?"}),
            json.dumps({"action": "needs_action", "message_id": "m3", "text": "?"}),
            json.dumps({"action": "auto_reply",   "message_id": "m4", "text": "?"}),
        ]) + "\n")
        # Reply to m1 only
        fm.mark_replied(["m1"])
        pending = fm.list_needs_reply()
        ids = {p["message_id"] for p in pending}
        assert ids == {"m2", "m3"}, f"expected {{m2, m3}}, got {ids}"


def test_list_needs_reply_dedupes_within_log():
    """If the audit log contains the same message_id twice, list_needs_reply returns it once."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        fm.AUDIT_LOG.write_text("\n".join([
            json.dumps({"action": "needs_answer", "message_id": "m1", "text": "?"}),
            json.dumps({"action": "needs_answer", "message_id": "m1", "text": "?"}),
        ]) + "\n")
        pending = fm.list_needs_reply()
        assert len(pending) == 1, f"expected 1 (deduped), got {len(pending)}"


def test_list_needs_reply_skips_malformed_lines():
    """Malformed JSON lines in the audit log are silently skipped, not crashed on."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _isolate(tmp)
        fm.AUDIT_LOG.write_text("\n".join([
            "not-valid-json",
            json.dumps({"action": "needs_answer", "message_id": "m1"}),
            "{also bad",
        ]) + "\n")
        pending = fm.list_needs_reply()
        assert len(pending) == 1


# Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
