#!/usr/bin/env python3
"""Tests for tools/build_invocations_log.py.

Pins the aggregator's robustness against malformed per-case files (adversary
impl-review case adv-2026-05-08-161753-file-issue-impl gap #5). Without these
tests, silent corruption in per-case files would slip through and the
aggregate index would show `?` placeholders without any warning.

Run: python3 tools/test_build_invocations_log.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_invocations_log as bil  # noqa: E402


# Helpers ─────────────────────────────────────────────────────────────────────

def _make_subagent_dir(tmp: Path, files: dict[str, str]) -> Path:
    """Create a synthetic subagents/foo/ tree under tmp with the given files."""
    sub = tmp / "foo"
    inv = sub / "invocations"
    inv.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (inv / name).write_text(content)
    return sub


def _good_case(case_id: str = "test-2026-05-08-080000-foo") -> str:
    return f"""---
case_id: {case_id}
date_utc: 2026-05-08T08:00:00Z
trigger: smoke
verdict: approve
output_sha256: deadbeefcafebabe
---

## Reviewer raw output

```
ok
```
"""


# Tests ─────────────────────────────────────────────────────────────────────

def test_parse_frontmatter_good_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case.md"
        p.write_text(_good_case())
        fields, warnings = bil.parse_frontmatter(p)
        assert warnings == [], f"expected no warnings; got: {warnings}"
        assert fields.get("case_id") == "test-2026-05-08-080000-foo"
        assert fields.get("output_sha256") == "deadbeefcafebabe"
        assert fields.get("verdict") == "approve"


def test_parse_frontmatter_no_leading_marker():
    """Missing `---\\n` at start should produce a warning, not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case.md"
        p.write_text("just markdown, no frontmatter\n")
        fields, warnings = bil.parse_frontmatter(p)
        assert fields == {}
        assert len(warnings) == 1
        assert "missing leading" in warnings[0]


def test_parse_frontmatter_unclosed():
    """Frontmatter that opens but never closes should warn."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case.md"
        p.write_text("---\ncase_id: foo\nno closing marker\n")
        fields, warnings = bil.parse_frontmatter(p)
        assert fields == {}
        assert len(warnings) == 1
        assert "not closed" in warnings[0]


def test_parse_frontmatter_handles_digit_keys():
    """Regression for gap #5 (mid-implementation bug): regex `[a-z_]+` missed
    `output_sha256` because of digits. Must use `[a-z0-9_]+`.
    """
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "case.md"
        p.write_text("---\noutput_sha256: abc123\nbody_sha256: def456\n---\n")
        fields, _ = bil.parse_frontmatter(p)
        assert fields.get("output_sha256") == "abc123", \
            f"output_sha256 should parse despite digits in key; got: {fields}"
        assert fields.get("body_sha256") == "def456"


def test_build_index_well_formed():
    with tempfile.TemporaryDirectory() as tmp:
        sub = _make_subagent_dir(Path(tmp), {"good.md": _good_case()})
        content, warnings = bil.build_index(sub)
        assert warnings == [], f"expected no warnings; got: {warnings}"
        assert "deadbeefcafe" in content, "sha256 should appear (truncated) in row"
        assert "approve" in content
        assert "Total invocations: 1" in content


def test_build_index_warns_on_missing_required_fields():
    """Per-file missing required fields must warn, not silently use `?`."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = "---\nfoo: bar\n---\nno required fields\n"
        sub = _make_subagent_dir(Path(tmp), {"bad.md": bad})
        content, warnings = bil.build_index(sub)
        # Warning should name the missing fields
        assert any("case_id" in w for w in warnings), \
            f"warning should mention case_id; got: {warnings}"
        assert any("date_utc" in w for w in warnings), \
            f"warning should mention date_utc; got: {warnings}"
        # Index still gets built (no crash) with placeholder values
        assert "Total invocations: 1" in content


def test_build_index_warns_on_malformed_file():
    """A file with no closing `---` should produce a warning."""
    with tempfile.TemporaryDirectory() as tmp:
        bad = "---\ncase_id: foo\nno closing\n"  # never closed
        sub = _make_subagent_dir(Path(tmp), {"malformed.md": bad})
        content, warnings = bil.build_index(sub)
        assert any("not closed" in w for w in warnings), \
            f"should warn about unclosed frontmatter; got: {warnings}"


def test_build_index_handles_quoted_values_with_colons():
    """The migrated entries had `date_utc: "2026-05-07T13:34:00Z"` (quotes
    + embedded colons). Parser must extract the inner value cleanly.
    """
    with tempfile.TemporaryDirectory() as tmp:
        case = '''---
case_id: test
date_utc: "2026-05-08T08:00:00Z"
trigger: "smoke (V1: bootstrap)"
verdict: approve
output_sha256: x
---
'''
        sub = _make_subagent_dir(Path(tmp), {"case.md": case})
        content, warnings = bil.build_index(sub)
        assert warnings == [], f"no warnings expected; got: {warnings}"
        # Quoted value should appear unquoted in the index row
        assert "2026-05-08T08:00:00Z" in content
        assert '"2026-05-08' not in content, "quotes should be stripped"


# Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
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
