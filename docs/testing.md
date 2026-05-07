# Testing

This project uses **test-driven development** as a primary correctness guardrail. Strong testing is non-negotiable. The discipline:

> **Every revision of tooling, especially bug fixes, MUST add or extend test cases that prove the change works as expected.**

This is not aspirational. It is the rule. Tools that get changed without tests are revisions that bypass the project's primary correctness guardrail.

---

## When tests are required

| Change type | Tests required |
|---|---|
| **Bug fix in a tool or sweep module** | Failing test demonstrating the bug, then passing test after fix. The test stays in the repo as a regression guard. |
| **New tool added to `tools/`** | A `tools/test_<tool_name>.py` covering filter/parser logic, output shape, edge cases, and at least one realistic end-to-end scenario. |
| **New flag added to an existing tool** | Tests covering each accepted value of the flag (e.g., `--pass identify` AND `--pass explain`). |
| **Behavior change to a sweep module (`sweep/*.py`)** | Tests in `sweep/test_*.py` or `tools/test_*.py` covering the new behavior. Plus the 5-gate workflow in `skills/test-sweep-changes/SKILL.md`. |
| **Schema or contract change (config files, result formats)** | Round-trip tests (write then read; read then write) demonstrating compatibility. |
| **Refactor with no behavior change** | Existing tests must continue to pass; no new tests required (but allowed). |

## Test-file conventions

- **Location:** alongside the code under test. `tools/foo.py` → `tools/test_foo.py`. `sweep/bar.py` → `sweep/test_bar.py`.
- **Naming:** `test_<module>.py` for the file; functions or methods named `test_<behavior>`.
- **Style:** plain `assert` statements (with descriptive messages on failure) OR `unittest.TestCase`. Both are acceptable; pick what fits the file. Plain functions are simpler for new test files; `unittest` is preferred when you need fixtures (setUp/tearDown).
- **Runner:** every test file must be runnable directly: `python3 tools/test_foo.py`. Exit non-zero on any failure. This makes the tests cron-friendly and CI-friendly without depending on a test framework discovery convention.
- **No external dependencies:** tests should not require GPU, network, or large fixtures. Use `tempfile.TemporaryDirectory()` + small synthetic inputs. If a test needs a real artifact (e.g., a historical sweep result), guard it with `if not path.is_file(): print("SKIP ..."); return`.

## Bug-fix workflow (failing-test-first)

Per TDD discipline, bug fixes follow this sequence:

1. **Reproduce** the bug locally. Capture the failure mode (error message, wrong output, etc.).
2. **Write a failing test** that asserts the correct behavior. Run the test. It must fail with a clear assertion message that matches the captured failure mode.
3. **Fix the bug** in the code under test. Run the test. It must pass.
4. **Run the full test file** to ensure no regression in adjacent behavior.
5. **Commit both together:** the failing-test-then-passing-test commit pattern, OR a single commit with the test added alongside the fix. The commit message should reference the bug.

The test stays in the repo as a regression guard. If a future change reintroduces the bug, the test fails immediately.

## Test catalog (what's covered today)

| Test file | Covers |
|---|---|
| `tools/test_generate_cohort.py` | Filter parser correctness (`==`, `!=`, `in` with single + multi values), output `_metadata` shape, dedupe across modes, force-overwrite behavior, error paths, regression guard for the 2026-05-06 broken-cohort bug |
| `tools/test_feedback_monitor.py` | `load_state` migration of old-format state files, `mark_replied` dedup, `list_needs_reply` audit-log filtering + dedupe + malformed-line resilience. Backfilled for the May-4 incident bug fix. |
| `tools/test_sweep_watchdog_check.py` | `decide_verdict` logic across all PID/progress/grace combinations, the 2026-04-30 PID-change-reset bug regression guard, `--pass identify\|explain` flag selection of checkpoint+results files |
| `sweep/test_explain.py` | Explain pass result schema |
| `sweep/test_results_loader.py` | Sweep result file loading across schema variants |
| `sweep/test_venv_setup.py` | Venv resolution + provisioning |

## How to add a new test file

Template — copy this and fill in:

```python
#!/usr/bin/env python3
"""Tests for <module>.

[One-paragraph description of what's covered.]

Run: python3 <path>/test_<module>.py
Exit non-zero on any failure.
"""
import sys
from pathlib import Path

# Import-under-test
sys.path.insert(0, str(Path(__file__).resolve().parent))
import <module> as m  # noqa: E402


def test_<behavior_1>():
    ...


def test_<behavior_2>():
    ...


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
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
```

## What enforces this

Today: discipline + commit-message review + this doc. **Open loop:** a pre-commit hook or CI step that runs all `test_*.py` files and fails the commit if any test fails or if a tool change has no corresponding test diff.

Until that's mechanized, the rule is: **PRs without tests for behavior changes will be reverted.** No exceptions for "small fixes" — that's exactly the size of fix that breaks silently when reintroduced.

## Why this matters (the cautionary tale)

2026-05-06 NGB verify ran 3 hours with a broken cohort because:
- `tools/generate_cohort.py` didn't exist (cohort hand-rolled instead)
- No tests existed for cohort-generation correctness
- No tests existed for the cohort-loading path's `_metadata` handling
- Bug fixes throughout the day landed without regression tests

Every one of those cost more than the test would have. The skill tells us "smoke pre-flight catches symptoms;" tests catch root causes BEFORE the symptoms reach a sweep launch.
