"""Pre-launch lifecycle gate enforcement for Tier-A discovery experiments.

Enforces (per discovery/EXPERIMENT_LIFECYCLE.md):
  1. discovery/smoke_test.py exits 0 (re-runs if last invocation > 1 hour old)
  2. The targeted experiment plan.md has all `## Pre-launch gates` checkboxes ticked

Bypass: `--lifecycle-bypass --reason "<text>"` writes the override reason to
the plan.md and proceeds. Auditable in git.

Usage from launcher:

    from discovery._lifecycle_gate import check_or_die
    check_or_die(plan_path, args)  # exits non-zero with explanation if gate fails
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SMOKE_TIMESTAMP_FILE = Path("/tmp/.discovery_smoke_last_pass")
SMOKE_FRESHNESS_SECONDS = 3600  # 1 hour

GATE_HEADER_RE = re.compile(r"^## Pre-launch gates", re.MULTILINE)
CHECKBOX_RE = re.compile(r"^- \[( |x|X)\] (.+)$", re.MULTILINE)


def add_lifecycle_args(parser: argparse.ArgumentParser) -> None:
    """Standard CLI args for any launcher that uses the lifecycle gate."""
    parser.add_argument(
        "--lifecycle-bypass", action="store_true",
        help="Bypass the experiment-lifecycle pre-launch gate. Requires --reason.",
    )
    parser.add_argument(
        "--reason", default=None,
        help="Reason for --lifecycle-bypass; written to plan.md as audit trail.",
    )


def _smoke_test_recently_passed() -> bool:
    """True iff smoke_test.py exited 0 within the last hour."""
    if not SMOKE_TIMESTAMP_FILE.exists():
        return False
    age = time.time() - SMOKE_TIMESTAMP_FILE.stat().st_mtime
    return age <= SMOKE_FRESHNESS_SECONDS


def _run_smoke_test() -> tuple[bool, str]:
    """Run smoke_test.py Layer 1 (cheap). Layer 2 is per-case; caller can opt in
    if they want to gate on it. Returns (passed, message)."""
    try:
        res = subprocess.run(
            ["/home/pengwu/envs/torch211/bin/python", "-m",
             "discovery.smoke_test", "--skip-cases"],
            cwd=str(REPO), capture_output=True, text=True, timeout=120,
        )
        if res.returncode == 0:
            SMOKE_TIMESTAMP_FILE.touch()
            return True, "smoke_test passed"
        return False, f"smoke_test exit {res.returncode}\n{res.stderr[-500:]}"
    except Exception as e:
        return False, f"smoke_test crashed: {type(e).__name__}: {e}"


def _check_plan_gates(plan_path: Path) -> tuple[bool, str]:
    """Verify the plan.md has a `## Pre-launch gates` section with all
    checkboxes ticked (`- [x]`). Returns (passed, message)."""
    if not plan_path.exists():
        return False, f"plan.md not found at {plan_path}"
    text = plan_path.read_text()
    if not GATE_HEADER_RE.search(text):
        return False, (
            f"plan.md has no '## Pre-launch gates' section. "
            f"Copy the template from discovery/EXPERIMENT_LIFECYCLE.md and tick gates as you advance."
        )
    # Extract just the gate section (until next ## or EOF)
    m = GATE_HEADER_RE.search(text)
    section = text[m.start():]
    # Truncate at the next ## section
    next_section = re.search(r"\n## ", section[3:])
    if next_section:
        section = section[:next_section.start() + 3]
    boxes = CHECKBOX_RE.findall(section)
    if not boxes:
        return False, "## Pre-launch gates section has no checkboxes"
    unchecked = [item for state, item in boxes if state == " "]
    if unchecked:
        return False, (
            f"{len(unchecked)} unchecked gate(s) in plan.md:\n"
            + "\n".join(f"  - [ ] {item}" for item in unchecked)
            + "\nTick the gates with `- [x]` after completing each. See discovery/EXPERIMENT_LIFECYCLE.md."
        )
    return True, f"all {len(boxes)} gates ticked"


def _record_bypass(plan_path: Path, reason: str, launcher: str) -> None:
    """Append an audit entry to the plan.md noting the bypass."""
    if not plan_path.exists():
        plan_path.write_text(f"# Plan\n\n## Lifecycle bypasses\n")
    audit_block = (
        f"\n## Lifecycle bypass — {time.strftime('%Y-%m-%d %H:%M ET', time.localtime())}\n"
        f"**Launcher:** {launcher}\n"
        f"**Reason:** {reason}\n"
    )
    with open(plan_path, "a") as f:
        f.write(audit_block)


def check_or_die(plan_path: Path, args: argparse.Namespace, launcher: str = "<unknown>") -> None:
    """Enforce the pre-launch gate. Exits 1 with explanation if the gate fails
    and no bypass was provided. Returns on success.

    Calling pattern in a launcher:
        parser = argparse.ArgumentParser()
        # ... your args ...
        add_lifecycle_args(parser)
        args = parser.parse_args()
        check_or_die(your_plan_path, args, launcher='discovery/run_case.py')
        # ... proceed to launch ...
    """
    bypass = getattr(args, "lifecycle_bypass", False)
    reason = getattr(args, "reason", None)

    if bypass:
        if not reason:
            print("ERROR: --lifecycle-bypass requires --reason.", file=sys.stderr)
            sys.exit(2)
        _record_bypass(plan_path, reason, launcher)
        print(
            f"⚠️  Lifecycle gate BYPASSED for {launcher}: {reason}\n"
            f"   Audit entry written to {plan_path}",
            file=sys.stderr,
        )
        return

    failures: list[str] = []

    # Gate 1: smoke test
    if _smoke_test_recently_passed():
        print(f"✓ smoke_test passed within last hour", file=sys.stderr)
    else:
        ok, msg = _run_smoke_test()
        if ok:
            print(f"✓ smoke_test ran fresh and passed", file=sys.stderr)
        else:
            failures.append(f"GATE 1 (smoke_test) FAILED: {msg}")

    # Gate 0/2/3: plan.md gates
    ok, msg = _check_plan_gates(plan_path)
    if ok:
        print(f"✓ plan.md gates: {msg}", file=sys.stderr)
    else:
        failures.append(f"GATE 0/2/3 (plan.md) FAILED: {msg}")

    if failures:
        print("\n=== LIFECYCLE GATE FAILED ===", file=sys.stderr)
        for f in failures:
            print(f"\n{f}", file=sys.stderr)
        print(
            f"\n→ See {REPO}/discovery/EXPERIMENT_LIFECYCLE.md for the gate template "
            f"and methodology.\n→ To bypass for emergency: --lifecycle-bypass --reason \"<text>\"",
            file=sys.stderr,
        )
        sys.exit(1)


# CLI for testing the gate directly
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Manually exercise the lifecycle gate.")
    p.add_argument("--plan", required=True, help="path to plan.md")
    add_lifecycle_args(p)
    args = p.parse_args()
    check_or_die(Path(args.plan), args, launcher="_lifecycle_gate.py (manual test)")
    print("OK — lifecycle gate would allow launch.")
