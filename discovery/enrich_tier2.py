"""Enrich an existing discovery run with tier-2 (realistic-input) perf data.

For each trial in a run dir:
  1. Restore watched files from .original.
  2. Apply the trial's captured agent_diff.patch.
  3. Run tier-2 perf measurement.
  4. Add `perf_tier2` to the trial's result.json.

Use after a discovery run completes if tier-2 wasn't enabled at trial time
(e.g. older runs, or to add tier-2 selectively without re-paying agent cost).

Usage:
    python -m discovery.enrich_tier2 --case dbrx_moe_data_dep --run-dir <path>
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

from discovery.runner import _add_baseline_comparison, _restore_watched_files


def _split_diff_per_file(diff_text: str) -> list[tuple[str, str]]:
    """Split a unified diff into [(target_path, segment_text), ...].

    Patches with absolute paths trip GNU patch's "dangerous file name" guard,
    so we apply each per-file segment with an explicit target.
    """
    lines = diff_text.splitlines(keepends=True)
    segments: list[tuple[str, str]] = []
    current: list[str] = []
    src_path: str | None = None
    dst_path: str | None = None

    def _flush() -> None:
        if current and dst_path:
            segments.append((dst_path, "".join(current)))

    for line in lines:
        if line.startswith("--- "):
            _flush()
            current = [line]
            src_path = line[4:].split("\t", 1)[0].strip()
            dst_path = None
        elif line.startswith("+++ "):
            current.append(line)
            dst_path = line[4:].split("\t", 1)[0].strip()
        else:
            current.append(line)
    _flush()
    return segments


def _apply_diff(diff_text: str) -> tuple[bool, str]:
    """Apply a (possibly multi-file, absolute-path) diff. Returns (ok, error)."""
    for target, segment in _split_diff_per_file(diff_text):
        if not Path(target).exists():
            return False, f"target missing: {target}"
        # patch <file> reads the diff from stdin and applies it to <file>,
        # bypassing the +++/--- auto-detection (and its dangerous-path guard).
        res = subprocess.run(
            ["patch", "--silent", target],
            input=segment, text=True, capture_output=True, check=False,
        )
        if res.returncode != 0:
            return False, f"{target}: {res.stderr.strip()[:200]}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--run-dir", required=True, help="discovery run directory")
    args = parser.parse_args()

    case_mod = importlib.import_module(f"discovery.cases.{args.case}")
    case = case_mod.get_case_spec()

    if case.perf_cmd_tier2 is None:
        print(f"ERROR: case {args.case} has no perf_cmd_tier2 configured", file=sys.stderr)
        return 1

    run_dir = Path(args.run_dir)
    trial_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir())

    for trial_dir in trial_dirs:
        result_path = trial_dir / "result.json"
        diff_path = trial_dir / "agent_diff.patch"
        if not result_path.exists() or not diff_path.exists():
            continue

        result = json.loads(result_path.read_text())
        if result.get("perf_tier2"):
            print(f"  [{trial_dir.name}] already has perf_tier2 — skipping")
            continue
        if not result.get("perf") or result["perf"].get("error"):
            print(f"  [{trial_dir.name}] no clean tier-1 perf — skipping")
            continue

        print(f"  [{trial_dir.name}] applying diff + measuring tier-2 ...", flush=True)

        _restore_watched_files(case.watched_files)
        ok, err = _apply_diff(diff_path.read_text())
        if not ok:
            print(f"    patch failed: {err}", file=sys.stderr)
            _restore_watched_files(case.watched_files)
            continue

        flags: list[str] = []
        try:
            res = subprocess.run(
                case.perf_cmd_tier2, capture_output=True, text=True, timeout=600, check=False,
            )
            (trial_dir / "perf_tier2_stdout.log").write_text(res.stdout)
            (trial_dir / "perf_tier2_stderr.log").write_text(res.stderr)
            try:
                perf_tier2 = json.loads(res.stdout.strip().split("\n")[-1])
            except (json.JSONDecodeError, IndexError):
                perf_tier2 = {"raw_stdout": res.stdout, "parse_error": True}
                flags.append("perf_tier2-parse-error")
        except Exception as e:
            perf_tier2 = None
            flags.append(f"perf_tier2-crashed:{type(e).__name__}")

        _add_baseline_comparison(perf_tier2, case.baseline_path_tier2, flags)

        # Disagreement check.
        if (perf_tier2 and result.get("perf")
                and result["perf"].get("speedup") and perf_tier2.get("speedup")):
            t1_wins = result["perf"]["speedup"] > 1.0
            t2_wins = perf_tier2["speedup"] > 1.0
            if t1_wins != t2_wins:
                flags.append("tier1-tier2-direction-mismatch")

        result["perf_tier2"] = perf_tier2
        if flags:
            result["flags"] = list(set(result.get("flags", []) + flags))
        result_path.write_text(json.dumps(result, indent=2))

        if perf_tier2 and not perf_tier2.get("parse_error"):
            print(f"    tier-2: eager={perf_tier2.get('eager_ms', 0):.2f}ms "
                  f"compiled={perf_tier2.get('compiled_ms', 0):.2f}ms "
                  f"speedup={perf_tier2.get('speedup', 0):.2f}x "
                  f"vs_baseline={perf_tier2.get('compile_speedup_vs_baseline', 0):.2f}x")

        # Restore for next trial.
        _restore_watched_files(case.watched_files)

    return 0


if __name__ == "__main__":
    sys.exit(main())
