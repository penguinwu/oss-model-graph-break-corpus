#!/usr/bin/env python3
"""Nightly regression sweep — runs all sources and dynamic variants.

Orchestrates four run_sweep.py phases:
    1. Static identify (hf + diffusers)
    2. Dynamic-all identify (hf + diffusers)
    3. Dynamic-batch identify (hf + diffusers)
    4. Custom models identify

Usage:
    python sweep/run_nightly.py
    python sweep/run_nightly.py --workers 8 --output-dir sweep_results/nightly/2026-04-15
    SWEEP_PYTHON=/path/to/python python sweep/run_nightly.py
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SWEEP_SCRIPT = Path(__file__).parent / "run_sweep.py"
SUMMARY_SCRIPT = Path(__file__).parent.parent / "tools" / "generate_nightly_summary.py"

PHASES = [
    {
        "name": "Static (hf + diffusers)",
        "subdir": "",
        "args": ["--source", "hf", "diffusers", "--identify-only"],
    },
    {
        "name": "Dynamic-all (hf + diffusers)",
        "subdir": "dynamic_true",
        "args": ["--source", "hf", "diffusers", "--identify-only", "--dynamic-dim", "all"],
    },
    {
        "name": "Dynamic-batch (hf + diffusers)",
        "subdir": "dynamic_mark",
        "args": ["--source", "hf", "diffusers", "--identify-only", "--dynamic-dim", "batch"],
    },
    {
        "name": "Custom models",
        "subdir": "custom",
        "args": ["--source", "custom", "--identify-only"],
    },
]


def main():
    parser = argparse.ArgumentParser(description="Nightly regression sweep")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", default=None,
                        help="Base output dir (default: sweep_results/nightly/YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    python = os.environ.get("SWEEP_PYTHON", sys.executable)
    date = time.strftime("%Y-%m-%d")
    base_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent.parent / "sweep_results" / "nightly" / date)

    print(f"=== Nightly sweep {date} ===")
    print(f"Python: {python}")
    print(f"Output: {base_dir}")
    print(f"Start:  {time.strftime('%H:%M:%S')}")

    failed = []
    for i, phase in enumerate(PHASES, 1):
        phase_dir = base_dir / phase["subdir"] if phase["subdir"] else base_dir
        phase_dir.mkdir(parents=True, exist_ok=True)
        log_file = base_dir / f"phase{i}.log"

        print(f"\n{'=' * 60}")
        print(f"Phase {i}/{len(PHASES)}: {phase['name']}")
        print(f"Output: {phase_dir}")
        print(f"{'=' * 60}")

        cmd = [
            python, str(SWEEP_SCRIPT), "sweep",
            "--device", args.device,
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--output-dir", str(phase_dir),
            *phase["args"],
        ]
        if args.resume:
            cmd.append("--resume")

        start = time.perf_counter()
        with open(log_file, "w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - start

        if result.returncode == 0:
            print(f"  Done in {elapsed:.0f}s")
        else:
            print(f"  FAILED (exit {result.returncode}) after {elapsed:.0f}s — see {log_file}")
            failed.append(phase["name"])

    # Generate summary
    if SUMMARY_SCRIPT.exists():
        print(f"\nGenerating nightly summary...")
        subprocess.run([sys.executable, str(SUMMARY_SCRIPT), "--date", date])

    print(f"\n{'=' * 60}")
    if failed:
        print(f"Nightly sweep completed with {len(failed)} failures:")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)
    else:
        print(f"Nightly sweep completed successfully ({len(PHASES)} phases)")
    print(f"Results: {base_dir}")
    print(f"End: {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
