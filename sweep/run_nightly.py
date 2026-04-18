#!/usr/bin/env python3
"""Nightly regression sweep — full pipeline with targeted explain.

Architecture (3-step):
    Step 0: Refresh nightly venv to latest PyTorch nightly
    Step 1: Identify all models (static, detect regressions/fixes)
    Step 2: Targeted explain on models whose status changed or that we're tracking
    Step 3: Corpus update + summary

Usage:
    # Full nightly pipeline (default: static, hf+diffusers+custom)
    SWEEP_PYTHON=/home/pengwu/envs/torch-nightly/bin/python python sweep/run_nightly.py

    # Skip venv refresh (already up to date)
    SWEEP_PYTHON=... python sweep/run_nightly.py --skip-refresh

    # Include dynamic variants (longer runtime)
    SWEEP_PYTHON=... python sweep/run_nightly.py --include-dynamic
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
RUN_EXPERIMENT = TOOLS_DIR / "run_experiment.py"
CORPUS_PATH = REPO_ROOT / "corpus" / "corpus.json"

TRACKED_MODELS_FILE = REPO_ROOT / "sweep" / "tracked_models.json"


def _run(cmd, desc, log_file=None, allow_fail=False):
    """Run a subprocess, optionally logging to file."""
    print(f"\n{'=' * 60}")
    print(f"NIGHTLY: {desc}")
    print(f"{'=' * 60}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    if log_file:
        print(f"  LOG: {log_file}")
        with open(log_file, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=str(REPO_ROOT))
    else:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if not allow_fail:
            sys.exit(1)
    return result.returncode == 0


def step0_refresh_venv(python, venv_dir, force=False):
    """Refresh nightly venv to latest PyTorch nightly build.

    Returns (version_string, git_version, is_stale).
    is_stale=True means the nightly build is identical to the last sweep.
    """
    old_ver = _get_torch_version(python)
    print(f"  Current: {old_ver}")

    ok = _run(
        [sys.executable, str(RUN_EXPERIMENT), "refresh-nightly",
         "--venv", str(venv_dir), "--with-transformers"],
        "Refresh nightly venv",
    )

    new_ver = _get_torch_version(python)
    git_ver = _get_torch_git_version(python)
    print(f"  After refresh: {new_ver} (git: {git_ver[:12]})")

    if old_ver != new_ver:
        print(f"  Updated: {old_ver} → {new_ver}")

    last_git = _get_last_sweep_git_version()
    is_stale = False
    if last_git and git_ver == last_git:
        is_stale = True
        print(f"\n  *** STALE NIGHTLY DETECTED ***")
        print(f"  Git commit {git_ver[:12]} is identical to last sweep.")
        print(f"  No new PyTorch changes to test.")
        if force:
            print(f"  --force specified — sweeping anyway.")
        else:
            print(f"  Aborting to avoid wasting a weekly sweep.")
            print(f"  Use --force to override.")
    elif last_git:
        print(f"  Last sweep git: {last_git[:12]} → current: {git_ver[:12]} (new commits)")

    return new_ver, git_ver, is_stale


def _get_torch_version(python):
    """Get torch version from a python executable."""
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _get_torch_git_version(python):
    """Get torch git commit hash from a python executable."""
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.version.git_version)"],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _get_last_sweep_git_version():
    """Get git_version from the most recent nightly sweep."""
    nightly_dir = REPO_ROOT / "sweep_results" / "nightly"
    if not nightly_dir.exists():
        return None
    dates = sorted(d.name for d in nightly_dir.iterdir()
                   if d.is_dir() and d.name[:4].isdigit())
    if not dates:
        return None
    last = nightly_dir / dates[-1] / "versions.json"
    if not last.exists():
        return None
    with open(last) as f:
        data = json.load(f)
    return data.get("git_version") or data.get("torch_git")


def step1_identify(python, output_dir, args):
    """Run identify pass on all models. Returns path to results file."""
    phases = [
        {
            "name": "Static (hf + diffusers + custom)",
            "subdir": "",
            "args": ["--source", "hf", "diffusers", "custom"],
        },
    ]

    if args.include_dynamic:
        phases.extend([
            {
                "name": "Dynamic-all (hf + diffusers)",
                "subdir": "dynamic_true",
                "args": ["--source", "hf", "diffusers", "--dynamic-dim", "all"],
            },
            {
                "name": "Dynamic-batch (hf + diffusers)",
                "subdir": "dynamic_mark",
                "args": ["--source", "hf", "diffusers", "--dynamic-dim", "batch"],
            },
        ])

    failed = []
    static_results_dir = None

    for i, phase in enumerate(phases, 1):
        phase_dir = output_dir / phase["subdir"] if phase["subdir"] else output_dir
        phase_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"phase{i}_identify.log"

        cmd = [
            python, str(REPO_ROOT / "sweep" / "run_sweep.py"), "sweep",
            "--device", args.device,
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--output-dir", str(phase_dir),
            "--identify-only",
            *phase["args"],
        ]
        if args.resume:
            cmd.append("--resume")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

        start = time.perf_counter()
        ok = _run(cmd, f"Identify {i}/{len(phases)}: {phase['name']}", log_file,
                  allow_fail=True)
        elapsed = time.perf_counter() - start

        if ok:
            print(f"  Done in {elapsed:.0f}s")
        else:
            print(f"  FAILED after {elapsed:.0f}s — see {log_file}")
            failed.append(phase["name"])

        if i == 1:
            static_results_dir = phase_dir

    if failed:
        print(f"\n  WARNING: {len(failed)} phase(s) failed: {', '.join(failed)}")

    return static_results_dir


def step2_detect_and_explain(python, results_dir, args):
    """Detect status changes vs corpus, run explain on changed + tracked models."""
    identify_file = results_dir / "identify_results.json"
    if not identify_file.exists():
        print(f"  ERROR: {identify_file} not found — skipping explain")
        return

    # Load identify results
    with open(identify_file) as f:
        identify_data = json.load(f)
    new_results = {(r["name"], r["mode"]): r["status"]
                   for r in identify_data.get("results", [])}

    # Load current corpus
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    old_results = {}
    for m in corpus["models"]:
        for mode in ("eval", "train"):
            status = m.get(mode, {}).get("status")
            if status:
                old_results[(m["name"], mode)] = status

    # Detect changes
    changed = set()
    for key, new_status in new_results.items():
        old_status = old_results.get(key)
        if old_status and old_status != new_status:
            changed.add(key[0])  # model name
    for key in new_results:
        if key not in old_results:
            changed.add(key[0])

    # Load tracked models (models we're specifically monitoring)
    tracked = set()
    if TRACKED_MODELS_FILE.exists():
        with open(TRACKED_MODELS_FILE) as f:
            tracked_data = json.load(f)
        tracked = set(tracked_data.get("models", []))

    explain_set = changed | tracked
    explain_models = [r for r in identify_data.get("results", [])
                      if r["name"] in explain_set
                      and r.get("status") in ("graph_break", "compile_error")]

    print(f"\n  Status changes detected: {len(changed)} models")
    if changed:
        for name in sorted(changed)[:10]:
            old = old_results.get((name, "eval"), "?")
            new = new_results.get((name, "eval"), "?")
            print(f"    {name}: {old} → {new}")
        if len(changed) > 10:
            print(f"    ... and {len(changed) - 10} more")
    print(f"  Tracked models: {len(tracked)}")
    print(f"  Models needing explain: {len(explain_models)}")

    if not explain_models:
        print("  No models need explain — skipping")
        return

    # Write filtered identify results for explain pass
    explain_input = results_dir / "explain_targets.json"
    explain_data = {"metadata": identify_data.get("metadata", {}),
                    "results": explain_models}
    with open(explain_input, "w") as f:
        json.dump(explain_data, f, indent=2)

    # Run explain pass
    explain_dir = results_dir / "explain"
    explain_dir.mkdir(exist_ok=True)
    log_file = results_dir / "explain.log"

    cmd = [
        python, str(RUN_EXPERIMENT), "explain",
        str(explain_input),
        "--device", args.device,
        "--workers", str(args.workers),
        "--timeout", str(args.timeout),
        "--output-dir", str(explain_dir),
    ]

    start = time.perf_counter()
    ok = _run(cmd, f"Targeted explain ({len(explain_models)} models)", log_file,
              allow_fail=True)
    elapsed = time.perf_counter() - start

    if ok:
        print(f"  Done in {elapsed:.0f}s")
    else:
        print(f"  Explain failed after {elapsed:.0f}s — see {log_file}")


def step3_corpus_and_summary(python, results_dir, nightly_version, args):
    """Update corpus from results and generate summary."""
    # Build corpus (overlay mode — preserves existing metadata per our fix)
    identify_file = results_dir / "identify_results.json"
    if not identify_file.exists():
        print("  ERROR: No identify results — skipping corpus update")
        return

    # Convert identify_results.json to results.jsonl format for corpus subcommand
    with open(identify_file) as f:
        data = json.load(f)
    results_jsonl = results_dir / "results.jsonl"
    with open(results_jsonl, "w") as f:
        for r in data.get("results", []):
            f.write(json.dumps(r) + "\n")

    # Check if explain results exist
    explain_dir = results_dir / "explain"
    explain_flag = []
    explain_results = explain_dir / "results.jsonl"
    if explain_results.exists():
        explain_flag = ["--explain", str(explain_dir)]

    cmd = [
        sys.executable, str(RUN_EXPERIMENT), "corpus",
        str(results_dir), *explain_flag,
    ]
    ok = _run(cmd, "Corpus update (overlay mode)")

    if not ok:
        print("  Corpus update failed")
        return

    # Generate changelog
    changelog = results_dir / "changelog.md"
    summary_file = results_dir / "summary.txt"

    # Build nightly summary
    _generate_summary(results_dir, nightly_version, summary_file)

    print(f"\n  Summary: {summary_file}")
    if summary_file.exists():
        print(summary_file.read_text())


def _generate_summary(results_dir, nightly_version, summary_file):
    """Generate a human-readable nightly summary."""
    identify_file = results_dir / "identify_results.json"
    if not identify_file.exists():
        return

    with open(identify_file) as f:
        data = json.load(f)
    results = data.get("results", [])

    # Count statuses
    from collections import Counter
    status_counts = Counter()
    for r in results:
        status_counts[r.get("status", "unknown")] += 1

    # Load corpus for comparison
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    old_results = {}
    for m in corpus["models"]:
        for mode in ("eval", "train"):
            status = m.get(mode, {}).get("status")
            if status:
                old_results[(m["name"], mode)] = status

    # Find changes
    regressions = []
    fixes = []
    for r in results:
        key = (r["name"], r["mode"])
        old = old_results.get(key)
        new = r.get("status")
        if not old or old == new:
            continue
        if old == "full_graph" and new != "full_graph":
            regressions.append((r["name"], r["mode"], old, new))
        elif old != "full_graph" and new == "full_graph":
            fixes.append((r["name"], r["mode"], old, new))

    lines = [
        f"Nightly Sweep Summary — {time.strftime('%Y-%m-%d')}",
        f"PyTorch: {nightly_version}",
        f"Models tested: {len(results)}",
        "",
        "Status breakdown:",
    ]
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {status}: {count}")

    if fixes:
        lines.append(f"\nFixes ({len(fixes)}):")
        for name, mode, old, new in fixes:
            lines.append(f"  {name} ({mode}): {old} → {new}")
    if regressions:
        lines.append(f"\nRegressions ({len(regressions)}):")
        for name, mode, old, new in regressions:
            lines.append(f"  {name} ({mode}): {old} → {new}")
    if not fixes and not regressions:
        lines.append("\nNo status changes from corpus.")

    text = "\n".join(lines) + "\n"
    summary_file.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Nightly regression sweep (full pipeline)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output-dir", default=None,
                        help="Base output dir (default: sweep_results/nightly/YYYY-MM-DD)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-refresh", action="store_true",
                        help="Skip venv refresh (use current nightly as-is)")
    parser.add_argument("--include-dynamic", action="store_true",
                        help="Include dynamic variant phases (longer runtime)")
    parser.add_argument("--skip-explain", action="store_true",
                        help="Skip targeted explain pass")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip corpus update")
    parser.add_argument("--venv", default=os.path.expanduser("~/envs/torch-nightly"),
                        help="Nightly venv path (default: ~/envs/torch-nightly)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max models to test (0 = all, for smoke testing)")
    parser.add_argument("--force", action="store_true",
                        help="Run sweep even if nightly is unchanged from last sweep")
    args = parser.parse_args()

    python = os.environ.get("SWEEP_PYTHON",
                            str(Path(args.venv) / "bin" / "python"))
    date = time.strftime("%Y-%m-%d")
    output_dir = Path(args.output_dir) if args.output_dir else (
        REPO_ROOT / "sweep_results" / "nightly" / date)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Nightly Sweep Pipeline {date} ===")
    print(f"Python: {python}")
    print(f"Output: {output_dir}")
    print(f"Start:  {time.strftime('%H:%M:%S')}")

    # Step 0: Refresh venv
    git_version = None
    if not args.skip_refresh:
        nightly_version, git_version, is_stale = step0_refresh_venv(
            python, args.venv, force=args.force)
        if is_stale and not args.force:
            print(f"\n=== Sweep aborted: nightly unchanged since last sweep ===")
            sys.exit(0)
    else:
        nightly_version = _get_torch_version(python)
        git_version = _get_torch_git_version(python)
        print(f"\nSkipping venv refresh (current: {nightly_version}, git: {git_version[:12]})")

    # Save version info
    versions_file = output_dir / "versions.json"
    versions_file.write_text(json.dumps({
        "pytorch": nightly_version,
        "git_version": git_version,
        "date": date,
        "python": python,
        "refreshed": not args.skip_refresh,
    }, indent=2))

    # Step 1: Identify
    results_dir = step1_identify(python, output_dir, args)

    # Step 2: Detect changes + targeted explain
    if not args.skip_explain and results_dir:
        step2_detect_and_explain(python, results_dir, args)
    elif args.skip_explain:
        print("\nSkipping explain pass (--skip-explain)")

    # Step 3: Corpus update + summary
    if not args.skip_corpus and results_dir:
        step3_corpus_and_summary(python, results_dir, nightly_version, args)
    elif args.skip_corpus:
        print("\nSkipping corpus update (--skip-corpus)")

    # Done
    elapsed_total = time.strftime('%H:%M:%S')
    print(f"\n{'=' * 60}")
    print(f"Nightly sweep pipeline complete")
    print(f"  Results: {output_dir}")
    print(f"  Version: {nightly_version}")
    print(f"  End: {elapsed_total}")

    summary = output_dir / "summary.txt"
    if summary.exists():
        print(f"\n{summary.read_text()}")


if __name__ == "__main__":
    main()
