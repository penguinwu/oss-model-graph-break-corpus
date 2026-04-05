#!/usr/bin/env python3
"""Pre-generate tlparse HTML reports for the top N models by break count.

Outputs to docs/traces/<ModelName>_<mode>/ so the corpus dashboard can
link to them. After running, re-run generate_index.py to update links.

Usage:
    python3 tools/generate_traces.py                    # top 30 models
    python3 tools/generate_traces.py --top 50           # top 50
    python3 tools/generate_traces.py --skip-existing    # resume interrupted run
    python3 tools/generate_traces.py --version v2.9
    python3 tools/generate_traces.py --list             # show which models would be selected
"""
import argparse
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_ranked_models(version):
    """Rank model-mode pairs by break count from explain results."""
    explain_path = os.path.join(REPO_ROOT, "sweep_results", version, "explain_results.json")
    if not os.path.exists(explain_path):
        print(f"ERROR: Explain results not found: {explain_path}")
        sys.exit(1)

    with open(explain_path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])

    entries = []
    for r in results:
        if r.get("status") != "ok":
            continue
        bc = r.get("graph_break_count", 0)
        if bc == 0:
            continue
        entries.append({
            "name": r["name"],
            "mode": r.get("mode", "eval"),
            "break_count": bc,
            "graph_count": r.get("graph_count", 0),
            "trace_dir": f"{r['name']}_{r.get('mode', 'eval')}",
        })

    # Sort by break count descending
    entries.sort(key=lambda x: -x["break_count"])
    return entries


def run_tlparse(trace_dir, output_dir):
    """Run tlparse on a single trace directory. Returns True on success."""
    try:
        result = subprocess.run(
            ["tlparse", "parse", trace_dir, "-o", output_dir, "--overwrite"],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, "tlparse not found"


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate tlparse reports for top N models by break count",
    )
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top model-mode pairs to generate (default: 30)")
    parser.add_argument("--version", default="v2.10",
                        help="PyTorch version (default: v2.10)")
    parser.add_argument("--output-dir",
                        help="Output directory (default: docs/traces/)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip models that already have reports")
    parser.add_argument("--list", action="store_true",
                        help="List which models would be selected, don't generate")
    args = parser.parse_args()

    traces_base = os.path.join(REPO_ROOT, "sweep_results", args.version, "traces")
    output_base = args.output_dir or os.path.join(REPO_ROOT, "docs", "traces")

    ranked = get_ranked_models(args.version)

    # Filter to models that have trace directories
    available = []
    for entry in ranked:
        trace_path = os.path.join(traces_base, entry["trace_dir"])
        if os.path.isdir(trace_path):
            entry["trace_path"] = trace_path
            available.append(entry)

    selected = available[:args.top]

    if args.list:
        print(f"Top {len(selected)} model-mode pairs by break count (version {args.version}):\n")
        print(f"{'#':>3}  {'Model':<40} {'Mode':>5}  {'Breaks':>6}  {'Graphs':>6}")
        print(f"{'─' * 65}")
        for i, e in enumerate(selected, 1):
            print(f"{i:>3}  {e['name']:<40} {e['mode']:>5}  {e['break_count']:>6}  {e['graph_count']:>6}")
        print(f"\n{len(available)} total model-mode pairs with traces available")
        return

    if not selected:
        print("No models with traces found.")
        sys.exit(1)

    os.makedirs(output_base, exist_ok=True)

    print(f"Generating tlparse reports for top {len(selected)} models")
    print(f"Traces: {traces_base}")
    print(f"Output: {output_base}")
    print()

    ok = 0
    failed = 0
    skipped = 0

    for i, entry in enumerate(selected, 1):
        out_path = os.path.join(output_base, entry["trace_dir"])

        if args.skip_existing and os.path.exists(os.path.join(out_path, "index.html")):
            print(f"[{i}/{len(selected)}] {entry['trace_dir']} — skipped (exists)")
            skipped += 1
            continue

        print(f"[{i}/{len(selected)}] {entry['trace_dir']} ({entry['break_count']} breaks) ...", end=" ", flush=True)
        success, err = run_tlparse(entry["trace_path"], out_path)
        if success:
            print("OK")
            ok += 1
        else:
            print(f"FAILED: {err[:100]}")
            failed += 1

    print(f"\nDone: {ok} generated, {skipped} skipped, {failed} failed")
    print(f"\nNext: run 'python3 tools/generate_index.py' to update dashboard links")


if __name__ == "__main__":
    main()
