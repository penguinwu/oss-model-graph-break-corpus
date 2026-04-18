#!/usr/bin/env python3
"""Generate a nightly summary markdown file from sweep results.

Compares nightly results against the latest release sweep and produces
a markdown file suitable for results/nightly/YYYY-MM-DD.md.

Usage:
    # Auto-detect latest nightly and release
    python tools/generate_nightly_summary.py

    # Specify nightly date and release version
    python tools/generate_nightly_summary.py --date 2026-04-12 --release v2.11

    # Write to stdout instead of file
    python tools/generate_nightly_summary.py --stdout

    # Dry run (show what would be generated)
    python tools/generate_nightly_summary.py --dry-run
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_results(path):
    """Load identify_results.json into {(name, mode): status} map."""
    with open(path) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    m = {}
    for r in results:
        if isinstance(r, dict) and "name" in r:
            m[(r["name"], r.get("mode", "eval"))] = r.get("status", "unknown")
    return m, data.get("metadata", {})


def load_versions(base_dir):
    """Load versions from results metadata (embedded in identify_results.json)."""
    identify_path = os.path.join(base_dir, "identify_results.json")
    if os.path.exists(identify_path):
        with open(identify_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            versions = data.get("metadata", {}).get("versions", {})
            if versions:
                return versions
    return {}


def compute_stats(result_map):
    """Compute per-mode status counts."""
    eval_c = Counter()
    train_c = Counter()
    models = set()
    for (name, mode), status in result_map.items():
        models.add(name)
        if mode == "eval":
            eval_c[status] += 1
        elif mode == "train":
            train_c[status] += 1
    # Models with break in any mode
    model_statuses = defaultdict(dict)
    for (name, mode), status in result_map.items():
        model_statuses[name][mode] = status
    has_break = sum(1 for modes in model_statuses.values()
                    if any(s == "graph_break" for s in modes.values()))
    return eval_c, train_c, len(models), has_break


def compute_changes(release_map, nightly_map):
    """Compare release vs nightly results."""
    fixes = []       # graph_break -> full_graph
    regressions = [] # full_graph -> graph_break
    other = []       # any other change
    new_models = []

    all_keys = set(release_map.keys()) | set(nightly_map.keys())
    for key in sorted(all_keys):
        old = release_map.get(key)
        new = nightly_map.get(key)
        if old is None and new is not None:
            new_models.append((key, new))
        elif old is not None and new is not None and old != new:
            if old == "graph_break" and new == "full_graph":
                fixes.append(key)
            elif old == "full_graph" and new == "graph_break":
                regressions.append(key)
            else:
                other.append((key, old, new))

    return fixes, regressions, other, new_models


def find_latest_nightly():
    """Find the most recent nightly directory."""
    nightly_base = os.path.join(REPO_ROOT, "sweep_results", "nightly")
    if not os.path.isdir(nightly_base):
        return None
    dates = sorted(d for d in os.listdir(nightly_base)
                   if os.path.isdir(os.path.join(nightly_base, d)))
    return dates[-1] if dates else None


def find_latest_release():
    """Find the highest versioned release directory."""
    sr = os.path.join(REPO_ROOT, "sweep_results")
    import re

    def version_key(name):
        parts = re.sub(r"^pt", "", name).split(".")
        return tuple(int(p) for p in parts if p.isdigit())

    versions = sorted(
        (d for d in os.listdir(sr) if re.match(r"^pt2\.\d+$", d)),
        key=version_key,
    )
    return versions[-1] if versions else None


def generate_summary(nightly_date, release_ver):
    """Generate the markdown summary."""
    nightly_dir = os.path.join(REPO_ROOT, "sweep_results", "nightly", nightly_date)
    release_dir = os.path.join(REPO_ROOT, "sweep_results", release_ver)

    # Load static results
    nightly_static_path = os.path.join(nightly_dir, "identify_results.json")
    release_path = os.path.join(release_dir, "identify_results.json")

    if not os.path.exists(nightly_static_path):
        print(f"Error: {nightly_static_path} not found", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(release_path):
        print(f"Error: {release_path} not found", file=sys.stderr)
        sys.exit(1)

    nightly_map, nightly_meta = load_results(nightly_static_path)
    release_map, _ = load_results(release_path)

    nightly_versions = load_versions(nightly_dir)
    # Check subdirs for versions if not at top level
    if not nightly_versions:
        for subdir in ["dynamic_true", "dynamic_mark"]:
            v = load_versions(os.path.join(nightly_dir, subdir))
            if v:
                nightly_versions = v
                break

    release_versions = load_versions(release_dir)
    torch_ver = nightly_versions.get("torch", "unknown")
    tf_ver = nightly_versions.get("transformers", "unknown")
    diff_ver = nightly_versions.get("diffusers", "unknown")

    # Static stats
    eval_c, train_c, n_models, has_break = compute_stats(nightly_map)

    # Changes from release
    fixes, regressions, other, new_models = compute_changes(release_map, nightly_map)

    # Dynamic results (optional)
    dyn_true_path = os.path.join(nightly_dir, "dynamic_true", "identify_results.json")
    has_dynamic = os.path.exists(dyn_true_path)
    if has_dynamic:
        dyn_map, _ = load_results(dyn_true_path)
        dyn_eval, dyn_train, _, _ = compute_stats(dyn_map)

    # Build markdown
    lines = []
    lines.append(f"# Nightly Sweep — {nightly_date}")
    lines.append("")
    lines.append(f"**PyTorch:** {torch_ver}")
    lines.append(f"**Transformers:** {tf_ver}")
    lines.append(f"**Diffusers:** {diff_ver}")
    rel_torch = release_versions.get("torch", release_ver)
    lines.append(f"**Compared against:** PyTorch {rel_torch}")
    lines.append("")

    # Static sweep
    lines.append(f"## Static Sweep ({n_models} base models)")
    lines.append("")
    lines.append("|  | eval | train |")
    lines.append("|---|---|---|")
    for status in ["full_graph", "graph_break", "create_error", "eager_error", "timeout", "worker_error"]:
        e = eval_c.get(status, 0)
        t = train_c.get(status, 0)
        if e > 0 or t > 0:
            lines.append(f"| **{status.replace('_', '\\_')}** | {e} | {t} |")
    lines.append("")
    lines.append(f"- **{has_break} models** have graph breaks in at least one mode")
    if new_models:
        lines.append(f"- {len(new_models) // 2 if len(new_models) > 1 else len(new_models)} new model(s)")
    lines.append("")

    # Dynamic sweep
    if has_dynamic:
        lines.append(f"## Dynamic Sweep ({n_models} base models)")
        lines.append("")
        lines.append("|  | dynamic=True eval | dynamic=True train |")
        lines.append("|---|---|---|")
        lines.append(f"| **full\\_graph** | {dyn_eval.get('full_graph', 0)} | {dyn_train.get('full_graph', 0)} |")
        lines.append(f"| **graph\\_break** | {dyn_eval.get('graph_break', 0)} | {dyn_train.get('graph_break', 0)} |")
        lines.append("")
        dyn_extra_eval = dyn_eval.get("graph_break", 0) - eval_c.get("graph_break", 0)
        dyn_extra_train = dyn_train.get("graph_break", 0) - train_c.get("graph_break", 0)
        lines.append(f"Dynamic shapes add {dyn_extra_eval:+d} eval, {dyn_extra_train:+d} train graph breaks compared to static.")
        lines.append("")

    # Changes from release
    lines.append(f"## Changes from {release_ver}")
    lines.append("")

    lines.append(f"### Fixes ({len(fixes)})")
    lines.append("")
    if fixes:
        lines.append("| Model | Mode | Release | Nightly |")
        lines.append("|-------|------|---------|---------|")
        for (name, mode) in fixes:
            lines.append(f"| {name} | {mode} | graph\\_break | full\\_graph |")
    else:
        lines.append("No fixes.")
    lines.append("")

    lines.append(f"### Regressions ({len(regressions)})")
    lines.append("")
    if regressions:
        lines.append("| Model | Mode | Release | Nightly |")
        lines.append("|-------|------|---------|---------|")
        for (name, mode) in regressions:
            lines.append(f"| {name} | {mode} | full\\_graph | graph\\_break |")
    else:
        lines.append("No full\\_graph → graph\\_break regressions.")
    lines.append("")

    if other:
        lines.append(f"### Other Changes ({len(other)})")
        lines.append("")
        lines.append("| Model | Mode | Release | Nightly | Notes |")
        lines.append("|-------|------|---------|---------|-------|")
        for (name, mode), old, new in other:
            lines.append(f"| {name} | {mode} | {old.replace('_', '\\_')} | {new.replace('_', '\\_')} | |")
        lines.append("")

    if new_models:
        lines.append(f"### New Models ({len(set(n for (n, m), s in new_models))})")
        lines.append("")
        lines.append("| Model | eval | train |")
        lines.append("|-------|------|-------|")
        model_new_statuses = defaultdict(dict)
        for (name, mode), status in new_models:
            model_new_statuses[name][mode] = status
        for name in sorted(model_new_statuses):
            e = model_new_statuses[name].get("eval", "—")
            t = model_new_statuses[name].get("train", "—")
            lines.append(f"| {name} | {e.replace('_', '\\_')} | {t.replace('_', '\\_')} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate nightly sweep summary")
    parser.add_argument("--date", help="Nightly date (YYYY-MM-DD)")
    parser.add_argument("--release", help="Release version dir (e.g., v2.11)")
    parser.add_argument("--stdout", action="store_true", help="Print to stdout instead of file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be generated")
    args = parser.parse_args()

    nightly_date = args.date or find_latest_nightly()
    release_ver = args.release or find_latest_release()

    if not nightly_date:
        print("Error: no nightly results found", file=sys.stderr)
        sys.exit(1)
    if not release_ver:
        print("Error: no release results found", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"Would generate: results/nightly/{nightly_date}.md")
        print(f"  Nightly: sweep_results/nightly/{nightly_date}/")
        print(f"  Release: sweep_results/{release_ver}/")
        return

    summary = generate_summary(nightly_date, release_ver)

    if args.stdout:
        print(summary)
    else:
        out_dir = os.path.join(REPO_ROOT, "results", "nightly")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{nightly_date}.md")
        with open(out_path, "w") as f:
            f.write(summary + "\n")
        print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
