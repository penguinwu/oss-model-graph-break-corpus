#!/usr/bin/env python3
"""Analyze graph break trends across PyTorch versions.

Compares identify (model counts) and explain (total break counts) across multiple
PyTorch versions. Normalizes by testable models to avoid confounding from
changing eager_error/create_error rates.

Usage:
    # Default: auto-discover pt2.* directories
    python tools/analyze_trend.py

    # Custom version dirs
    python tools/analyze_trend.py sweep_results/pt2.8 sweep_results/pt2.9 --labels "2.8" "2.9"

    # Include train mode
    python tools/analyze_trend.py --train

    # Generate markdown version trend table
    python tools/analyze_trend.py --markdown

    # Validate results/*.md against computed data
    python tools/analyze_trend.py --validate

    # JSON output
    python tools/analyze_trend.py --json

    # Show per-model details for fixed/regressed models
    python tools/analyze_trend.py --details
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from corpus_constants import (
    REPO_ROOT, SWEEP_RESULTS_DIR, RESULTS_DIR,
    VERSION_DIR_PATTERN, FIX_TRANSITION, REGRESSION_TRANSITION,
    find_version_dirs as _find_version_dirs,
)


def load_identify(path):
    """Load identify results into {name: {mode: status}}."""
    from pathlib import Path as _Path
    p = _Path(path)
    d = defaultdict(dict)
    if p.name == "corpus.json":
        with open(path) as f:
            data = json.load(f)
        for m in data["models"]:
            for mode in ("eval", "train"):
                if mode in m:
                    d[m["name"]][mode] = m[mode].get("status", "unknown")
        return d
    # identify_results.json — route through canonical loader (merges amendments)
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from sweep.results_loader import load_results_list
    for r in load_results_list(path):
        d[r["name"]][r.get("mode", "eval")] = r.get("status", "unknown")
    return d


def load_explain(path):
    """Load explain results into {name: {mode: result_dict}}."""
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    d = defaultdict(dict)
    results = data if isinstance(data, list) else data.get("results", [])
    for r in results:
        d[r["name"]][r.get("mode", "eval")] = r
    return d


def find_version_dirs_from_args(args):
    """Find version directories and their data files."""
    versions = []

    if args.dirs:
        for i, d in enumerate(args.dirs):
            label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(d)
            p1 = os.path.join(d, "identify_results.json")
            p2 = os.path.join(d, "explain_results.json")
            if not os.path.exists(p1):
                print(f"ERROR: {p1} not found", file=sys.stderr)
                sys.exit(1)
            versions.append({"label": label, "identify": p1, "explain": p2, "dir": d})
    else:
        for label, path in _find_version_dirs():
            versions.append({
                "label": label,
                "identify": os.path.join(path, "identify_results.json"),
                "explain": os.path.join(path, "explain_results.json"),
                "dir": path,
            })

    return versions


def analyze_mode(versions_data, mode, common_models):
    """Analyze a single mode (eval or train) across versions."""
    results = []

    for vd in versions_data:
        p1 = vd["identify_data"]
        p2 = vd.get("explain_data")

        # Identify stats
        statuses = Counter(p1[n].get(mode, "missing") for n in common_models)
        testable = statuses.get("full_graph", 0) + statuses.get("graph_break", 0)

        # Explain stats (if available)
        total_breaks = 0
        explain_ok = 0
        explain_error = 0
        broken_with_breaks = 0

        if p2:
            for n in common_models:
                if n in p2 and mode in p2[n]:
                    r = p2[n][mode]
                    if r.get("status") == "ok":
                        explain_ok += 1
                        bc = r.get("graph_break_count", 0)
                        total_breaks += bc
                        if bc > 0:
                            broken_with_breaks += 1
                    elif r.get("status") == "explain_error":
                        explain_error += 1

        results.append({
            "label": vd["label"],
            "mode": mode,
            "total_models": len(common_models),
            "full_graph": statuses.get("full_graph", 0),
            "graph_break": statuses.get("graph_break", 0),
            "eager_error": statuses.get("eager_error", 0),
            "create_error": statuses.get("create_error", 0),
            "timeout": statuses.get("timeout", 0),
            "testable": testable,
            "total_graph_breaks": total_breaks if p2 else None,
            "explain_ok": explain_ok if p2 else None,
            "explain_error": explain_error if p2 else None,
            "avg_breaks_per_broken": round(total_breaks / statuses.get("graph_break", 1), 1) if statuses.get("graph_break", 0) > 0 else None,
            "has_explain": p2 is not None,
        })

    return results


def find_transitions(versions_data, mode, common_models):
    """Find models that changed status between first and last version."""
    if len(versions_data) < 2:
        return {}

    first = versions_data[0]["identify_data"]
    last = versions_data[-1]["identify_data"]

    transitions = defaultdict(list)
    for n in sorted(common_models):
        s_first = first[n].get(mode, "missing")
        s_last = last[n].get(mode, "missing")
        if s_first != s_last:
            # Get intermediate statuses
            path = [versions_data[i]["identify_data"][n].get(mode, "?") for i in range(len(versions_data))]
            transitions[f"{s_first} → {s_last}"].append((n, path))

    return transitions


def find_apples_to_apples(versions_data, mode, common_models):
    """Find models broken in ALL versions for normalized break count comparison."""
    broken_in_all = set(common_models)
    for vd in versions_data:
        broken_in_all = {n for n in broken_in_all if vd["identify_data"][n].get(mode) == "graph_break"}
    return broken_in_all


def print_report(versions_data, modes, common_models, show_details=False):
    """Print the full trend report."""
    labels = [vd["label"] for vd in versions_data]
    has_any_explain = any(vd.get("explain_data") for vd in versions_data)

    for mode in modes:
        stats = analyze_mode(versions_data, mode, common_models)

        print(f"\n{'='*70}")
        print(f"VERSION TREND — {mode.upper()} MODE ({len(common_models)} models)")
        print(f"{'='*70}\n")

        # Header
        col_w = max(len(l) for l in labels) + 2
        header = f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels)
        print(header)
        print("-" * len(header))

        # Rows
        rows = [
            ("Full graph", "full_graph"),
            ("Graph break", "graph_break"),
            ("Eager error", "eager_error"),
            ("Create error", "create_error"),
            ("Timeout", "timeout"),
        ]
        for row_label, key in rows:
            vals = "".join(f"{s[key]:>{col_w}}" for s in stats)
            print(f"{row_label:<35}{vals}")

        print(f"{'---':<35}" + "".join(f"{'---':>{col_w}}" for _ in labels))
        vals = "".join(f"{s['testable']:>{col_w}}" for s in stats)
        print(f"{'Testable (full_graph + graph_break)':<35}{vals}")

        if has_any_explain:
            print()
            vals = "".join(f"{s['total_graph_breaks'] if s['total_graph_breaks'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Total graph breaks (explain)':<35}{vals}")
            vals = "".join(f"{s['explain_ok'] if s['explain_ok'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Explain OK':<35}{vals}")
            vals = "".join(f"{s['explain_error'] if s['explain_error'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Explain error':<35}{vals}")
            vals = "".join(f"{s['avg_breaks_per_broken'] if s['avg_breaks_per_broken'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Avg breaks per broken model':<35}{vals}")

        # Apples-to-apples comparison
        if has_any_explain:
            a2a = find_apples_to_apples(versions_data, mode, common_models)
            if a2a:
                print(f"\n{'─'*70}")
                print(f"APPLES-TO-APPLES: {len(a2a)} models broken in ALL versions ({mode})")
                print(f"{'─'*70}\n")

                a2a_header = f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels)
                print(a2a_header)
                print("-" * len(a2a_header))

                for vd in versions_data:
                    p2 = vd.get("explain_data")
                    if not p2:
                        continue

                a2a_breaks = []
                a2a_ok = []
                a2a_err = []
                for vd in versions_data:
                    p2 = vd.get("explain_data")
                    if not p2:
                        a2a_breaks.append(None)
                        a2a_ok.append(None)
                        a2a_err.append(None)
                        continue
                    tb = 0
                    ok = 0
                    err = 0
                    for n in a2a:
                        if n in p2 and mode in p2[n]:
                            r = p2[n][mode]
                            if r.get("status") == "ok":
                                ok += 1
                                tb += r.get("graph_break_count", 0)
                            elif r.get("status") == "explain_error":
                                err += 1
                    a2a_breaks.append(tb)
                    a2a_ok.append(ok)
                    a2a_err.append(err)

                vals = "".join(f"{b if b is not None else 'N/A':>{col_w}}" for b in a2a_breaks)
                print(f"{'Total graph breaks':<35}{vals}")
                vals = "".join(f"{o if o is not None else 'N/A':>{col_w}}" for o in a2a_ok)
                print(f"{'Explain OK':<35}{vals}")
                vals = "".join(f"{e if e is not None else 'N/A':>{col_w}}" for e in a2a_err)
                print(f"{'Explain error':<35}{vals}")
                if any(b is not None for b in a2a_breaks):
                    num_a2a = len(a2a)
                    vals = "".join(
                        f"{round(b/num_a2a, 1) if b is not None else 'N/A':>{col_w}}"
                        for b in a2a_breaks
                    )
                    print(f"{'Avg breaks per model':<35}{vals}")

        # Transitions
        transitions = find_transitions(versions_data, mode, common_models)
        if transitions:
            print(f"\n{'─'*70}")
            print(f"STATUS TRANSITIONS: {labels[0]} → {labels[-1]} ({mode})")
            print(f"{'─'*70}\n")

            for transition, models in sorted(transitions.items(), key=lambda x: -len(x[1])):
                print(f"  {transition}: {len(models)} models")
                if show_details:
                    for name, path in models:
                        path_str = " → ".join(path)
                        print(f"    {name}: {path_str}")

        # Fixed models detail
        if show_details and has_any_explain:
            fix_label = f"{FIX_TRANSITION[0]} → {FIX_TRANSITION[1]}"
            fixed = []
            for t, models in transitions.items():
                if fix_label in t:
                    fixed.extend(models)
            if fixed:
                print(f"\n{'─'*70}")
                print(f"FIXED MODELS — break counts in {labels[0]} ({mode})")
                print(f"{'─'*70}\n")
                p2_first = versions_data[0].get("explain_data")
                if p2_first:
                    for name, path in sorted(fixed):
                        r = p2_first.get(name, {}).get(mode, {})
                        bc = r.get("graph_break_count", "?")
                        print(f"  {name}: {bc} breaks")


def count_fixes(versions_data, mode, common_models):
    """Count graph_break->full_graph transitions between consecutive versions."""
    fixes_per_version = []
    regressions_per_version = []
    for i in range(len(versions_data)):
        if i == 0:
            fixes_per_version.append(None)
            regressions_per_version.append(None)
            continue
        prev = versions_data[i - 1]["identify_data"]
        curr = versions_data[i]["identify_data"]
        fixes = 0
        regressions = 0
        for n in common_models:
            old = prev[n].get(mode, "missing")
            new = curr[n].get(mode, "missing")
            if (old, new) == FIX_TRANSITION:
                fixes += 1
            elif (old, new) == REGRESSION_TRANSITION:
                regressions += 1
        fixes_per_version.append(fixes)
        regressions_per_version.append(regressions)
    return fixes_per_version, regressions_per_version


def generate_markdown(versions_data, common_models):
    """Generate the markdown version trend table for results files."""
    lines = []
    lines.append(f"## Version Trend ({len(common_models)} common models)\n")
    lines.append("| Version | eval full\\_graph | train full\\_graph | eval break | train break | Fixes | Regressions |")
    lines.append("|---------|-----------------|-------------------|------------|-------------|-------|-------------|")

    eval_stats = analyze_mode(versions_data, "eval", common_models)
    train_stats = analyze_mode(versions_data, "train", common_models)
    eval_fixes, eval_regs = count_fixes(versions_data, "eval", common_models)

    for i, vd in enumerate(versions_data):
        label = vd["label"].replace("pt", "")
        e = eval_stats[i]
        t = train_stats[i]
        e_pct = round(100 * e["full_graph"] / len(common_models))
        t_pct = round(100 * t["full_graph"] / len(common_models))
        fixes = eval_fixes[i]
        regs = eval_regs[i]
        fix_str = "--" if fixes is None else str(fixes)
        reg_str = "--" if regs is None else str(regs)
        lines.append(
            f"| {label} | {e['full_graph']} ({e_pct}%) | {t['full_graph']} ({t_pct}%) "
            f"| {e['graph_break']} | {t['graph_break']} | {fix_str} | {reg_str} |"
        )

    return "\n".join(lines)


def validate_results(versions_data, common_models):
    """Validate that results/*.md version trend numbers match computed data."""
    eval_stats = analyze_mode(versions_data, "eval", common_models)
    train_stats = analyze_mode(versions_data, "train", common_models)
    errors = []

    for i, vd in enumerate(versions_data):
        label = vd["label"]
        ver = label.replace("pt", "")
        md_path = os.path.join(RESULTS_DIR, f"{label}.md")
        if not os.path.exists(md_path):
            continue

        with open(md_path) as f:
            content = f.read()

        e = eval_stats[i]
        t = train_stats[i]

        # Look for version trend table rows with this version's numbers
        # Match patterns like "| 345 (74%) | 328 (70%) |"
        for line in content.split("\n"):
            if "Version Trend" in line and "common" not in line.lower() and "original" not in line.lower():
                continue
            if f"| {ver} " not in line and f"| **{ver}**" not in line:
                continue
            # Extract numbers from the line
            nums = re.findall(r"\b(\d+)\s*\(\d+%\)", line)
            if len(nums) >= 2:
                md_eval = int(nums[0])
                md_train = int(nums[1])
                if md_eval != e["full_graph"]:
                    errors.append(
                        f"{md_path}: {label} eval full_graph: "
                        f"file says {md_eval}, computed {e['full_graph']}"
                    )
                if md_train != t["full_graph"]:
                    errors.append(
                        f"{md_path}: {label} train full_graph: "
                        f"file says {md_train}, computed {t['full_graph']}"
                    )

    # Also check results/README.md and top-level README.md
    # Only validate lines within "Version Trend" sections (not "Expanded Corpus" etc.)
    for readme in [os.path.join(RESULTS_DIR, "README.md"), os.path.join(REPO_ROOT, "README.md")]:
        if not os.path.exists(readme):
            continue
        with open(readme) as f:
            lines = f.read().split("\n")

        in_trend_section = False
        for line in lines:
            if "version trend" in line.lower():
                in_trend_section = True
                continue
            if line.startswith("##") and "version trend" not in line.lower():
                in_trend_section = False
                continue
            if not in_trend_section:
                continue

            for i, vd in enumerate(versions_data):
                label = vd["label"]
                ver = label.replace("pt", "")
                e = eval_stats[i]
                t = train_stats[i]
                if f"| {ver} " not in line and f"| [{ver}]" not in line and f"| **{ver}**" not in line:
                    continue
                nums = re.findall(r"\b(\d+)\s*\(\d+%\)", line)
                if len(nums) >= 2:
                    md_eval = int(nums[0])
                    md_train = int(nums[1])
                    if md_eval != e["full_graph"]:
                        errors.append(
                            f"{readme}: {ver} eval full_graph: "
                            f"file says {md_eval}, computed {e['full_graph']}"
                        )
                    if md_train != t["full_graph"]:
                        errors.append(
                            f"{readme}: {ver} train full_graph: "
                            f"file says {md_train}, computed {t['full_graph']}"
                        )

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Analyze graph break trends across PyTorch versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dirs", nargs="*", help="Version directories containing identify_results.json (auto-discovered if omitted)")
    parser.add_argument("--labels", nargs="*", help="Labels for each version directory")
    parser.add_argument("--train", action="store_true", help="Include train mode analysis")
    parser.add_argument("--details", action="store_true", help="Show per-model details for transitions")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--markdown", action="store_true", help="Output version trend table as markdown")
    parser.add_argument("--validate", action="store_true", help="Validate results/*.md match computed data")
    args = parser.parse_args()

    versions = find_version_dirs_from_args(args)
    if len(versions) < 2:
        print("ERROR: Need at least 2 versions to compare", file=sys.stderr)
        sys.exit(1)

    # Load data
    for v in versions:
        v["identify_data"] = load_identify(v["identify"])
        v["explain_data"] = load_explain(v["explain"])

    # Find common models
    all_sets = [set(v["identify_data"].keys()) for v in versions]
    common_models = all_sets[0]
    for s in all_sets[1:]:
        common_models &= s

    if args.validate:
        # For validation, only include versions that have results files.
        # This prevents pt2.12 (nightly) from narrowing the common model set.
        validated = [v for v in versions if os.path.exists(
            os.path.join(RESULTS_DIR, f"{v['label']}.md"))]
        if len(validated) < 2:
            print("ERROR: Need at least 2 versions with results files to validate", file=sys.stderr)
            sys.exit(1)
        val_common = set.intersection(*(set(v["identify_data"].keys()) for v in validated))
        errors = validate_results(validated, val_common)
        if errors:
            print(f"VALIDATION FAILED — {len(errors)} error(s):\n", file=sys.stderr)
            for e in errors:
                print(f"  ✗ {e}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"OK — all version trend numbers in results/*.md match computed data ({len(val_common)} common models)")
            sys.exit(0)

    if args.markdown:
        print(generate_markdown(versions, common_models))
        sys.exit(0)

    if not args.json:
        print(f"Comparing {len(versions)} versions: {', '.join(v['label'] for v in versions)}")
        for v in versions:
            print(f"  {v['label']}: identify={v['identify']}")
            if os.path.exists(v["explain"]):
                print(f"  {' ' * len(v['label'])}  explain={v['explain']}")

    modes = ["eval"]
    if args.train:
        modes.append("train")

    if args.json:
        output = {}
        for mode in modes:
            output[mode] = {
                "summary": analyze_mode(versions, mode, common_models),
                "transitions": {
                    k: [{"name": n, "path": p} for n, p in v]
                    for k, v in find_transitions(versions, mode, common_models).items()
                },
                "apples_to_apples_models": sorted(find_apples_to_apples(versions, mode, common_models)),
            }
        print(json.dumps(output, indent=2))
    else:
        print_report(versions, modes, common_models, show_details=args.details)


if __name__ == "__main__":
    main()
