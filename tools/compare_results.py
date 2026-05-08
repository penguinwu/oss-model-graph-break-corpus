#!/usr/bin/env python3
"""Compare experiment runner results against sweep results.

Usage:
    python tools/compare_results.py experiments/results/pt2.11-confidence/results.jsonl sweep_results/pt2.11/identify_results.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


STATUS_GROUPS = {
    "success": {"full_graph"},
    "break": {"graph_break"},
    "error": {"create_error", "eager_error", "compile_error", "worker_error", "launch_error"},
    "timeout": {"timeout"},
}


def group_status(status):
    for group, members in STATUS_GROUPS.items():
        if status in members:
            return group
    return "other"


def load_experiment(path):
    results = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # Skip metadata header (run_experiment.py provenance header)
            if isinstance(r, dict) and r.get("_record_type") == "metadata":
                continue
            name = r.get("name", r.get("model"))
            key = (name, r.get("mode", "eval"))
            results[key] = r["status"]
    return results


def load_sweep(path):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from sweep.results_loader import load_results_list
    results = {}
    for r in load_results_list(path):
        name = r["name"]
        key = (name, r.get("mode", "eval"))
        results[key] = r["status"]
    return results


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    exp_path = Path(sys.argv[1])
    sweep_path = Path(sys.argv[2])

    exp = load_experiment(exp_path)
    sweep = load_sweep(sweep_path)

    print(f"Experiment: {len(exp)} results")
    print(f"Sweep:      {len(sweep)} results")
    print()

    # Status distributions
    exp_counts = Counter(exp.values())
    sweep_counts = Counter(sweep.values())
    print("Status distribution:")
    print(f"  {'Status':<20} {'Experiment':>10} {'Sweep':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10}")
    all_statuses = sorted(set(exp_counts.keys()) | set(sweep_counts.keys()))
    for s in all_statuses:
        print(f"  {s:<20} {exp_counts.get(s, 0):>10} {sweep_counts.get(s, 0):>10}")
    print()

    # Match on common keys
    common = set(exp.keys()) & set(sweep.keys())
    exp_only = set(exp.keys()) - set(sweep.keys())
    sweep_only = set(sweep.keys()) - set(exp.keys())

    print(f"Coverage:")
    print(f"  Common models:         {len(common)}")
    print(f"  Experiment-only:       {len(exp_only)}")
    print(f"  Sweep-only:            {len(sweep_only)}")
    print()

    if exp_only:
        print(f"  Experiment-only models (first 20):")
        for name, mode in sorted(exp_only)[:20]:
            print(f"    {name} ({mode})")
        print()

    if sweep_only and len(sweep_only) <= 40:
        print(f"  Sweep-only models (first 20):")
        for name, mode in sorted(sweep_only)[:20]:
            print(f"    {name} ({mode})")
        print()

    # Agreement on common models
    agree = 0
    disagree = []
    group_agree = 0
    group_disagree = []
    for key in sorted(common):
        e_status = exp[key]
        s_status = sweep[key]
        if e_status == s_status:
            agree += 1
            group_agree += 1
        else:
            disagree.append((key, e_status, s_status))
            if group_status(e_status) == group_status(s_status):
                group_agree += 1
            else:
                group_disagree.append((key, e_status, s_status))

    print(f"Agreement (exact status):")
    print(f"  Match:     {agree}/{len(common)} ({100*agree/len(common):.1f}%)" if common else "  No common models")
    print(f"  Mismatch:  {len(disagree)}/{len(common)}" if common else "")
    print()
    print(f"Agreement (grouped: success/break/error/timeout):")
    print(f"  Match:     {group_agree}/{len(common)} ({100*group_agree/len(common):.1f}%)" if common else "  No common models")
    print(f"  Mismatch:  {len(group_disagree)}/{len(common)}" if common else "")
    print()

    if disagree:
        print(f"Exact mismatches (first 30):")
        print(f"  {'Model':<40} {'Mode':<6} {'Experiment':<16} {'Sweep':<16}")
        print(f"  {'-'*40} {'-'*6} {'-'*16} {'-'*16}")
        for (name, mode), e_s, s_s in disagree[:30]:
            marker = " ***" if group_status(e_s) != group_status(s_s) else ""
            print(f"  {name:<40} {mode:<6} {e_s:<16} {s_s:<16}{marker}")
        if len(disagree) > 30:
            print(f"  ... and {len(disagree) - 30} more")
        print()

    if group_disagree:
        # Classify mismatches: environment changes vs compilation regressions
        env_changes = []  # create_error involved — likely library version change
        compilation_regressions = []  # real signal — compilation behavior changed

        for item in group_disagree:
            (name, mode), e_s, s_s = item
            if "create_error" in (e_s, s_s) or "eager_error" in (e_s, s_s):
                env_changes.append(item)
            else:
                compilation_regressions.append(item)

        if compilation_regressions:
            print(f"COMPILATION REGRESSIONS ({len(compilation_regressions)}):")
            print(f"  Real signal — compilation behavior changed between runs")
            print(f"  {'Model':<40} {'Mode':<6} {'Experiment':<16} {'Sweep':<16}")
            print(f"  {'-'*40} {'-'*6} {'-'*16} {'-'*16}")
            for (name, mode), e_s, s_s in compilation_regressions:
                print(f"  {name:<40} {mode:<6} {e_s:<16} {s_s:<16}")
            print()

        if env_changes:
            print(f"ENVIRONMENT CHANGES ({len(env_changes)}):")
            print(f"  Likely library version differences (create/eager errors)")
            print(f"  {'Model':<40} {'Mode':<6} {'Experiment':<16} {'Sweep':<16}")
            print(f"  {'-'*40} {'-'*6} {'-'*16} {'-'*16}")
            for (name, mode), e_s, s_s in env_changes:
                print(f"  {name:<40} {mode:<6} {e_s:<16} {s_s:<16}")
            print()


if __name__ == "__main__":
    main()
