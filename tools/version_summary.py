#!/usr/bin/env python3
"""Per-PyTorch-version summary of completed sweeps.

Scans sweep_results/**/sweep_state.json for finished runs (status "done"),
loads each run's merged corpus.json, and writes ONE JSON file keyed by torch
version with fullgraph / graph-break counts (total, by source, by mode) and
Dynamo recompile totals/averages over graph-breaking rows/models.

Recompile counts come from the explain pass and live nested at
row["explain"]["recompile_count"] / ["recompile_count_steady"]. Rows without
them (explain error/skip, or torch without the metrics API) are excluded
from averages and counted in n_missing.

Usage:
    python3 tools/version_summary.py [--root sweep_results] [--out sweep_results/version_summary.json]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RC_FIELDS = ("recompile_count", "recompile_count_steady")


def _recompile_values(row):
    """Return {field: int} for a graph_break row, or None if no data."""
    ex = row.get("explain") or {}
    vals = {f: ex.get(f, row.get(f)) for f in RC_FIELDS}
    if any(v is None for v in vals.values()):
        return None
    return vals


def _recompile_block(values):
    """values: list of {field: int} or None."""
    have = [v for v in values if v is not None]
    out = {}
    for f in RC_FIELDS:
        total = sum(v[f] for v in have)
        out[f] = {"total": total, "avg": round(total / len(have), 4) if have else None}
    out["n_with_data"] = len(have)
    out["n_missing"] = len(values) - len(have)
    return out


def _row_stats(rows):
    gb = [r for r in rows if r.get("status") == "graph_break"]
    return {
        "full_graph": sum(1 for r in rows if r.get("status") == "full_graph"),
        "graph_break": len(gb),
        "recompile": _recompile_block([_recompile_values(r) for r in gb]),
    }


def _model_stats(models):
    """models: list of row-lists, one per (source, name)."""
    fg = gb = other = 0
    rc_values = []
    for rows in models:
        statuses = [r.get("status") for r in rows]
        if any(s == "graph_break" for s in statuses):
            gb += 1
            per_mode = [_recompile_values(r) for r in rows if r.get("status") == "graph_break"]
            have = [v for v in per_mode if v is not None]
            # Sum across the model's gb modes; count as "with data" only if every gb mode has data.
            if have and len(have) == len(per_mode):
                rc_values.append({f: sum(v[f] for v in have) for f in RC_FIELDS})
            else:
                rc_values.append(None)
        elif statuses and all(s == "full_graph" for s in statuses):
            fg += 1
        else:
            other += 1
    return {
        "n_models": len(models),
        "full_graph": fg,
        "graph_break": gb,
        "other": other,
        "recompile": _recompile_block(rc_values),
    }


def summarize_run(run_dir):
    state = json.loads((run_dir / "sweep_state.json").read_text())
    corpus = json.loads((run_dir / "corpus.json").read_text())
    rows = corpus["models"]

    by_source = defaultdict(list)
    by_mode = defaultdict(list)
    models = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
        by_mode[r["mode"]].append(r)
        models[(r["source"], r["name"])].append(r)

    models_by_source = defaultdict(list)
    for (src, _), mrows in models.items():
        models_by_source[src].append(mrows)

    return {
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir),
        "versions": {k: v for k, v in (state.get("versions") or {}).items() if k != "python"},
        "rows": {
            "total": _row_stats(rows),
            "by_source": {s: _row_stats(v) for s, v in sorted(by_source.items())},
            "by_mode": {m: _row_stats(v) for m, v in sorted(by_mode.items())},
        },
        "models": {
            "total": _model_stats(list(models.values())),
            "by_source": {s: _model_stats(v) for s, v in sorted(models_by_source.items())},
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "sweep_results")
    ap.add_argument("--out", type=Path, default=REPO / "sweep_results" / "version_summary.json")
    args = ap.parse_args()

    result = {}
    for state_path in sorted(args.root.rglob("sweep_state.json")):
        run_dir = state_path.parent
        state = json.loads(state_path.read_text())
        if state.get("status") != "done" or not (run_dir / "corpus.json").exists():
            print(f"skip {run_dir} (status={state.get('status')}, corpus.json="
                  f"{(run_dir / 'corpus.json').exists()})", file=sys.stderr)
            continue
        torch_v = (state.get("versions") or {}).get("torch")
        if not torch_v:
            print(f"skip {run_dir} (no torch version in sweep_state.json)", file=sys.stderr)
            continue
        if torch_v in result:
            sys.exit(f"duplicate torch version {torch_v}: {result[torch_v]['run_dir']} and {run_dir}")
        result[torch_v] = summarize_run(run_dir)

    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out} ({len(result)} versions)")
    for v, s in result.items():
        t, m = s["rows"]["total"], s["models"]["total"]
        print(f"  {v:16s} rows fg={t['full_graph']} gb={t['graph_break']} "
              f"rc_total={t['recompile']['recompile_count']['total']} | "
              f"models fg={m['full_graph']} gb={m['graph_break']} "
              f"rc_avg={m['recompile']['recompile_count']['avg']}")


if __name__ == "__main__":
    main()
