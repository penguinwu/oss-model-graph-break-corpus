"""Driver: run discovery trials for one case.

Usage:
  python -m discovery.run_case --case dbrx_moe_data_dep --variants V0 --n 1
  python -m discovery.run_case --case dbrx_moe_data_dep --variants V0,V1 --n 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path

from discovery.runner import run_trial
from discovery.variants import ALL_VARIANTS

DISCOVERY_RESULTS = Path("/tmp/discovery-runs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="case id, e.g. dbrx_moe_data_dep")
    parser.add_argument("--variants", required=True, help="comma-separated variant ids, e.g. V0,V1")
    parser.add_argument("--n", type=int, default=1, help="trials per variant")
    parser.add_argument("--timeout", type=int, default=1200, help="per-trial agent timeout in seconds")
    args = parser.parse_args()

    case_mod = importlib.import_module(f"discovery.cases.{args.case}")
    case = case_mod.get_case_spec()

    variants = [ALL_VARIANTS[vid.strip()] for vid in args.variants.split(",")]

    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_dir = DISCOVERY_RESULTS / args.case / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_dir = {run_dir}")

    summary: list[dict] = []
    for variant in variants:
        for trial_idx in range(args.n):
            trial_label = f"{variant.id}_{trial_idx + 1}"
            trial_dir = run_dir / trial_label
            print(f"\n--- {trial_label} ---", flush=True)
            t0 = time.time()
            result = run_trial(case, variant, trial_label, trial_dir, timeout_s=args.timeout)
            dt = time.time() - t0
            print(f"  exit={result.agent_exit_code} elapsed={dt:.1f}s flags={result.flags}")
            if result.validation:
                gb = result.validation.get("graph_break_count")
                md = result.validation.get("max_diff_compiled_vs_eager_now")
                print(f"  graph_break_count={gb} max_diff={md}")
            if result.perf:
                print(f"  perf: eager={result.perf.get('eager_ms'):.2f}ms compiled={result.perf.get('compiled_ms'):.2f}ms speedup={result.perf.get('speedup'):.2f}x")
            summary.append(result.to_dict())

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSummary at {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
