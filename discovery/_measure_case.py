"""Run measure_perf for a case in a fresh subprocess.

Used by runner.run_trial — subprocessing avoids module-state contamination
when the agent has rewritten the model source between trials.

Usage:
    python -m discovery._measure_case --case dbrx_moe_data_dep
    python -m discovery._measure_case --case dbrx_moe_data_dep --tier realistic
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from discovery.perf import measure_perf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--tier", default="fast", choices=["fast", "realistic"],
                        help="fast = tiny inputs (default, for discovery loop). "
                             "realistic = case-defined production-like inputs.")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-repeat", type=int, default=20)
    args = parser.parse_args()

    case_mod = importlib.import_module(f"discovery.cases.{args.case}")

    # Cases expose make_model + make_inputs (fast tier). For realistic tier,
    # they may expose make_model_realistic + make_inputs_realistic. Fall back
    # to fast-tier functions if a case hasn't opted in to tier-2.
    if args.tier == "realistic":
        model_fn = getattr(case_mod, "make_model_realistic", case_mod.make_model)
        inputs_fn = getattr(case_mod, "make_inputs_realistic", case_mod.make_inputs)
    else:
        model_fn = case_mod.make_model
        inputs_fn = case_mod.make_inputs

    result = measure_perf(
        model_fn, inputs_fn,
        n_warmup=args.n_warmup,
        n_repeat=args.n_repeat,
    )
    out = result.to_dict()
    out["tier"] = args.tier
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
