"""Add fix_status verdict to a discovery trial result.

The original `validate.py` per case ran the model with canonical case-spec inputs.
That's `gb_under_canonical_inputs` (model-layer-only check).

This script adds the missing piece: re-runs the agent's edited baseline_<model>.py
script in a subprocess, parses its `graph_break_count=N` line, and combines with
the existing `validation.graph_break_count` to compute a clear `fix_status` verdict:

  - `general`: model-layer changes alone fix it. gb=0 under both regimes.
  - `setup-required`: agent's full fix (model + setup-layer edits to the run script)
    achieves gb=0, but model alone doesn't. Legit fix, scoped to the agent's setup.
  - `none`: gb>0 even in agent's own run.

For trials that already ran, this rewrites their result.json `validation` field
in place with the structured schema:

```
{
  "integrity": {"import_ok", "eager_ok", "compile_ok"},
  "fix_status": "general" | "setup-required" | "none",
  "details": {
    "gb_in_agent_run": int,
    "gb_under_canonical_inputs": int,
    "max_diff_compiled_vs_eager": float,
    "max_diff_vs_baseline": float
  }
}
```

The original validation field is preserved at `validation_legacy` so we don't lose
the raw integrity flags from the original run.

Usage:
  python -m discovery.revalidate --trial-dir <dir>
  python -m discovery.revalidate --case mistral3_data_dep --run-id 20260425-041832

The second form re-validates ALL trials in a run_dir, in parallel via threads.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

DISCOVERY_RESULTS = Path("/tmp/discovery-runs")
REPO = Path(__file__).resolve().parents[1]


def revalidate_trial(trial_dir: Path) -> dict:
    """Re-validate one trial; rewrite result.json in place. Returns the new validation field."""
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return {"_error": f"no result.json in {trial_dir}"}
    result = json.loads(result_path.read_text())
    case_id = result.get("case_id")
    if not case_id:
        return {"_error": f"no case_id in {result_path}"}

    # Load the case spec to find the agent's baseline script path + restore mechanism.
    case_mod = importlib.import_module(f"discovery.cases.{case_id}")
    case = case_mod.get_case_spec()

    # Find the baseline script in the watched files (the one whose path has "baseline" in it).
    baseline_wf = None
    for wf in case.watched_files:
        if "baseline" in wf.path.name:
            baseline_wf = wf
            break
    if baseline_wf is None:
        return {"_error": "case has no baseline-* watched file"}

    # Apply the agent's diff to a clean state, then run their script.
    diff_path = trial_dir / "agent_diff.patch"
    if not diff_path.exists() or diff_path.stat().st_size == 0:
        # No diff — agent didn't edit. Use canonical-only assessment.
        agent_gb = None
    else:
        # Restore all watched files to .original FIRST (clean slate).
        for wf in case.watched_files:
            if wf.original_backup.exists():
                wf.path.write_bytes(wf.original_backup.read_bytes())

        # Rewrite the diff so both --- and +++ point at the live file path
        # (diff was generated with .original as old path, live as new path —
        # patch's path heuristic picks the .original path which we don't
        # want to mutate).
        diff_text = diff_path.read_text()
        rewritten_lines = []
        for line in diff_text.splitlines():
            if line.startswith("--- "):
                # Find the matching live path. The diff has --- .original_path
                # \t timestamp; the next +++ line has the live path.
                old_path = line[4:].split("\t")[0]
                # Map .original to live path via watched_files lookup
                for wf in case.watched_files:
                    if str(wf.original_backup) == old_path:
                        rewritten_lines.append(f"--- {wf.path}")
                        break
                else:
                    rewritten_lines.append(line)
            elif line.startswith("+++ "):
                # Live path — keep it but strip timestamp
                new_path = line[4:].split("\t")[0]
                rewritten_lines.append(f"+++ {new_path}")
            else:
                rewritten_lines.append(line)
        rewritten = "\n".join(rewritten_lines) + "\n"

        # Apply the rewritten diff via patch.
        patch_res = subprocess.run(
            ["patch", "-p0", "--no-backup-if-mismatch", "--quiet"],
            input=rewritten, cwd="/",
            capture_output=True, text=True,
        )
        if patch_res.returncode != 0:
            # Restore + bail
            for wf in case.watched_files:
                if wf.original_backup.exists():
                    wf.path.write_bytes(wf.original_backup.read_bytes())
            return {"_error": f"patch failed: rc={patch_res.returncode} {patch_res.stderr[:300]}"}

        # Now run the agent's edited baseline script. Capture gb count.
        run_res = subprocess.run(
            ["/home/pengwu/envs/torch211/bin/python", str(baseline_wf.path)],
            capture_output=True, text=True, timeout=300,
        )
        # Parse "graph_break_count=N" from stdout/stderr.
        text = run_res.stdout + "\n" + run_res.stderr
        m = re.search(r"graph_break_count\s*=\s*(\d+)", text)
        agent_gb = int(m.group(1)) if m else None

        # Restore watched files to .original after the test.
        for wf in case.watched_files:
            if wf.original_backup.exists():
                wf.path.write_bytes(wf.original_backup.read_bytes())

    # Existing validation field (canonical-inputs check)
    legacy = result.get("validation") or {}
    canonical_gb = legacy.get("graph_break_count")
    integrity = {
        "import_ok": legacy.get("import_ok", False),
        "eager_ok": legacy.get("eager_ok", False),
        "compile_ok": legacy.get("compile_ok", False),
    }

    # Compute fix_status
    if agent_gb is None:
        # No diff — fix_status driven by canonical alone.
        if canonical_gb == 0:
            fix_status = "general"
        elif canonical_gb is not None and canonical_gb > 0:
            fix_status = "none"
        else:
            fix_status = "unknown"
    else:
        if agent_gb == 0 and canonical_gb == 0:
            fix_status = "general"
        elif agent_gb == 0 and canonical_gb is not None and canonical_gb > 0:
            fix_status = "setup-required"
        elif agent_gb > 0:
            fix_status = "none"
        else:
            fix_status = "unknown"

    validation_v2 = {
        "integrity": integrity,
        "fix_status": fix_status,
        "details": {
            "gb_in_agent_run": agent_gb,
            "gb_under_canonical_inputs": canonical_gb,
            "max_diff_compiled_vs_eager": legacy.get("max_diff_compiled_vs_eager_now"),
            "max_diff_vs_baseline": legacy.get("max_diff_vs_eager_baseline"),
        },
    }
    # NEVER overwrite the original `validation` field. Add as `validation_v2`.
    result["validation_v2"] = validation_v2
    result_path.write_text(json.dumps(result, indent=2))
    return validation_v2


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trial-dir", help="path to a single trial dir")
    p.add_argument("--case", help="case_id (with --run-id, re-validate all trials in that run)")
    p.add_argument("--run-id", help="run_id for batch mode")
    p.add_argument("--workers", type=int, default=1, help="parallel workers (single trial dir only)")
    args = p.parse_args()

    if args.trial_dir:
        out = revalidate_trial(Path(args.trial_dir))
        print(json.dumps(out, indent=2))
        return 0

    if args.case and args.run_id:
        run_dir = DISCOVERY_RESULTS / args.case / args.run_id
        trial_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        # SEQUENTIAL — agent diffs apply to shared model files in the env, can't parallelize safely.
        results = {}
        for td in trial_dirs:
            print(f"--- {td.name} ---", flush=True)
            r = revalidate_trial(td)
            results[td.name] = r
            fs = r.get("fix_status", "?")
            details = r.get("details", {})
            print(f"  fix_status={fs}  gb_agent={details.get('gb_in_agent_run')}  gb_canonical={details.get('gb_under_canonical_inputs')}", flush=True)
        # Print summary table
        print("\n=== fix_status by trial ===")
        from collections import Counter
        verdicts = Counter(r.get("fix_status", "?") for r in results.values())
        for fs, count in verdicts.most_common():
            print(f"  {fs}: {count}")
        return 0

    p.error("provide either --trial-dir, or --case + --run-id")


if __name__ == "__main__":
    sys.exit(main())
