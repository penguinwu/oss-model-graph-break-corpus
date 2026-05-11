#!/usr/bin/env python3
"""Amend a completed sweep with verified post-fix results.

USAGE
-----
    python tools/amend_sweep.py \\
        --sweep-dir sweep_results/nightly/2026-05-03 \\
        --models /tmp/aria_validate.json \\
        --reason aria-fix \\
        --fix-commit 106bd19 \\
        --fix-description "HF non_deterministic_models pattern + comparison-side skip" \\
        --trigger "post-nightly regression triage" \\
        --python /home/pengwu/envs/torch-nightly-cu126/bin/python3

WHEN TO USE
-----------
After a sweep completes and you discover regressions that turn out to be
fixable in the harness (not real upstream regressions). Workflow:
  1. Sweep completes → identify_results.json written
  2. You analyze, find regressions, fix in code, push commit <sha>
  3. Run amend_sweep with --fix-commit <sha> to re-run the affected models
     under the new harness, against the SAME torch venv
  4. Amendment data is appended to identify_results.json under "amendments"
  5. Reports re-render via results_loader and show post-fix data with
     per-row provenance (result_source: "amended:<id>")

WHAT IT DOES
------------
- Reads metadata.environment from identify_results.json — extracts torch
  version. REFUSES if --python's torch version doesn't match (would
  conflate environments).
- Re-runs the harness on the specified models (single sweep pass)
- Atomically read-modify-write identify_results.json: appends a new entry
  to "amendments[]" with provenance + the new rows
- NEVER modifies the existing "results[]" array

GUARDS
------
- Env-match required (refuse otherwise)
- --fix-commit required (links amendment to code change)
- Amendment ID must be unique within this sweep dir (use --force-supersede
  to override; writes a new amendment with supersedes=<prior_id>)
- Atomic write: temp file + rename, prevents corruption on mid-write crash
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import results_loader from sweep/ (canonical reader, handles both JSON + JSONL formats)
sys.path.insert(0, str(REPO_ROOT / "sweep"))
from results_loader import load_raw, is_jsonl_format  # noqa: E402
def _run_harness_for_models(
    python_bin: str, models_json: Path, modes: list[str], workers: int,
    timeout_s: int, output_dir: Path,
) -> list[dict]:
    """Run a single identify pass on the listed models, return result rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, str(REPO_ROOT / "tools" / "run_experiment.py"), "sweep",
        "--models", str(models_json),
        "--modes", *modes,
        "--workers", str(workers),
        "--timeout", str(timeout_s),
        "--identify-only", "--no-auto-retry",
        "--output-dir", str(output_dir),
        # amend_sweep operates on a small ad-hoc cohort (re-run the affected
        # models). The cohort-validator's metadata requirement (designed for
        # full sweep launches) is overkill here — pass --allow-bare-cohort
        # so amend_sweep accepts a flat list. The amendment audit trail
        # already captures provenance via fix_commit + fix_description.
        "--allow-bare-cohort",
    ]
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"harness exited {result.returncode}")
    results_file = output_dir / "identify_results.json"
    if not results_file.exists():
        raise RuntimeError(f"harness did not write {results_file}")
    data = load_raw(results_file)
    return data["results"]


def _run_explain_for_amended(
    python_bin: str, identify_results_path: Path, workers: int,
    timeout_s: int, output_dir: Path,
) -> list[dict]:
    """Run explain pass on the amended identify results.

    The explain pass reads identify_results.json and only processes rows with
    status='graph_break'. Returns the list of explain entries written to
    explain_checkpoint.jsonl. Returns [] if no amended row needs explain.
    """
    cmd = [
        python_bin, str(REPO_ROOT / "tools" / "run_experiment.py"), "explain",
        str(identify_results_path),
        "--workers", str(workers),
        "--timeout", str(timeout_s),
        "--output-dir", str(output_dir),
    ]
    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"explain harness exited {result.returncode}")
    explain_ckpt = output_dir / "explain_checkpoint.jsonl"
    if not explain_ckpt.exists():
        # No graph_break rows in this amendment → nothing to explain
        return []
    rows = []
    with open(explain_ckpt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _verify_env_match(
    python_bin: str, sweep_metadata: dict,
) -> dict[str, str]:
    """Refuse if --python's torch version doesn't match sweep's torch.

    Returns the env constraints dict for the amendment metadata.
    """
    sweep_env_str = sweep_metadata.get("python", "")
    # Sweep metadata uses string format like "torch=2.13.0.dev20260502+cu126,
    # transformers=5.8.0.dev0, diffusers=0.38.0" — parse out per-package versions.
    sweep_versions = {}
    for piece in sweep_env_str.replace(";", ",").split(","):
        piece = piece.strip()
        if "=" in piece:
            pkg, ver = piece.split("=", 1)
            sweep_versions[pkg.strip()] = ver.strip()
    # Query the new python's torch / transformers / diffusers
    probe = subprocess.run(
        [python_bin, "-c",
         "import torch, transformers, diffusers; "
         "print(torch.__version__, transformers.__version__, diffusers.__version__)"],
        capture_output=True, text=True, check=True,
    )
    new_torch, new_tx, new_df = probe.stdout.strip().split()
    new_versions = {"torch": new_torch, "transformers": new_tx, "diffusers": new_df}
    # Strict match on torch (most important); warn but allow drift on transformers/diffusers
    sweep_torch = sweep_versions.get("torch", "")
    if sweep_torch and new_torch != sweep_torch:
        raise RuntimeError(
            f"Env mismatch: sweep torch={sweep_torch!r} vs --python torch={new_torch!r}. "
            f"Amendment must use the same torch build as the original sweep — "
            f"different builds = different sweep."
        )
    return new_versions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, type=Path,
                    help="Path to sweep dir containing identify_results.json")
    ap.add_argument("--models", required=True, type=Path,
                    help="JSON file with [{name, source, ...}, ...] specs")
    ap.add_argument("--reason", required=True,
                    help="Short slug used to compose amendment_id (e.g. 'aria-fix')")
    ap.add_argument("--fix-commit", required=True,
                    help="Git SHA of the commit that justifies this amendment")
    ap.add_argument("--fix-description", required=True,
                    help="One-line description of what the fix does")
    ap.add_argument("--trigger", default="post-sweep regression triage",
                    help="Human-readable cause that prompted this amendment")
    ap.add_argument("--python", required=True,
                    help="Python binary to use for the harness re-run")
    ap.add_argument("--modes", nargs="+", default=["eval", "train"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--force-supersede", action="store_true",
                    help="Allow re-using an existing amendment_id (records as supersedes=<prior>)")
    args = ap.parse_args()

    if not args.sweep_dir.exists():
        print(f"ERROR: sweep dir not found: {args.sweep_dir}", file=sys.stderr)
        return 1
    results_path = args.sweep_dir / "identify_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — sweep incomplete?", file=sys.stderr)
        return 1
    if not args.models.exists():
        print(f"ERROR: models JSON not found: {args.models}", file=sys.stderr)
        return 1

    # Read sweep data (handles both JSON and JSONL formats transparently)
    sweep_data = load_raw(results_path)
    sweep_metadata = sweep_data.get("metadata", {})

    # Verify env match (refuses if torch differs)
    print("Verifying environment match against sweep metadata...", file=sys.stderr)
    env_constraints = _verify_env_match(args.python, sweep_metadata)
    print(f"  OK — torch={env_constraints['torch']}", file=sys.stderr)

    # Compose amendment_id
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    amendment_id = f"{timestamp}-{args.reason}"

    # Dedup guard: refuse if a previous amendment with the SAME fix_commit covers
    # the SAME (name, mode) keys. Catches accidental re-runs from misdiagnosed
    # "killed" processes (the disowned child often completed even when the
    # foreground Bash wrapper got SIGTERM'd) — we lost trust by writing 3 dup
    # amendments on 2026-05-04 morning.
    existing_amendments = sweep_data.get("amendments", [])
    new_keys = set()
    with open(args.models) as f:
        spec_list = json.load(f)
    for spec in spec_list:
        for mode in args.modes:
            new_keys.add((spec["name"], mode))
    for prior in existing_amendments:
        if prior.get("fix_commit") != args.fix_commit:
            continue
        prior_keys = {(r["name"], r["mode"]) for r in prior.get("rows", [])}
        if new_keys.issubset(prior_keys):
            if not args.force_supersede:
                print(
                    f"ERROR: a prior amendment with fix_commit={args.fix_commit} "
                    f"already covers all {len(new_keys)} (name, mode) keys "
                    f"you're about to amend.\n"
                    f"  Prior amendment_id: {prior['amendment_id']!r}\n"
                    f"  Applied at: {prior['applied_at']}\n"
                    f"  This is usually a duplicate run after a misdiagnosed "
                    f"'killed' process (the disowned child likely ran to completion).\n"
                    f"  Verify with: ls {results_path.parent}/_amend_workspace_*\n"
                    f"  Use --force-supersede to add a new amendment anyway.",
                    file=sys.stderr,
                )
                return 1

    # Check for ID collisions
    existing_ids = {a.get("amendment_id") for a in existing_amendments}
    supersedes = None
    if amendment_id in existing_ids:
        if not args.force_supersede:
            print(f"ERROR: amendment_id {amendment_id!r} already exists. "
                  f"Use --force-supersede to add a new amendment that supersedes it.",
                  file=sys.stderr)
            return 1
        supersedes = amendment_id
        amendment_id = f"{amendment_id}-r{uuid.uuid4().hex[:6]}"

    # Run the identify harness
    print(f"Amendment id: {amendment_id}", file=sys.stderr)
    work_dir = args.sweep_dir / f"_amend_workspace_{amendment_id}"
    explain_rows: list[dict] = []
    try:
        rows = _run_harness_for_models(
            args.python, args.models, args.modes, args.workers,
            args.timeout, work_dir,
        )
        # Run explain on amended rows that ended up as graph_break.
        # Explain pass only runs on graph_break (full_graph has 0 breaks; errors
        # don't compile). Skip the explain step if no row qualifies.
        gb_rows = [r for r in rows if r.get("status") == "graph_break"]
        if gb_rows:
            print(f"\n  Running explain pass on {len(gb_rows)} graph_break rows...",
                  file=sys.stderr)
            workspace_identify = work_dir / "identify_results.json"
            if not workspace_identify.exists():
                raise RuntimeError(f"workspace identify_results.json missing: {workspace_identify}")
            explain_rows = _run_explain_for_amended(
                args.python, workspace_identify, args.workers,
                args.timeout, work_dir,
            )
            # Filter explain_rows to only the (name, mode) pairs we just amended
            # (the explain pass may reuse a stale checkpoint or include extras)
            new_keys = {(r["name"], r["mode"]) for r in rows}
            explain_rows = [er for er in explain_rows
                            if (er.get("name"), er.get("mode")) in new_keys]
        else:
            print(f"\n  No graph_break rows in amendment — skipping explain pass.",
                  file=sys.stderr)
    finally:
        # Clean up the workspace; keep only the amended rows
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    # Build the amendment record
    amendment = {
        "amendment_id": amendment_id,
        "applied_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fix_commit": args.fix_commit,
        "fix_description": args.fix_description,
        "trigger": args.trigger,
        "env_constraints": env_constraints,
        "supersedes": supersedes,
        "row_count": len(rows),
        "rows": rows,
        "explain_row_count": len(explain_rows),
    }

    # Persist the amendment.
    #
    # JSONL format (new sweeps): true append — write a single "amendment" line.
    #   No rewrite needed: just append one JSON object to the file.
    #
    # JSON format (legacy sweeps pre-migration): use the old atomic read-modify-
    #   write so we don't leave the file half-valid.
    if is_jsonl_format(results_path):
        with open(results_path, "a") as f:
            f.write(json.dumps({"_record_type": "amendment", **amendment}) + "\n")
        total_amendments = len(sweep_data.get("amendments", [])) + 1
    else:
        # Legacy JSON path: atomic rewrite (keep for backward compat window)
        sweep_data.setdefault("amendments", []).append(amendment)
        tmp = results_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(sweep_data, indent=2))
        os.replace(tmp, results_path)
        total_amendments = len(sweep_data["amendments"])

    # Append explain rows to the sweep's explain_checkpoint.jsonl, tagged with
    # amendment_id so load_effective_explain can identify them as amended.
    # JSONL append is atomic enough for our use case (no concurrent writers).
    if explain_rows:
        explain_path = args.sweep_dir / "explain_checkpoint.jsonl"
        with open(explain_path, "a") as f:
            for er in explain_rows:
                tagged = {**er, "amendment_id": amendment_id}
                f.write(json.dumps(tagged) + "\n")

    print(f"\n  ✓ Amendment {amendment_id} applied to {results_path}", file=sys.stderr)
    print(f"    {len(rows)} identify rows added; original results[] untouched.", file=sys.stderr)
    if explain_rows:
        print(f"    {len(explain_rows)} explain rows appended to explain_checkpoint.jsonl.",
              file=sys.stderr)
    print(f"    Total amendments in this sweep: {total_amendments}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
