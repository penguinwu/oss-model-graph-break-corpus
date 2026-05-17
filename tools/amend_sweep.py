#!/usr/bin/env python3
"""Amend a completed sweep with additional row data.

TWO MODES
=========

Mode 1 — re-run (existing): run the harness on a subset of models after a
code fix and append the post-fix rows to identify_results.json::amendments[].

    python tools/amend_sweep.py \\
        --sweep-dir sweep_results/nightly/2026-05-03 \\
        --models /tmp/aria_validate.json \\
        --reason aria-fix \\
        --fix-commit 106bd19 \\
        --fix-description "HF non_deterministic_models pattern + comparison-side skip" \\
        --python /home/pengwu/envs/torch-nightly-cu126/bin/python3

Mode 2 — data-only merge (new, 2026-05-16): merge rows from an EXISTING
sweep dir into the target sweep WITHOUT re-running the harness. Used when
a subset re-run already happened in a separate sweep dir and you want to
fold that data into a broader baseline.

    python tools/amend_sweep.py \\
        --sweep-dir sweep_results/nightly/2026-05-03 \\
        --from-existing-results experiments/results/aria-revalidate-2026-05-04 \\
        --reason aria-revalidate-merge

WHEN TO USE
-----------
Mode 1 (re-run): after a sweep completes and you discover regressions that
are fixable in the harness. Workflow: sweep completes → analyze → fix in
code → re-run affected models via this tool → amendment appended.

Mode 2 (data-only): when you've already run a targeted reproduction sweep
in a separate dir, and the broader baseline should reflect that data
without spending compute to re-run.

WHAT IT DOES
------------
Mode 1:
- Verifies --python's torch version matches sweep's torch (refuse otherwise)
- Re-runs identify pass on the specified models
- Appends new entry to amendments[] with provenance + the new rows

Mode 2:
- Reads rows from <source-dir>/identify_streaming.jsonl (or results.jsonl)
- Strict validation: same env, same compile_kwargs, same dynamo_flags,
  same pass, source is a (name, mode) subset of target. Refuses otherwise.
- Appends amendment with merge_mode="data-only" + source_dir + source_sha256

NEVER modifies the existing "results[]" array in either mode.

GUARDS
------
Mode 1: --fix-commit + --python + strict env-match (all 3 packages).
Mode 2: --from-existing-results + same-pass + same-compile-config +
        same-env + subset check + graph_break→explain present check.
        NO --fix-commit (replaced with --reason).
Both: amendment_id uniqueness (--force-supersede to override).
Atomic write: JSONL append (new sweeps) or temp+rename (legacy JSON).

DEDUP SEMANTICS (Mode 2)
------------------------
The data-only dedup key is (source_dir, source_sha256). If the source sweep
is amended after a successful merge (its identify_streaming.jsonl grows),
the sha256 changes and a re-run of the data-only merge will SUCCEED,
appending a new amendment. This is intentional — the second merge picks
up the new rows. Use --force-supersede to allow a re-merge from an
unchanged source.

KNOWN GAPS (deferred per adversary review 2026-05-16 22:00 ET)
- amendment_id collision at minute granularity → forces --force-supersede
  with a misleading `supersedes` label when two distinct merges land in
  the same minute. Fix: add content-based suffix (first 8 chars of
  source_sha256 for Mode 2; first 8 of fix_commit for Mode 1).
- --reason is not regex-validated; in Mode 1 it ends up on the filesystem
  as _amend_workspace_<reason>. Add `[a-z0-9._-]+` validation at argparse.
- Dual-write (JSONL append vs JSON rewrite) is duplicated across modes —
  extract _persist_amendment helper.
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


def _parse_env_string(s: str) -> dict[str, str]:
    """Parse a sweep metadata env string into {pkg: version}.

    Accepted shapes (in order of precedence; first non-empty wins):
      - dict-shaped (already structured): pass-through
      - "pkg=ver,pkg=ver" or "pkg=ver;pkg=ver" — legacy comma/semicolon string
      - empty / unrecognized: returns {}

    Refuse-loud is the caller's job; this helper is intentionally permissive.
    """
    if isinstance(s, dict):
        # Future-proof: if metadata.python ever becomes structured, accept it.
        return {k: str(v) for k, v in s.items()}
    if not isinstance(s, str) or not s.strip():
        return {}
    out: dict[str, str] = {}
    for piece in s.replace(";", ",").split(","):
        piece = piece.strip()
        if "=" in piece:
            pkg, ver = piece.split("=", 1)
            pkg, ver = pkg.strip(), ver.strip()
            if pkg and ver:
                out[pkg] = ver
    return out


def _load_source_rows(source_dir: Path) -> tuple[list[dict], dict, str]:
    """Load rows + metadata from a source sweep dir for data-only merge mode.

    Reads <source_dir>/identify_streaming.jsonl (preferred — append-only
    streaming format) or falls back to <source_dir>/identify_results.json.
    Returns (rows, source_metadata, source_sha256).

    source_metadata is the {"torch": "...", "transformers": "...", "diffusers": "..."}
    dict from the source's sweep_state.json::launcher_python probe OR
    versions.json if present. None if not derivable.

    source_sha256 is computed over the raw rows file content; pinned in
    the amendment record so future readers can detect drift.
    """
    import hashlib

    streaming = source_dir / "identify_streaming.jsonl"
    results = source_dir / "identify_results.json"
    if streaming.exists():
        raw = streaming.read_bytes()
        rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
        source_sha256 = hashlib.sha256(raw).hexdigest()
    elif results.exists():
        raw = results.read_bytes()
        data = load_raw(results)
        rows = data.get("results", [])
        source_sha256 = hashlib.sha256(raw).hexdigest()
    else:
        raise FileNotFoundError(
            f"Source dir missing identify_streaming.jsonl AND identify_results.json: {source_dir}"
        )

    # Derive source env. Try versions.json first; fall back to sweep_state.json's
    # launcher_python probe (post-v4 watchdog field). If neither available, leave
    # empty and let validation catch the mismatch downstream.
    versions_path = source_dir / "versions.json"
    sweep_state_path = source_dir / "sweep_state.json"
    source_env: dict = {}
    if versions_path.exists():
        try:
            source_env = json.loads(versions_path.read_text())
        except Exception:
            pass
    if not source_env and sweep_state_path.exists():
        try:
            sst = json.loads(sweep_state_path.read_text())
            launcher_python = sst.get("launcher_python")
            if launcher_python:
                probe = subprocess.run(
                    [launcher_python, "-c",
                     "import torch, transformers, diffusers; "
                     "print(torch.__version__, transformers.__version__, diffusers.__version__)"],
                    capture_output=True, text=True, check=True, timeout=30,
                )
                t, tx, df = probe.stdout.strip().split()
                source_env = {"torch": t, "transformers": tx, "diffusers": df}
        except Exception:
            pass

    return rows, source_env, source_sha256


def _validate_data_only_merge(
    source_rows: list[dict], source_env: dict, target_metadata: dict,
    target_rows: list[dict],
) -> None:
    """Validate a data-only merge: same env, same compile_config, same pass, subset.

    Refuses (raises RuntimeError) on any mismatch. Designed to be strict —
    silent passes on partial mismatch would corrupt the target sweep's
    semantic integrity.
    """
    # 1. Env match (torch is strict; tx + diffusers MUST match too — data-only
    #    merges can't paper over modellib drift).
    target_env_str = target_metadata.get("python", "")
    target_env = _parse_env_string(target_env_str)
    if not target_env:
        raise RuntimeError(
            f"Target metadata has no parseable env (metadata.python={target_env_str!r}). "
            f"Refusing data-only merge — cannot prove env match. "
            f"Expected: dict-shaped OR 'pkg=ver,pkg=ver,...' string."
        )
    if not source_env:
        raise RuntimeError(
            "Source dir has no detectable env (no versions.json, no sweep_state.json "
            "with launcher_python). Refusing data-only merge — cannot prove env match."
        )
    for pkg in ("torch", "transformers", "diffusers"):
        src = source_env.get(pkg, "")
        tgt = target_env.get(pkg, "")
        if src and tgt and src != tgt:
            raise RuntimeError(
                f"Env mismatch ({pkg}): source={src!r} vs target={tgt!r}. "
                f"Data-only merge requires identical env."
            )
        if not src or not tgt:
            raise RuntimeError(
                f"Env unknown ({pkg}): source={src!r} target={tgt!r}. "
                f"Refusing — both sides must declare the same version."
            )

    # 2. Build target index for per-(name, mode) lookup.
    target_idx = {(r.get("name"), r.get("mode")): r for r in target_rows}

    # 3. Source must be a subset of target on (name, mode).
    source_keys = {(r.get("name"), r.get("mode")) for r in source_rows}
    missing = sorted(k for k in source_keys if k not in target_idx)
    if missing:
        raise RuntimeError(
            f"Subset check failed: {len(missing)} (name, mode) keys in source are "
            f"NOT present in target. Examples: {missing[:5]}. Data-only merge "
            f"requires source rows to be a subset of target rows."
        )

    # 4. Per-row compile_kwargs + dynamo_flags + pass match. Strict: every source
    #    row must agree with its target counterpart on (compile_kwargs, dynamo_flags,
    #    pass). Mismatch = different experiment; refuse.
    def _norm(d):
        # Treat None/missing as the same; sort keys for stable comparison.
        if d is None:
            return None
        return json.dumps(d, sort_keys=True)

    config_mismatches = []
    for sr in source_rows:
        key = (sr.get("name"), sr.get("mode"))
        tr = target_idx.get(key)
        if tr is None:
            continue  # already caught above
        for field in ("compile_kwargs", "dynamo_flags", "pass"):
            src_v = _norm(sr.get(field))
            tgt_v = _norm(tr.get(field))
            if src_v != tgt_v:
                config_mismatches.append((key, field, src_v, tgt_v))
                break
    if config_mismatches:
        sample = config_mismatches[:3]
        msg_lines = [
            f"Compile-config mismatch on {len(config_mismatches)} rows. "
            f"Source must match target on (compile_kwargs, dynamo_flags, pass). Examples:",
        ]
        for (name, mode), field, src_v, tgt_v in sample:
            msg_lines.append(f"  {name}/{mode}: {field} src={src_v} tgt={tgt_v}")
        raise RuntimeError("\n".join(msg_lines))


def _verify_env_match(
    python_bin: str, sweep_metadata: dict,
) -> dict[str, str]:
    """Refuse if --python's env doesn't strictly match the sweep's env.

    STRICT — all 3 packages (torch, transformers, diffusers) MUST match.
    Per Peng directive 2026-05-16: amendment provenance is the bedrock of
    corpus trust; lenient env-match silently undermines that. Tightened
    from torch-only after adversary review 2026-05-16 22:00 ET.

    Returns the env constraints dict for the amendment metadata.
    """
    sweep_versions = _parse_env_string(sweep_metadata.get("python", ""))
    if not sweep_versions:
        raise RuntimeError(
            f"Sweep metadata has no parseable env (metadata.python missing/malformed). "
            f"Cannot verify env match. Expected: 'pkg=ver,pkg=ver,...' string."
        )
    probe = subprocess.run(
        [python_bin, "-c",
         "import torch, transformers, diffusers; "
         "print(torch.__version__, transformers.__version__, diffusers.__version__)"],
        capture_output=True, text=True, check=True,
    )
    new_torch, new_tx, new_df = probe.stdout.strip().split()
    new_versions = {"torch": new_torch, "transformers": new_tx, "diffusers": new_df}
    for pkg, new_v in new_versions.items():
        sweep_v = sweep_versions.get(pkg, "")
        if sweep_v and new_v != sweep_v:
            raise RuntimeError(
                f"Env mismatch ({pkg}): sweep={sweep_v!r} vs --python={new_v!r}. "
                f"Amendment must use the identical env as the original sweep — "
                f"different versions = different experiment. Per Peng's strict-X "
                f"discipline (2026-05-16)."
            )
        if not sweep_v:
            raise RuntimeError(
                f"Env unknown ({pkg}): sweep declares no version, probe shows {new_v!r}. "
                f"Refusing — both sides must declare the same version."
            )
    return new_versions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, type=Path,
                    help="Path to sweep dir containing identify_results.json")
    # Re-run mode (Mode 1) args — required in re-run mode, ignored in data-only mode
    ap.add_argument("--models", type=Path,
                    help="(re-run mode) JSON file with [{name, source, ...}, ...] specs")
    ap.add_argument("--fix-commit",
                    help="(re-run mode) Git SHA of the commit that justifies this amendment")
    ap.add_argument("--fix-description",
                    help="(re-run mode) One-line description of what the fix does")
    ap.add_argument("--python",
                    help="(re-run mode) Python binary to use for the harness re-run")
    # Data-only mode (Mode 2, new 2026-05-16) — required in data-only mode
    ap.add_argument("--from-existing-results", type=Path, dest="from_existing_results",
                    help="(data-only mode) Source sweep dir to merge rows FROM "
                         "(without re-running the harness)")
    # Common args
    ap.add_argument("--reason", required=True,
                    help="Short slug used to compose amendment_id (e.g. 'aria-fix')")
    ap.add_argument("--trigger", default="post-sweep regression triage",
                    help="Human-readable cause that prompted this amendment")
    ap.add_argument("--modes", nargs="+", default=["eval", "train"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--force-supersede", action="store_true",
                    help="Allow re-using an existing amendment_id (records as supersedes=<prior>)")
    args = ap.parse_args()

    # Mode discrimination (mutually exclusive). Default to re-run mode for back-compat.
    if args.from_existing_results is not None:
        mode = "data-only"
        # Forbid re-run-only args
        if args.models or args.fix_commit or args.fix_description or args.python:
            print("ERROR: --from-existing-results (data-only mode) is mutually exclusive "
                  "with --models / --fix-commit / --fix-description / --python (re-run mode).",
                  file=sys.stderr)
            return 1
    else:
        mode = "re-run"
        # Re-run mode requires these
        missing = [n for n, v in [("--models", args.models), ("--fix-commit", args.fix_commit),
                                  ("--fix-description", args.fix_description),
                                  ("--python", args.python)] if not v]
        if missing:
            print(f"ERROR: re-run mode requires {', '.join(missing)}. "
                  f"(Use --from-existing-results for data-only merge mode.)",
                  file=sys.stderr)
            return 1

    if not args.sweep_dir.exists():
        print(f"ERROR: sweep dir not found: {args.sweep_dir}", file=sys.stderr)
        return 1
    results_path = args.sweep_dir / "identify_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found — sweep incomplete?", file=sys.stderr)
        return 1

    # Read sweep data (handles both JSON and JSONL formats transparently)
    sweep_data = load_raw(results_path)
    sweep_metadata = sweep_data.get("metadata", {})
    existing_amendments = sweep_data.get("amendments", [])

    # Compose amendment_id (timestamped + reason slug). Used for ID-collision
    # detection regardless of mode.
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    amendment_id = f"{timestamp}-{args.reason}"

    # ============================================================
    # Mode 2 — data-only merge (new path, 2026-05-16)
    # ============================================================
    if mode == "data-only":
        if not args.from_existing_results.exists():
            print(f"ERROR: --from-existing-results dir not found: "
                  f"{args.from_existing_results}", file=sys.stderr)
            return 1
        print(f"Data-only merge mode: loading rows from {args.from_existing_results}",
              file=sys.stderr)
        try:
            source_rows, source_env, source_sha256 = _load_source_rows(args.from_existing_results)
        except Exception as e:
            print(f"ERROR: failed to load source rows: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return 1
        print(f"  Loaded {len(source_rows)} source rows; sha256={source_sha256[:16]}...",
              file=sys.stderr)
        print(f"  Source env: {source_env}", file=sys.stderr)

        try:
            _validate_data_only_merge(source_rows, source_env, sweep_metadata,
                                      sweep_data.get("results", []))
        except RuntimeError as e:
            print(f"ERROR: data-only validation failed:\n  {e}", file=sys.stderr)
            return 1
        print(f"  Validation passed (env match + subset + per-row config match)",
              file=sys.stderr)

        # Dedup guard: refuse if a previous data-only amendment from the SAME
        # source_dir + same source_sha256 already exists. Catches re-runs.
        for prior in existing_amendments:
            if (prior.get("merge_mode") == "data-only"
                    and prior.get("source_dir") == str(args.from_existing_results)
                    and prior.get("source_sha256") == source_sha256
                    and not args.force_supersede):
                print(f"ERROR: a prior data-only amendment from the SAME source "
                      f"({args.from_existing_results}) with the SAME sha256 "
                      f"already exists.\n  Prior amendment_id: {prior['amendment_id']!r}\n"
                      f"  Applied at: {prior['applied_at']}\n"
                      f"  Use --force-supersede to add a new one anyway.",
                      file=sys.stderr)
                return 1

        # ID-collision check (shared with re-run mode below)
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

        # Merge source's explain_checkpoint.jsonl into target's, tagged with
        # this amendment_id, so load_effective_explain returns the amended
        # break_reasons for graph_break rows. WITHOUT this, identify-side
        # amended → graph_break but explain-side stays stale = wrong join.
        # (adversary HIGH gap #3, 2026-05-16)
        source_explain_path = args.from_existing_results / "explain_checkpoint.jsonl"
        source_explain_rows: list[dict] = []
        source_keys = {(r.get("name"), r.get("mode")) for r in source_rows}
        if source_explain_path.exists():
            for line in source_explain_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    er = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (er.get("name"), er.get("mode")) in source_keys:
                    source_explain_rows.append(er)
        # Refuse-loud: if source has graph_break identify rows but no explain
        # data, the merge would silently leave stale explain — that's the bug
        # this guard exists to prevent.
        source_gb_rows = [r for r in source_rows if r.get("status") == "graph_break"]
        if source_gb_rows and not source_explain_rows:
            raise_msg = (
                f"Source has {len(source_gb_rows)} graph_break identify rows but "
                f"NO explain data in {source_explain_path}. Data-only merge would "
                f"leave target's explain stale for these rows. Refusing. "
                f"Run explain on the source dir first, or use re-run mode."
            )
            print(f"ERROR: {raise_msg}", file=sys.stderr)
            return 1

        # Build the data-only amendment record. No fix_commit (no code change);
        # provenance is source_dir + source_sha256.
        amendment = {
            "amendment_id": amendment_id,
            "applied_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "merge_mode": "data-only",
            "source_dir": str(args.from_existing_results),
            "source_sha256": source_sha256,
            "reason": args.reason,
            "trigger": args.trigger,
            "env_constraints": source_env,
            "supersedes": supersedes,
            "row_count": len(source_rows),
            "rows": source_rows,
            "explain_row_count": len(source_explain_rows),
        }
        # Persist (same JSONL append / legacy-JSON rewrite path as re-run mode below).
        if is_jsonl_format(results_path):
            with open(results_path, "a") as f:
                f.write(json.dumps({"_record_type": "amendment", **amendment}) + "\n")
            total_amendments = len(existing_amendments) + 1
        else:
            sweep_data.setdefault("amendments", []).append(amendment)
            tmp = results_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(sweep_data, indent=2))
            os.replace(tmp, results_path)
            total_amendments = len(sweep_data["amendments"])

        # Append source's explain rows to target's explain_checkpoint.jsonl,
        # tagged with this amendment_id.
        if source_explain_rows:
            target_explain = args.sweep_dir / "explain_checkpoint.jsonl"
            with open(target_explain, "a") as f:
                for er in source_explain_rows:
                    f.write(json.dumps({**er, "amendment_id": amendment_id}) + "\n")

        print(f"\n  ✓ Data-only amendment {amendment_id} applied to {results_path}",
              file=sys.stderr)
        print(f"    {len(source_rows)} identify rows merged from {args.from_existing_results}; "
              f"original results[] untouched.", file=sys.stderr)
        if source_explain_rows:
            print(f"    {len(source_explain_rows)} explain rows merged into "
                  f"explain_checkpoint.jsonl.", file=sys.stderr)
        print(f"    Total amendments in this sweep: {total_amendments}", file=sys.stderr)
        return 0

    # ============================================================
    # Mode 1 — re-run (existing path, unchanged)
    # ============================================================
    if not args.models.exists():
        print(f"ERROR: models JSON not found: {args.models}", file=sys.stderr)
        return 1

    # Verify env match (refuses if torch differs)
    print("Verifying environment match against sweep metadata...", file=sys.stderr)
    env_constraints = _verify_env_match(args.python, sweep_metadata)
    print(f"  OK — torch={env_constraints['torch']}", file=sys.stderr)

    # Dedup guard: refuse if a previous amendment with the SAME fix_commit covers
    # the SAME (name, mode) keys. Catches accidental re-runs from misdiagnosed
    # "killed" processes (the disowned child often completed even when the
    # foreground Bash wrapper got SIGTERM'd) — we lost trust by writing 3 dup
    # amendments on 2026-05-04 morning.
    new_keys = set()
    with open(args.models) as f:
        spec_list = json.load(f)
    for spec in spec_list:
        for mode_name in args.modes:
            new_keys.add((spec["name"], mode_name))
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

    # Build the amendment record. merge_mode="re-run" makes it uniform with
    # Mode 2's "data-only" tagging so consumers can switch on a single field
    # rather than inferring from presence/absence of fix_commit.
    amendment = {
        "amendment_id": amendment_id,
        "applied_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "merge_mode": "re-run",
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
