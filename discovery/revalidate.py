"""Re-evaluate completed discovery trials under the current schema.

Apply each trial's `agent_diff.patch` to clean state, re-run the canonical
check (gap 5 + gap 4 fields), optionally re-run perf (gaps 1+2+3 fields),
compute fix_survives_perf, and write everything to a SEPARATE
`result_revalidated.json` per trial dir (original `result.json` untouched).

This is the post-hoc tool. The validate.py shim now produces the new
schema natively for fresh runs — revalidate.py is only needed to backfill
trials run before the schema change.

Schema produced (in `result_revalidated.json`, alongside the original):

```
{
  "case_id": ..., "variant_id": ..., "trial_label": ...,  // copied from original
  "validation_revalidated": {
    "integrity": {"import_ok", "eager_ok", "compile_ok"},
    "fix_status": "general" | "setup-required" | "none" | "unknown",
    "details": {
      "gb_in_agent_run", "gb_under_canonical_inputs", "gb_call_sites",
      "eager_self_diff", "eager_deterministic",
      "max_diff_compiled_vs_eager", "max_diff_vs_baseline"
    }
  },
  "perf_revalidated": {... PerfResult.to_dict() ... or null ...},
  "perf_tier2_revalidated": {... PerfResult.to_dict() ... or null ...},
  "fix_survives_perf": true | false | null,
  "revalidate_meta": {
    "ran_canonical": bool,
    "ran_perf": bool,
    "diff_applied": bool,
    "agent_baseline_subprocess_ok": bool | null,
    "errors": [str, ...]
  }
}
```

Usage:
  python -m discovery.revalidate --trial-dir <dir>
  python -m discovery.revalidate --case <case_id> --run-id <YYYYMMDD-HHMMSS>
  python -m discovery.revalidate --case <case_id> --run-id <id> --rerun-perf
  python -m discovery.revalidate --case <case_id> --run-id <id> --legacy-only

`--rerun-canonical` is ON by default. `--legacy-only` disables it (back-compat
path that reuses legacy `validation.graph_break_count` only). `--rerun-perf`
is OFF by default (opt-in due to ~5min/trial cost).
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path

DISCOVERY_RESULTS = Path("/tmp/discovery-runs")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ----- helpers -----

def _restore_baseline(case) -> None:
    """Copy each .original backup over its target path."""
    for wf in case.watched_files:
        if wf.original_backup.exists():
            wf.path.write_bytes(wf.original_backup.read_bytes())


def _apply_agent_diff(case, trial_dir: Path) -> tuple[bool, str | None]:
    """Apply agent_diff.patch to clean state. Returns (applied, error)."""
    diff_path = trial_dir / "agent_diff.patch"
    if not diff_path.exists() or diff_path.stat().st_size == 0:
        return False, None

    _restore_baseline(case)

    # Rewrite diff so --- lines point at LIVE paths (original diff was
    # generated against .original snapshots).
    diff_text = diff_path.read_text()
    rewritten_lines = []
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            old_path = line[4:].split("\t")[0]
            for wf in case.watched_files:
                if str(wf.original_backup) == old_path:
                    rewritten_lines.append(f"--- {wf.path}")
                    break
            else:
                rewritten_lines.append(line)
        elif line.startswith("+++ "):
            new_path = line[4:].split("\t")[0]
            rewritten_lines.append(f"+++ {new_path}")
        else:
            rewritten_lines.append(line)
    rewritten = "\n".join(rewritten_lines) + "\n"

    patch_res = subprocess.run(
        ["patch", "-p0", "--no-backup-if-mismatch", "--quiet"],
        input=rewritten, cwd="/",
        capture_output=True, text=True,
    )
    if patch_res.returncode != 0:
        _restore_baseline(case)
        return False, f"patch failed: rc={patch_res.returncode} {patch_res.stderr[:300]}"
    return True, None


def _run_agent_baseline_subprocess(case) -> int | None:
    """Run the agent's edited baseline_*.py via subprocess; parse `graph_break_count=N`.
    Returns the count, or None if not found. Caller is responsible for diff application."""
    baseline_wf = None
    for wf in case.watched_files:
        if "baseline" in wf.path.name:
            baseline_wf = wf
            break
    if baseline_wf is None:
        return None
    try:
        run_res = subprocess.run(
            ["/home/pengwu/envs/torch211/bin/python", str(baseline_wf.path)],
            capture_output=True, text=True, timeout=300,
        )
        text = run_res.stdout + "\n" + run_res.stderr
        m = re.search(r"graph_break_count\s*=\s*(\d+)", text)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _run_perf_subprocess(case_id: str, tier: str) -> dict | None:
    """Invoke `python -m discovery._measure_case --case CASE --tier {fast,realistic}`.
    Returns the perf dict (PerfResult.to_dict() shape), or None if subprocess crashed."""
    try:
        res = subprocess.run(
            ["/home/pengwu/envs/torch211/bin/python", "-m",
             "discovery._measure_case", "--case", case_id, "--tier", tier],
            capture_output=True, text=True, timeout=600,
            cwd=str(REPO),
        )
        # _measure_case prints one JSON line as the last non-empty line.
        for line in reversed(res.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"_subprocess_error": "no JSON line found in stdout",
                "raw_stdout_tail": res.stdout[-500:],
                "raw_stderr_tail": res.stderr[-500:]}
    except Exception as e:
        return {"_subprocess_error": f"{type(e).__name__}: {e}"}


def _run_canonical_subprocess(case) -> dict | None:
    """Run canonical check via the case's validate.py shim in a subprocess.
    Avoids in-process module-state contamination that accumulates when
    multiple trials apply diffs to shared module files (sys.modules cleanup
    in _run_canonical_check is insufficient — first 2 trials work, third
    onward returns gb_count=None)."""
    try:
        res = subprocess.run(
            case.validate_cmd,
            capture_output=True, text=True, timeout=300,
        )
        if res.returncode != 0:
            return {"_subprocess_error": f"validate.py exit {res.returncode}",
                    "raw_stderr_tail": res.stderr[-500:]}
        try:
            v2 = json.loads(res.stdout)
        except json.JSONDecodeError:
            return {"_subprocess_error": "validate.py stdout not JSON",
                    "raw_stdout_tail": res.stdout[-500:]}
        # validate.py shim outputs validation_v2 schema. Map to _run_canonical_check
        # output shape (the dict that _build_result expects).
        details = v2.get("details", {})
        return {
            "import_ok": v2.get("integrity", {}).get("import_ok", False),
            "eager_ok": v2.get("integrity", {}).get("eager_ok", False),
            "compile_ok": v2.get("integrity", {}).get("compile_ok", False),
            "graph_count": None,  # not surfaced by validate_runner.main; only graph_break_count
            "graph_break_count": details.get("gb_under_canonical_inputs"),
            "graph_break_call_sites": details.get("gb_call_sites"),
            "eager_self_diff": details.get("eager_self_diff"),
            "eager_deterministic": details.get("eager_deterministic"),
            "max_diff_compiled_vs_eager_now": details.get("max_diff_compiled_vs_eager"),
            "max_diff_vs_eager_baseline": details.get("max_diff_vs_baseline"),
            "error": v2.get("error"),
        }
    except Exception as e:
        return {"_subprocess_error": f"{type(e).__name__}: {e}"}


def _derive_fix_status(agent_gb: int | None, canonical_gb: int | None) -> str:
    if agent_gb is None:
        if canonical_gb == 0:
            return "general"
        if canonical_gb is not None and canonical_gb > 0:
            return "none"
        return "unknown"
    if agent_gb == 0 and canonical_gb == 0:
        return "general"
    if agent_gb == 0 and canonical_gb is not None and canonical_gb > 0:
        return "setup-required"
    if agent_gb > 0:
        return "none"
    return "unknown"


def _derive_fix_survives_perf(
    fix_status: str,
    perf: dict | None,
    perf_tier2: dict | None,
) -> bool | None:
    """Mirror runner.py's fix_survives_perf logic. Inputs are revalidated perf dicts."""
    if fix_status in ("none", None):
        return None
    perf_sanity = (perf or {}).get("perf_shape_sanity")
    perf_tier2_sanity = (perf_tier2 or {}).get("perf_shape_sanity")
    if perf_sanity == "runtime_failure" or perf_tier2_sanity == "runtime_failure":
        return False
    if perf_sanity == "ok" and (perf_tier2 is None or perf_tier2_sanity == "ok"):
        return True
    return None


# ----- main entry -----

def revalidate_trial(
    trial_dir: Path,
    rerun_canonical: bool = True,
    rerun_perf: bool = False,
) -> dict:
    """Re-evaluate one trial; write result_revalidated.json. Returns the new result dict."""
    result_path = trial_dir / "result.json"
    if not result_path.exists():
        return {"_error": f"no result.json in {trial_dir}"}
    result = json.loads(result_path.read_text())
    case_id = result.get("case_id")
    if not case_id:
        return {"_error": f"no case_id in {result_path}"}

    case_mod = importlib.import_module(f"discovery.cases.{case_id}")
    case = case_mod.get_case_spec()

    errors: list[str] = []
    diff_applied = False
    agent_gb: int | None = None
    canonical: dict | None = None
    perf_t1: dict | None = None
    perf_t2: dict | None = None

    try:
        # 1. Apply agent diff to clean state.
        diff_applied, diff_err = _apply_agent_diff(case, trial_dir)
        if diff_err:
            errors.append(diff_err)
            return _build_result(result, case_id, canonical, agent_gb, perf_t1, perf_t2,
                                  rerun_canonical, rerun_perf, diff_applied, errors,
                                  legacy_canonical_gb=(result.get("validation") or {}).get("graph_break_count"))

        # 2. Run agent's baseline subprocess (always — same as legacy).
        if diff_applied:
            agent_gb = _run_agent_baseline_subprocess(case)

        # 3. Run canonical check via SUBPROCESS (was in-process; switched
        #    2026-04-27 because in-process accumulated module state across
        #    trials — first 2 trials worked, third onward returned
        #    gb_count=None). Subprocess matches what runner.py does for perf
        #    and gives clean per-trial isolation.
        if rerun_canonical:
            canonical = _run_canonical_subprocess(case)
            if canonical is not None and canonical.get("_subprocess_error"):
                errors.append(f"canonical subprocess: {canonical['_subprocess_error']}")
                canonical = None

        # 4. Run perf subprocesses if requested.
        if rerun_perf and diff_applied:
            perf_t1 = _run_perf_subprocess(case_id, "fast")
            if case.perf_cmd_tier2 is not None:
                perf_t2 = _run_perf_subprocess(case_id, "realistic")
    finally:
        # 5. Always restore.
        try:
            _restore_baseline(case)
        except Exception as e:
            errors.append(f"restore failed: {type(e).__name__}: {e}")

    legacy_canonical_gb = (result.get("validation") or {}).get("graph_break_count")
    return _build_result(result, case_id, canonical, agent_gb, perf_t1, perf_t2,
                         rerun_canonical, rerun_perf, diff_applied, errors,
                         legacy_canonical_gb=legacy_canonical_gb)


def _build_result(
    original_result: dict,
    case_id: str,
    canonical: dict | None,
    agent_gb: int | None,
    perf_t1: dict | None,
    perf_t2: dict | None,
    rerun_canonical: bool,
    rerun_perf: bool,
    diff_applied: bool,
    errors: list[str],
    legacy_canonical_gb: int | None,
) -> dict:
    """Assemble the result_revalidated.json contents."""
    if rerun_canonical and canonical is not None:
        canonical_gb = canonical.get("graph_break_count")
        integrity = {
            "import_ok": canonical.get("import_ok", False),
            "eager_ok": canonical.get("eager_ok", False),
            "compile_ok": canonical.get("compile_ok", False),
        }
        details = {
            "gb_in_agent_run": agent_gb,
            "gb_under_canonical_inputs": canonical_gb,
            "gb_call_sites": canonical.get("graph_break_call_sites"),
            "eager_self_diff": canonical.get("eager_self_diff"),
            "eager_deterministic": canonical.get("eager_deterministic"),
            "max_diff_compiled_vs_eager": canonical.get("max_diff_compiled_vs_eager_now"),
            "max_diff_vs_baseline": canonical.get("max_diff_vs_eager_baseline"),
        }
    else:
        # Legacy mode: reuse the original validation field.
        legacy = original_result.get("validation") or {}
        canonical_gb = legacy_canonical_gb
        integrity = {
            "import_ok": legacy.get("import_ok", False),
            "eager_ok": legacy.get("eager_ok", False),
            "compile_ok": legacy.get("compile_ok", False),
        }
        details = {
            "gb_in_agent_run": agent_gb,
            "gb_under_canonical_inputs": canonical_gb,
            "gb_call_sites": None,
            "eager_self_diff": None,
            "eager_deterministic": None,
            "max_diff_compiled_vs_eager": legacy.get("max_diff_compiled_vs_eager_now"),
            "max_diff_vs_baseline": legacy.get("max_diff_vs_eager_baseline"),
        }

    fix_status = _derive_fix_status(agent_gb, canonical_gb)
    fix_survives_perf = _derive_fix_survives_perf(fix_status, perf_t1, perf_t2)

    return {
        "case_id": case_id,
        "variant_id": original_result.get("variant_id"),
        "trial_label": original_result.get("trial_label"),
        "validation_revalidated": {
            "integrity": integrity,
            "fix_status": fix_status,
            "details": details,
        },
        "perf_revalidated": perf_t1,
        "perf_tier2_revalidated": perf_t2,
        "fix_survives_perf": fix_survives_perf,
        "revalidate_meta": {
            "ran_canonical": rerun_canonical and canonical is not None,
            "ran_perf": rerun_perf,
            "diff_applied": diff_applied,
            "agent_baseline_subprocess_ok": agent_gb is not None if diff_applied else None,
            "errors": errors,
        },
    }


def _summary_row(label: str, r: dict) -> str:
    """One-line summary row for the table output."""
    val = r.get("validation_revalidated") or {}
    fs = val.get("fix_status", "?")
    det = val.get("details") or {}
    gb_count = det.get("gb_under_canonical_inputs", "?")
    gb_sites = det.get("gb_call_sites") or []
    gb_types = sorted({s.get("type", "?") for s in gb_sites if s.get("file")}) if gb_sites else []
    gb_type_str = ",".join(gb_types) if gb_types else "(none)"
    perf_sanity = (r.get("perf_revalidated") or {}).get("perf_shape_sanity", "—")
    perf_tier2_sanity = (r.get("perf_tier2_revalidated") or {}).get("perf_shape_sanity", "—")
    fsp = r.get("fix_survives_perf")
    fsp_str = "True" if fsp is True else "False" if fsp is False else "None"
    return f"{label:24s}  {fs:14s}  {fsp_str:5s}  {gb_count!s:8s}  {gb_type_str:20s}  {perf_sanity:16s}  {perf_tier2_sanity:16s}"


def _print_summary(results: dict[str, dict], to_file: Path | None = None) -> None:
    header = f"{'trial':24s}  {'fix_status':14s}  {'fsp':5s}  {'gb_count':8s}  {'gb_type':20s}  {'sanity_t1':16s}  {'sanity_t2':16s}"
    rule = "-" * len(header)
    lines = ["", "=== Revalidation summary ===", "", header, rule]
    for label, r in sorted(results.items()):
        lines.append(_summary_row(label, r))
    lines.append("")
    # Distribution
    from collections import Counter
    fix_status_dist = Counter((r.get("validation_revalidated") or {}).get("fix_status", "?") for r in results.values())
    fsp_dist = Counter(r.get("fix_survives_perf") for r in results.values())
    lines.append("fix_status distribution:")
    for fs, c in fix_status_dist.most_common():
        lines.append(f"  {fs}: {c}")
    lines.append("")
    lines.append("fix_survives_perf distribution:")
    for fsp, c in fsp_dist.most_common():
        lines.append(f"  {fsp!s}: {c}")
    text = "\n".join(lines)
    print(text)
    if to_file:
        to_file.write_text(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trial-dir", help="path to a single trial dir")
    p.add_argument("--case", help="case_id (with --run-id, re-validate all trials in that run)")
    p.add_argument("--run-id", help="run_id for batch mode")
    p.add_argument("--rerun-perf", action="store_true",
                   help="Also re-run perf subprocesses (default off; ~5min/trial)")
    p.add_argument("--legacy-only", action="store_true",
                   help="Skip canonical re-run; reuse legacy validation field (back-compat)")
    args = p.parse_args()

    rerun_canonical = not args.legacy_only

    if args.trial_dir:
        out = revalidate_trial(Path(args.trial_dir),
                               rerun_canonical=rerun_canonical,
                               rerun_perf=args.rerun_perf)
        out_path = Path(args.trial_dir) / "result_revalidated.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Wrote {out_path}")
        print(json.dumps(out, indent=2))
        return 0

    if args.case and args.run_id:
        run_dir = DISCOVERY_RESULTS / args.case / args.run_id
        trial_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir()])
        results = {}
        for td in trial_dirs:
            print(f"--- {td.name} ---", flush=True)
            try:
                r = revalidate_trial(td,
                                     rerun_canonical=rerun_canonical,
                                     rerun_perf=args.rerun_perf)
            except Exception as e:
                r = {"_error": f"{type(e).__name__}: {e}", "_traceback": traceback.format_exc()}
            (td / "result_revalidated.json").write_text(json.dumps(r, indent=2))
            results[td.name] = r
            val = r.get("validation_revalidated") or {}
            fs = val.get("fix_status", "?")
            fsp = r.get("fix_survives_perf")
            print(f"  fix_status={fs}  fix_survives_perf={fsp}", flush=True)
        _print_summary(results, to_file=run_dir / "revalidation_summary.md")
        return 0

    p.error("provide either --trial-dir, or --case + --run-id")


if __name__ == "__main__":
    sys.exit(main())
