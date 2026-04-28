"""Run one discovery configuration end-to-end as an independent process.

Per the parallel runner design (`discovery/parallel-runner-design.md`):
- Each config gets its own per-trial sandbox dir (~50 MB)
- Sandbox holds a clone of the transformers package + per-watched-file copies
- Agent + validator + perf subprocesses inherit `PYTHONPATH=$SANDBOX:...`
- All paths in the prompt rewritten to point at the sandbox
- After the trial: write result.json, optionally clean up sandbox

Usage:
    python -m discovery.run_config \\
        --case vits_model_train --variant V8 --skill none \\
        --trial-label test1 --out-dir /tmp/runs/cfg1/

Designed to be invoked by `discovery/launch_parallel.py` for parallel
batches, but is fully usable standalone for single-config validation
(Gate 2 of the experiment lifecycle).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Hardcoded for now — assumes the corpus venv layout. Compute dynamically
# in a future hardening pass if we ever support multiple venvs.
TF_PACKAGE_ROOT = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers")
TF_PACKAGE_PARENT = TF_PACKAGE_ROOT.parent  # site-packages dir


# ----- sandbox setup -----

def _watched_file_needs_transformers(wf) -> bool:
    """True iff the watched file is inside the transformers package."""
    try:
        wf.path.relative_to(TF_PACKAGE_PARENT)
        return True
    except ValueError:
        return False


def _sandbox_path_for_watched(wf, sandbox_dir: Path) -> Path:
    """Where the watched file lives inside the sandbox."""
    if _watched_file_needs_transformers(wf):
        # Mirror the relative path under sandbox/
        rel = wf.path.relative_to(TF_PACKAGE_PARENT)
        return sandbox_dir / rel
    # Otherwise: flat copy into sandbox by basename
    return sandbox_dir / wf.path.name


def _setup_sandbox(case, sandbox_dir: Path) -> dict:
    """Create the per-trial sandbox.

    Returns a dict mapping original_path -> sandbox_path for every watched
    file (used downstream for prompt rewriting + diff capture).
    """
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # If any watched file is in transformers, we need the whole package
    needs_tf = any(_watched_file_needs_transformers(wf) for wf in case.watched_files)
    if needs_tf:
        tf_sandbox = sandbox_dir / "transformers"
        if tf_sandbox.exists():
            shutil.rmtree(tf_sandbox)
        shutil.copytree(TF_PACKAGE_ROOT, tf_sandbox)

    # Copy each watched file's .original into its sandbox location
    path_map: dict[Path, Path] = {}
    for wf in case.watched_files:
        sandbox_path = _sandbox_path_for_watched(wf, sandbox_dir)
        sandbox_path.parent.mkdir(parents=True, exist_ok=True)
        if wf.original_backup.exists():
            shutil.copyfile(wf.original_backup, sandbox_path)
        path_map[wf.path] = sandbox_path
    return path_map


def _rewrite_prompt(case_body: str, path_map: dict) -> str:
    """Replace every original watched-file path in case_body with its sandbox path."""
    out = case_body
    for orig, sandboxed in path_map.items():
        out = out.replace(str(orig), str(sandboxed))
    return out


def _build_sandboxed_case(case, path_map: dict, sandbox_dir: Path):
    """Return a CaseSpec whose paths point at the sandbox.

    Reuses the original case's case_id, validate_cmd, perf_cmd, baseline_path —
    those are sandbox-agnostic (the env var PYTHONPATH and the file mutations
    on disk drive everything).
    """
    from discovery.runner import CaseSpec, WatchedFile
    new_watched = [
        WatchedFile(path=path_map[wf.path], original_backup=wf.original_backup)
        for wf in case.watched_files
    ]
    return CaseSpec(
        case_id=case.case_id,
        case_body=_rewrite_prompt(case.case_body, path_map),
        watched_files=new_watched,
        validate_cmd=case.validate_cmd,
        perf_cmd=case.perf_cmd,
        perf_cmd_tier2=case.perf_cmd_tier2,
        baseline_path=case.baseline_path,
        baseline_path_tier2=case.baseline_path_tier2,
    )


# ----- agent invocation (mirrors runner._run_agent but accepts custom env) -----

def _run_agent(
    prompt: str,
    add_dirs: list[Path],
    out_dir: Path,
    env: dict,
    timeout_s: int = 1800,
    skill_prompt_file: Path | None = None,
) -> tuple[int, float]:
    """Invoke claude with stream-json output, with `env` (sandbox PYTHONPATH set)."""
    cmd = [
        "claude", "-p", prompt,
        "--permission-mode", "acceptEdits",
        "--allowedTools", "Read Write Edit Bash",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--bare",
        "--disable-slash-commands",
    ]
    if skill_prompt_file is not None:
        cmd.extend(["--append-system-prompt-file", str(skill_prompt_file)])
    for d in add_dirs:
        cmd.extend(["--add-dir", str(d)])

    t0 = time.time()
    with open(out_dir / "stream.jsonl", "w") as out, \
         open(out_dir / "claude_stderr.log", "w") as err:
        try:
            res = subprocess.run(
                cmd, stdout=out, stderr=err, timeout=timeout_s, check=False, env=env,
            )
            exit_code = res.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    return exit_code, time.time() - t0


# ----- diff / mutation check (mirror of runner.py logic, sandboxed paths) -----

def _capture_diff(watched, out_path: Path) -> None:
    """diff -u original_backup live for each watched file."""
    chunks = []
    for wf in watched:
        try:
            res = subprocess.run(
                ["diff", "-u", str(wf.original_backup), str(wf.path)],
                capture_output=True, text=True, check=False,
            )
            if res.stdout:
                chunks.append(res.stdout)
        except Exception as e:
            chunks.append(f"# diff failed for {wf.path}: {e}\n")
    out_path.write_text("\n".join(chunks))


def _check_mutations(watched) -> list[str]:
    """Flag any watched file mutated. Caller decides if mutation was expected."""
    import filecmp
    flags = []
    for wf in watched:
        if not wf.path.exists():
            flags.append(f"watched-file-missing:{wf.path.name}")
            continue
        if not filecmp.cmp(str(wf.path), str(wf.original_backup), shallow=False):
            flags.append(f"file-mutated:{wf.path.name}")
    return flags


# ----- subprocess helpers (validate + perf run with sandbox env) -----

def _run_subprocess_json(cmd: list, env: dict, out_dir: Path, log_prefix: str, flags: list) -> dict | None:
    """Run a subprocess that prints a JSON line; capture stdout/stderr; parse."""
    try:
        res = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600, check=False,
        )
        (out_dir / f"{log_prefix}_stdout.log").write_text(res.stdout)
        (out_dir / f"{log_prefix}_stderr.log").write_text(res.stderr)
        # Find the last JSON line
        for line in reversed(res.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        flags.append(f"{log_prefix}-parse-error")
        return {"raw_stdout_tail": res.stdout[-500:], "parse_error": True}
    except Exception as e:
        flags.append(f"{log_prefix}-crashed:{type(e).__name__}")
        return None


def _add_baseline_compare(perf, baseline_path, flags):
    """Mirror of runner._add_baseline_comparison."""
    if not perf or not baseline_path or not Path(baseline_path).exists():
        return
    try:
        baseline = json.loads(Path(baseline_path).read_text())["baseline"]
        if perf.get("compiled_ms") and baseline.get("compiled_ms"):
            perf["compile_speedup_vs_baseline"] = baseline["compiled_ms"] / perf["compiled_ms"]
        if perf.get("eager_ms") and baseline.get("eager_ms"):
            perf["eager_speedup_vs_baseline"] = baseline["eager_ms"] / perf["eager_ms"]
        if perf.get("compiled_ms") and baseline.get("eager_ms"):
            perf["end_to_end_speedup_vs_baseline_eager"] = baseline["eager_ms"] / perf["compiled_ms"]
    except Exception as e:
        flags.append(f"baseline-compare-failed:{type(e).__name__}")


# ----- fix_survives_perf (mirror of runner.py logic) -----

def _derive_fix_survives_perf(fix_status, perf, perf_tier2):
    if fix_status in ("none", None):
        return None
    p1 = (perf or {}).get("perf_shape_sanity")
    p2 = (perf_tier2 or {}).get("perf_shape_sanity")
    if p1 == "runtime_failure" or p2 == "runtime_failure":
        return False
    if p1 == "ok" and (perf_tier2 is None or p2 == "ok"):
        return True
    return None


# ----- main -----

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, help="case id, e.g. vits_model_train")
    p.add_argument("--variant", required=True, help="variant id, e.g. V8")
    p.add_argument("--skill", default="none",
                   help="'none' (no skill) or path to a skill markdown file")
    p.add_argument("--trial-label", required=True, help="human label for this trial, e.g. cfg1")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output dir; sandbox/ + result.json land here")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-agent timeout in seconds (default 1800 = 30 min)")
    p.add_argument("--keep-sandbox-on-failure", action="store_true", default=True,
                   help="if trial fails, keep sandbox dir for inspection")
    p.add_argument("--no-cleanup", action="store_true",
                   help="never clean up the sandbox (for debugging)")
    p.add_argument("--skip-agent", action="store_true",
                   help="skip claude agent invocation (writes empty stream + zero-byte diff). "
                        "Used by smoke tests to exercise the orchestration without spending agent time.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sandbox = args.out_dir / "sandbox"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Load case spec + variant
    case_mod = importlib.import_module(f"discovery.cases.{args.case}")
    case = case_mod.get_case_spec()

    from discovery.variants import ALL_VARIANTS, compose_prompt
    if args.variant not in ALL_VARIANTS:
        print(f"ERROR: unknown variant {args.variant}", file=sys.stderr)
        return 2
    variant = ALL_VARIANTS[args.variant]

    # 2. Setup sandbox + sandboxed case spec
    print(f"[{args.trial_label}] setting up sandbox at {sandbox}", file=sys.stderr)
    t_setup_start = time.time()
    path_map = _setup_sandbox(case, sandbox)
    sandboxed_case = _build_sandboxed_case(case, path_map, sandbox)
    setup_elapsed = time.time() - t_setup_start
    print(f"[{args.trial_label}] sandbox setup took {setup_elapsed:.1f}s", file=sys.stderr)

    # 2b. Filesystem-integrity baseline (Tier 1 + Tier 3 canaries).
    # Detects post-trial that any monitored dir (site-packages, corpus repo,
    # myclaw-shared) was modified. Sandbox + out_dir are allowed write roots.
    # See discovery/filesystem_integrity.py for design rationale.
    from discovery.filesystem_integrity import (
        DEFAULT_MONITORED_GLOBS, take_snapshot, plant_canaries,
        _resolve_monitored_roots,
    )
    t_fs_start = time.time()
    fs_snapshot = take_snapshot(
        monitored_globs=DEFAULT_MONITORED_GLOBS,
        allowed_roots=[args.out_dir],
    )
    fs_snapshot_path = args.out_dir / "_filesystem_baseline.json"
    fs_snapshot_path.write_text(json.dumps(fs_snapshot.to_dict()))
    canary_record = plant_canaries(
        monitored_dirs=_resolve_monitored_roots(DEFAULT_MONITORED_GLOBS),
        trial_id=args.trial_label,
    )
    fs_setup_elapsed = time.time() - t_fs_start
    print(f"[{args.trial_label}] filesystem baseline: {len(fs_snapshot.files)} files, "
          f"{len(canary_record.canaries)} canaries, took {fs_setup_elapsed:.1f}s",
          file=sys.stderr)

    # 3. Compose prompt
    prompt = compose_prompt(sandboxed_case.case_body, variant)
    (args.out_dir / "prompt.txt").write_text(prompt)

    # 4. Build env with sandbox PYTHONPATH prepended
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{sandbox}:{env.get('PYTHONPATH', '')}"
    env["DISCOVERY_SANDBOX"] = str(sandbox)  # convention; downstream may use

    # 5. Invoke agent (or skip in smoke-test mode)
    if args.skip_agent:
        print(f"[{args.trial_label}] --skip-agent: not invoking claude", file=sys.stderr)
        (args.out_dir / "stream.jsonl").write_text("")
        (args.out_dir / "claude_stderr.log").write_text("(skipped)\n")
        exit_code, elapsed = 0, 0.0
    else:
        skill_file = None if args.skill == "none" else Path(args.skill)
        add_dirs = list({wf.path.parent for wf in sandboxed_case.watched_files})
        print(f"[{args.trial_label}] launching agent (timeout={args.timeout}s)", file=sys.stderr)
        exit_code, elapsed = _run_agent(prompt, add_dirs, args.out_dir, env, args.timeout, skill_file)
        print(f"[{args.trial_label}] agent exited code={exit_code} after {elapsed:.0f}s", file=sys.stderr)

    # 6. Capture diff
    diff_path = args.out_dir / "agent_diff.patch"
    _capture_diff(sandboxed_case.watched_files, diff_path)

    # 7. Mutation check
    flags = _check_mutations(sandboxed_case.watched_files)

    # 8. Validate (subprocess inherits sandbox PYTHONPATH)
    validation = None
    if sandboxed_case.validate_cmd:
        try:
            res = subprocess.run(
                sandboxed_case.validate_cmd, env=env,
                capture_output=True, text=True, timeout=300, check=False,
            )
            (args.out_dir / "validation_stdout.log").write_text(res.stdout)
            (args.out_dir / "validation_stderr.log").write_text(res.stderr)
            try:
                validation = json.loads(res.stdout)
            except json.JSONDecodeError:
                validation = {"raw_stdout": res.stdout, "parse_error": True}
        except Exception as e:
            flags.append(f"validate-crashed:{type(e).__name__}")

    can_run_perf = bool(validation and not validation.get("error"))

    # 9. Perf
    perf = None
    if sandboxed_case.perf_cmd and can_run_perf:
        perf = _run_subprocess_json(sandboxed_case.perf_cmd, env, args.out_dir, "perf", flags)
        _add_baseline_compare(perf, sandboxed_case.baseline_path, flags)

    perf_tier2 = None
    if sandboxed_case.perf_cmd_tier2 and can_run_perf:
        perf_tier2 = _run_subprocess_json(sandboxed_case.perf_cmd_tier2, env, args.out_dir, "perf_tier2", flags)
        _add_baseline_compare(perf_tier2, sandboxed_case.baseline_path_tier2, flags)
        if perf and perf_tier2 and perf.get("speedup") and perf_tier2.get("speedup"):
            if (perf["speedup"] > 1.0) != (perf_tier2["speedup"] > 1.0):
                flags.append("tier1-tier2-direction-mismatch")

    # 10. Compute fix_survives_perf
    fix_status = (validation or {}).get("fix_status")
    fix_survives_perf = _derive_fix_survives_perf(fix_status, perf, perf_tier2)

    # 10b. Filesystem-integrity verification.
    # Compare post-trial state to the pre-trial baseline + verify canaries.
    # Any contamination flags the trial as `EXCLUDED_CONTAMINATED` for
    # downstream merge_results aggregation.
    from discovery.filesystem_integrity import (
        diff_against_snapshot, deep_inspect, verify_canaries,
    )
    t_fs_end = time.time()
    fs_diff = diff_against_snapshot(fs_snapshot, allowed_roots=[args.out_dir])
    canary_result = verify_canaries(canary_record)
    contamination_detected = fs_diff.has_changes or not canary_result.intact
    if contamination_detected:
        # Tier 2: deep inspect (md5 + line counts) for forensic record.
        # We don't have backups for arbitrary site-packages files, so this
        # records the new state only (no diff vs original).
        fs_diff = deep_inspect(fs_diff, backups={})
        flags.append("shared_filesystem_touched")
    fs_verify_elapsed = time.time() - t_fs_end
    fs_report_path = args.out_dir / "_filesystem_contamination.json"
    fs_report = {
        "tier1_baseline_path": str(fs_snapshot_path),
        "monitored_roots": fs_snapshot.monitored_roots,
        "allowed_roots": fs_snapshot.allowed_roots,
        "contamination_detected": contamination_detected,
        "tier1_diff": fs_diff.to_dict(),
        "tier3_canaries": canary_result.to_dict(),
        "tier3_canary_paths": list(canary_record.canaries.keys()),
        "verify_elapsed_s": fs_verify_elapsed,
    }
    fs_report_path.write_text(json.dumps(fs_report, indent=2))
    print(f"[{args.trial_label}] filesystem integrity: "
          f"{'CONTAMINATED' if contamination_detected else 'clean'}, "
          f"{len(fs_diff.changes)} file changes, "
          f"{'canaries OK' if canary_result.intact else f'{len(canary_result.failures)} canary failures'}, "
          f"verify took {fs_verify_elapsed:.1f}s",
          file=sys.stderr)

    # 11. Assemble + write result.json
    result = {
        "case_id": args.case,
        "variant_id": args.variant,
        "trial_label": args.trial_label,
        "skill_arm": "noskill" if args.skill == "none" else Path(args.skill).stem,
        "started_at": started_at,
        "elapsed_s": elapsed,
        "agent_exit_code": exit_code,
        "diff_path": str(diff_path),
        "validation": validation,
        "perf": perf,
        "perf_tier2": perf_tier2,
        "flags": flags,
        "fix_survives_perf": fix_survives_perf,
        "sandbox_dir": str(sandbox),
        "sandbox_setup_s": setup_elapsed,
        "filesystem_integrity": {
            "contamination_detected": contamination_detected,
            "n_changed_files": len(fs_diff.changes),
            "n_canary_failures": len(canary_result.failures),
            "report_path": str(fs_report_path),
        },
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.trial_label}] result written to {args.out_dir / 'result.json'}", file=sys.stderr)

    # 12. Cleanup sandbox
    succeeded = (exit_code in (0, 124)) and validation and not validation.get("error")
    keep = args.no_cleanup or (args.keep_sandbox_on_failure and not succeeded)
    if not keep and sandbox.exists():
        shutil.rmtree(sandbox)
        print(f"[{args.trial_label}] sandbox cleaned up", file=sys.stderr)
    elif keep:
        print(f"[{args.trial_label}] sandbox kept at {sandbox} (failure / no-cleanup)", file=sys.stderr)

    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
