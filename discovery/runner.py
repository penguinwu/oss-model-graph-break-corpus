"""Trial runner for the discovery agent.

Runs one trial = (case, variant, trial_idx). Per-trial responsibilities:

1. Restore *all* watched files from .original backups (model source + test/baseline file).
   This is the v0.2 hardening: pilot 4's harness only restored model source, so a
   flag added to baseline_dbrx.py by one trial silently contaminated subsequent trials.

2. Compose the prompt = case body + variant constraint.

3. Invoke the agent with stream-json + verbose + include-partial-messages.

4. Capture agent_diff.patch for *every* watched file (not just model source).

5. Post-trial mutation check: if any watched file differs from .original AND the
   diff is not present in agent_diff.patch, flag the trial as `test-file-mutated`.

6. Run the case's validate script (graph break count + correctness).

7. Run measure_perf on the modified model (perf row for the fingerprint).

8. Write a result JSON capturing prompt, diff, validation, perf, and any flags.

This module focuses on the harness shape. The actual subprocess invocation of
claude is a thin wrapper (`_run_agent`) that can be swapped out for testing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from discovery.variants import ALL_VARIANTS, Variant, compose_prompt


@dataclass
class WatchedFile:
    """A file the harness must restore between trials and check for mutations."""
    path: Path                  # absolute path to the file
    original_backup: Path       # absolute path to its .original backup


@dataclass
class CaseSpec:
    case_id: str
    case_body: str                            # base prompt (without variant constraint)
    watched_files: list[WatchedFile]
    validate_cmd: list[str]                   # e.g. ["python", "validate.py"]
    perf_cmd: list[str] | None = None         # tier-1 (fast); subprocess that prints perf JSON
    perf_cmd_tier2: list[str] | None = None   # tier-2 (realistic); same shape as perf_cmd
    baseline_path: Path | None = None         # JSON with baseline.eager_ms / .compiled_ms (tier-1)
    baseline_path_tier2: Path | None = None   # JSON with baseline (tier-2)


@dataclass
class TrialResult:
    case_id: str
    variant_id: str
    trial_label: str
    started_at: str
    elapsed_s: float
    agent_exit_code: int
    diff_path: str
    validation: dict | None
    perf: dict | None                          # tier-1 (fast)
    perf_tier2: dict | None = None             # tier-2 (realistic) — None if case opted out
    flags: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _restore_watched_files(watched: list[WatchedFile]) -> None:
    """Copy each .original backup over its target path."""
    for wf in watched:
        if not wf.original_backup.exists():
            raise FileNotFoundError(f"missing .original backup: {wf.original_backup}")
        shutil.copyfile(wf.original_backup, wf.path)


def _capture_full_diff(watched: list[WatchedFile], out_path: Path) -> None:
    """Diff every watched file against its .original; concat into one patch."""
    chunks = []
    for wf in watched:
        try:
            res = subprocess.run(
                ["diff", "-u", str(wf.original_backup), str(wf.path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.stdout:
                chunks.append(res.stdout)
        except Exception as e:
            chunks.append(f"# diff failed for {wf.path}: {e}\n")
    out_path.write_text("\n".join(chunks))


def _check_for_unrecorded_mutations(watched: list[WatchedFile], diff_path: Path) -> list[str]:
    """v0.2 mutation check.

    For each watched file: compare to .original. If it differs but the diff
    isn't reflected in the captured patch, flag it. Since we capture the diff
    of *every* watched file, this collapses to: any diff at all on a file the
    agent shouldn't have touched is recorded — caller decides which files are
    expected vs. unexpected via the flag list.

    Returns a list of flag strings; empty list = clean.
    """
    flags: list[str] = []
    for wf in watched:
        if not wf.path.exists():
            flags.append(f"watched-file-missing:{wf.path.name}")
            continue
        # Cheap content compare via filecmp.
        import filecmp
        if not filecmp.cmp(str(wf.path), str(wf.original_backup), shallow=False):
            # File mutated. We capture all diffs, so this is just informational
            # — the caller can decide whether mutation of this specific file is
            # expected (e.g. modeling_dbrx.py: yes; baseline_dbrx.py: no).
            flags.append(f"file-mutated:{wf.path.name}")
    return flags


def _run_perf_subprocess(
    cmd: list[str], out_dir: Path, log_prefix: str, flags: list[str],
) -> dict | None:
    """Run a perf measurement subprocess; capture stdout/stderr; parse JSON."""
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, check=False,
        )
        (out_dir / f"{log_prefix}_stdout.log").write_text(res.stdout)
        (out_dir / f"{log_prefix}_stderr.log").write_text(res.stderr)
        try:
            return json.loads(res.stdout.strip().split("\n")[-1])
        except (json.JSONDecodeError, IndexError):
            flags.append(f"{log_prefix}-parse-error")
            return {"raw_stdout": res.stdout, "parse_error": True}
    except Exception as e:
        flags.append(f"{log_prefix}-crashed:{type(e).__name__}")
        return None


def _add_baseline_comparison(
    perf: dict | None, baseline_path: Path | None, flags: list[str],
) -> None:
    """In-place enrich `perf` with vs-baseline ratios."""
    if not perf or not baseline_path or not baseline_path.exists():
        return
    try:
        baseline = json.loads(baseline_path.read_text())["baseline"]
        if perf.get("compiled_ms") and baseline.get("compiled_ms"):
            perf["compile_speedup_vs_baseline"] = (
                baseline["compiled_ms"] / perf["compiled_ms"]
            )
        if perf.get("eager_ms") and baseline.get("eager_ms"):
            perf["eager_speedup_vs_baseline"] = (
                baseline["eager_ms"] / perf["eager_ms"]
            )
        if perf.get("compiled_ms") and baseline.get("eager_ms"):
            perf["end_to_end_speedup_vs_baseline_eager"] = (
                baseline["eager_ms"] / perf["compiled_ms"]
            )
    except Exception as e:
        flags.append(f"baseline-compare-failed:{type(e).__name__}")


def _run_agent(
    prompt: str,
    add_dirs: list[Path],
    out_dir: Path,
    timeout_s: int = 1200,
) -> tuple[int, float]:
    """Invoke claude with stream-json output. Returns (exit_code, elapsed_s).

    Captures stream.jsonl + claude_stderr.log into out_dir.
    """
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
    for d in add_dirs:
        cmd.extend(["--add-dir", str(d)])

    t0 = time.time()
    with open(out_dir / "stream.jsonl", "w") as out, \
         open(out_dir / "claude_stderr.log", "w") as err:
        try:
            res = subprocess.run(
                cmd, stdout=out, stderr=err, timeout=timeout_s, check=False,
            )
            exit_code = res.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    elapsed = time.time() - t0
    return exit_code, elapsed


def run_trial(
    case: CaseSpec,
    variant: Variant,
    trial_label: str,
    out_dir: Path,
    timeout_s: int = 1200,
    add_dirs: list[Path] | None = None,
) -> TrialResult:
    """Run one trial end-to-end and return a TrialResult."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. Restore all watched files.
    _restore_watched_files(case.watched_files)

    # 2. Compose prompt.
    prompt = compose_prompt(case.case_body, variant)
    (out_dir / "prompt.txt").write_text(prompt)

    # 3. Invoke agent.
    add_dirs = add_dirs or [wf.path.parent for wf in case.watched_files]
    add_dirs = list(set(add_dirs))  # dedup
    exit_code, elapsed = _run_agent(prompt, add_dirs, out_dir, timeout_s)

    # 4. Capture full diff.
    diff_path = out_dir / "agent_diff.patch"
    _capture_full_diff(case.watched_files, diff_path)

    # 5. Mutation check.
    flags = _check_for_unrecorded_mutations(case.watched_files, diff_path)

    # 6. Validate.
    validation: dict | None = None
    try:
        res = subprocess.run(
            case.validate_cmd, capture_output=True, text=True, timeout=180, check=False,
        )
        (out_dir / "validation_stdout.log").write_text(res.stdout)
        (out_dir / "validation_stderr.log").write_text(res.stderr)
        try:
            validation = json.loads(res.stdout)
        except json.JSONDecodeError:
            validation = {"raw_stdout": res.stdout, "parse_error": True}
    except Exception as e:
        flags.append(f"validate-crashed:{type(e).__name__}")

    # 7. Perf rows (tier-1 always; tier-2 if case opts in).
    # Subprocess so module state from the agent's edits doesn't poison perf.
    can_run_perf = bool(validation and not validation.get("error"))

    perf: dict | None = None
    if case.perf_cmd is not None and can_run_perf:
        perf = _run_perf_subprocess(case.perf_cmd, out_dir, "perf", flags)
        _add_baseline_comparison(perf, case.baseline_path, flags)

    perf_tier2: dict | None = None
    if case.perf_cmd_tier2 is not None and can_run_perf:
        perf_tier2 = _run_perf_subprocess(case.perf_cmd_tier2, out_dir, "perf_tier2", flags)
        _add_baseline_comparison(perf_tier2, case.baseline_path_tier2, flags)
        # Disagreement check: catches strategies whose tier-1 win doesn't survive tier-2.
        if perf and perf_tier2 and perf.get("speedup") and perf_tier2.get("speedup"):
            t1_wins = perf["speedup"] > 1.0
            t2_wins = perf_tier2["speedup"] > 1.0
            if t1_wins != t2_wins:
                flags.append("tier1-tier2-direction-mismatch")

    # 8. Restore again for next trial (defense in depth).
    _restore_watched_files(case.watched_files)

    result = TrialResult(
        case_id=case.case_id,
        variant_id=variant.id,
        trial_label=trial_label,
        started_at=started_at,
        elapsed_s=elapsed,
        agent_exit_code=exit_code,
        diff_path=str(diff_path),
        validation=validation,
        perf=perf,
        perf_tier2=perf_tier2,
        flags=flags,
    )
    (out_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


# ----- self-test -----
if __name__ == "__main__":
    """Smoke test: verify the dual-file restore + mutation check work without
    actually invoking the agent. Use a tmp dir with two fake watched files."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Set up two "watched" files with .original backups.
        f1 = tmp / "model.py"
        f2 = tmp / "baseline.py"
        f1_orig = tmp / "model.py.original"
        f2_orig = tmp / "baseline.py.original"
        f1_orig.write_text("model = 'original'\n")
        f2_orig.write_text("baseline = 'original'\n")
        f1.write_text("model = 'mutated by agent'\n")  # agent mutation
        f2.write_text("baseline = 'original'\n")        # no mutation

        watched = [
            WatchedFile(path=f1, original_backup=f1_orig),
            WatchedFile(path=f2, original_backup=f2_orig),
        ]

        # Pre-restore check: f1 should differ.
        flags_before = _check_for_unrecorded_mutations(watched, tmp / "diff.patch")
        print(f"flags before restore: {flags_before}")
        assert "file-mutated:model.py" in flags_before
        assert all("baseline.py" not in f for f in flags_before)

        # Capture diff.
        _capture_full_diff(watched, tmp / "diff.patch")
        print(f"diff captured ({len((tmp/'diff.patch').read_text())} chars)")

        # Restore.
        _restore_watched_files(watched)

        # Post-restore check: clean.
        flags_after = _check_for_unrecorded_mutations(watched, tmp / "diff.patch")
        print(f"flags after restore: {flags_after}")
        assert flags_after == []

        # Now contaminate baseline (the pilot 4 bug shape) and verify we catch it.
        f2.write_text("baseline = 'contaminated by previous trial'\n")
        flags_contaminated = _check_for_unrecorded_mutations(watched, tmp / "diff.patch")
        print(f"flags with contamination: {flags_contaminated}")
        assert "file-mutated:baseline.py" in flags_contaminated

        print("OK — runner harness self-test passed.")
