#!/usr/bin/env python3
"""verify_repro - extract MRE/original from issue body, run, classify.

Library + CLI. Reusable for filing-time gate (current scope), on-demand
re-validation (Phase 4), and post-sweep bulk walker (Phase 4).

Per Peng directives 2026-05-09 07:18 / 07:27 / 07:40 / 07:55 / 08:04 ET +
adversary case files adv-2026-05-09-113538 (v1) and adv-2026-05-09-120800 (v3).

Two extraction modes:
- --target corpus: body has `python repro=true` fence (MRE) +
  `<!-- original_command: ... -->` HTML comment (original) +
  two `<details>` blocks with expected_signal JSON
- --target upstream: --script IS both the original AND the MRE
  (collapsed to single activity per v3.1 gap 1 disposition;
  --expected-signal-json carries the classification signal)

Two evidence types per cell:
- original: extract original_command, try cache (lookup_sweep_evidence)
  first; fall back to re-running the sweep command
- mre: extract MRE bytes, run as a clean subprocess

Classification driven by expected_signal.kind, NOT exit_code alone (gap 4):
- "exit_nonzero+stderr_contains" — for hard exceptions
- "stderr_contains" — for graph breaks via TORCH_LOGS
- "stdout_contains" — for numeric divergences printing max_diff

Canonicalization (gap 5): extracted_bytes_sha256 is computed over
TEXT INSIDE the fence/comment, whitespace-stripped, LF-normalized —
NOT raw body bytes. Body prose changes don't break sha equality.

Requires Python 3.9+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("ERROR: verify_repro.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent
NIGHTLY_VENV_DEFAULT = Path.home() / "envs" / "torch-nightly-cu128" / "bin" / "python"
STALENESS_DAYS_DEFAULT = 10  # Per Peng 2026-05-09 08:04 ET (was 7 in v2)
RERUN_TIMEOUT_DEFAULT = 1800  # 30 min hard cap on cache-miss reruns


@dataclass
class VerificationResult:
    """Uniform shape for both filing-time and Phase 4 bulk-walker use."""
    case_id: str
    target: str          # corpus | upstream
    evidence_type: str   # original | mre
    venv_name: str       # current | nightly
    venv_path: str
    torch_version: str
    torch_git_version: str
    venv_install_age_days: int | None
    wall_clock_utc: str
    elapsed_s: float
    evidence_source: str  # sweep_results | rerun
    sweep_path: str | None
    sweep_age_days: int | None
    extracted_bytes_sha256: str
    expected_signal: dict
    exit_code: int
    stdout_head_4k: str
    stdout_tail_4k: str
    stderr_head_4k: str
    stderr_tail_4k: str
    classification: str  # reproduces | does-not-reproduce | different-failure


# ── Extraction (corpus body) ───────────────────────────────────────────────

def canonicalize_extracted(s: str) -> bytes:
    """Per gap 5 disposition: strip whitespace + normalize LF.

    Hashed bytes are body-prose-invariant — adding/removing prose around
    the fence/comment doesn't change the sha. Mode B's body revisions can
    rewrite surrounding text without forcing re-verification.
    """
    return s.strip().replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


_REPRO_FENCE_PATTERN = re.compile(
    r"```python\s+repro=true\s*\n(.*?)\n```", re.DOTALL,
)


def extract_mre_from_body(body: str) -> str:
    """Find exactly ONE ```python repro=true fence; return inner text.

    Refuses if 0 (Mode B forgot the marker) or >1 (multiple repro candidates,
    body is ambiguous).
    """
    matches = _REPRO_FENCE_PATTERN.findall(body)
    if len(matches) == 0:
        raise ValueError("no `python repro=true` fence found in body")
    if len(matches) > 1:
        raise ValueError(
            f"expected exactly 1 `python repro=true` fence; found {len(matches)}"
        )
    return matches[0]


_ORIGINAL_COMMAND_PATTERN = re.compile(
    r"<!--\s*original_command:\s*(.*?)\s*-->", re.DOTALL,
)


def extract_original_command_from_body(body: str) -> str:
    """Find the <!-- original_command: ... --> HTML comment."""
    m = _ORIGINAL_COMMAND_PATTERN.search(body)
    if not m:
        raise ValueError(
            "no `<!-- original_command: ... -->` HTML comment found in body"
        )
    return m.group(1).strip()


def _expected_signal_pattern(label: str) -> re.Pattern:
    return re.compile(
        rf"<details><summary>Verification signal \({re.escape(label)}\)"
        r"</summary>\s*\n*\s*`(?P<json>\{.*?\})`",
        re.DOTALL | re.IGNORECASE,
    )


def extract_expected_signal_from_body(body: str, evidence_type: str) -> dict:
    """Find the <details> Verification-signal block matching the evidence_type."""
    label = "MRE" if evidence_type == "mre" else "original"
    m = _expected_signal_pattern(label).search(body)
    if not m:
        raise ValueError(
            f"no `<details>` Verification signal block for '{label}' found in body"
        )
    sig = json.loads(m.group("json"))
    if "kind" not in sig or "fragment" not in sig:
        raise ValueError(f"expected_signal missing kind/fragment: {sig!r}")
    return sig


# ── Classification ─────────────────────────────────────────────────────────

VALID_SIGNAL_KINDS = {"exit_nonzero+stderr_contains", "stderr_contains", "stdout_contains"}

# Per adversary case adv-2026-05-09 (gap 1, real-data validation against 3 dynamo
# issues): expected_signal.fragment must be substring-matchable across runs.
# Unstable patterns (run-to-run drift) silently break verification chains —
# verify_repro classifies as `does-not-reproduce` on the next cycle when the
# fragment changes. The adversary's canonical bad example: issue 99's
# "<built-in method div of type object at 0x..." — the 0x address shifts
# across processes.
_UNSTABLE_FRAGMENT_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]{4,}"), "pointer address"),
    (re.compile(r"\bpid\s*[:= ]?\s*\d+", re.IGNORECASE), "process ID"),
    (re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"), "ISO timestamp"),
    (re.compile(r"\bat line \d+", re.IGNORECASE), "line-number anchor"),
    (re.compile(r"/home/[a-zA-Z0-9_\-]+/"), "absolute home path"),
    (re.compile(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    ), "UUID"),
]


def validate_signal_fragment_stability(fragment: str) -> tuple[bool, str]:
    """Per adversary gap 1 disposition: refuse unstable fragments.

    Returns (ok, reason). ok=True if no unstable pattern matches.
    """
    for pattern, label in _UNSTABLE_FRAGMENT_PATTERNS:
        if pattern.search(fragment):
            return False, (
                f"expected_signal.fragment contains unstable pattern "
                f"({label}): {pattern.pattern!r}. Fragments must be "
                f"substring-matchable across runs. Pick a stable substring "
                f"around (not including) the volatile region. See persona.md "
                f"\"Choosing a STABLE fragment\" guidance."
            )
    return True, ""


def classify(exit_code: int, stdout: str, stderr: str, expected_signal: dict) -> str:
    """Classify per expected_signal.kind. Per gap 4: NOT exit_code alone."""
    kind = expected_signal["kind"]
    fragment = expected_signal["fragment"]
    if kind not in VALID_SIGNAL_KINDS:
        raise ValueError(
            f"unknown expected_signal.kind: {kind!r}; valid: {sorted(VALID_SIGNAL_KINDS)}"
        )
    if kind == "exit_nonzero+stderr_contains":
        if exit_code != 0 and fragment in stderr:
            return "reproduces"
        if exit_code != 0:
            return "different-failure"
        return "does-not-reproduce"
    if kind == "stderr_contains":
        return "reproduces" if fragment in stderr else "does-not-reproduce"
    if kind == "stdout_contains":
        return "reproduces" if fragment in stdout else "does-not-reproduce"
    raise ValueError(f"classify reached unreachable branch for kind={kind!r}")


# ── Venv probing ───────────────────────────────────────────────────────────

def get_venv_torch_info(venv_python: Path) -> tuple[str, str]:
    """Return (torch.__version__, torch.version.git_version)."""
    r = subprocess.run(
        [str(venv_python), "-c",
         "import torch; print(torch.__version__); print(torch.version.git_version)"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"failed to probe torch version in {venv_python}: {r.stderr}"
        )
    lines = r.stdout.strip().splitlines()
    return lines[0], lines[1] if len(lines) > 1 else ""


def get_venv_install_age_days(venv_python: Path) -> int | None:
    """Per gap 7 disposition: MAX of pip-torch site-packages mtime and venv-marker mtime.

    Freshness asks 'when did this venv last change' — most recent install
    activity is what matters, not the oldest part of it.
    """
    candidates: list[float] = []  # epoch seconds

    # Path 1: site-packages/torch dir mtime (proxy for last torch install)
    try:
        r = subprocess.run(
            [str(venv_python), "-m", "pip", "show", "torch"],
            capture_output=True, text=True, timeout=30,
        )
        for line in r.stdout.splitlines():
            if line.startswith("Location:"):
                loc = Path(line.split(":", 1)[1].strip())
                torch_dir = loc / "torch"
                if torch_dir.exists():
                    candidates.append(torch_dir.stat().st_mtime)
    except Exception:
        pass

    # Path 2: venv-marker file (if Peng has set one)
    try:
        venv_root = venv_python.parent.parent  # .../venv-name/bin/python -> .../venv-name
        marker = venv_root / ".install_date"
        if marker.exists():
            candidates.append(marker.stat().st_mtime)
    except Exception:
        pass

    if not candidates:
        return None

    most_recent = max(candidates)  # MAX, not MIN — per gap 7
    age_seconds = time.time() - most_recent
    return int(age_seconds / 86400)


# ── Run ─────────────────────────────────────────────────────────────────────

def run_mre(mre_text: str, venv_python: Path, timeout_s: int) -> tuple[int, str, str, float]:
    """Run MRE in a fresh tempdir under venv_python."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="verify_repro_mre_") as td:
        script_path = Path(td) / "repro.py"
        script_path.write_text(mre_text)
        start = time.time()
        try:
            r = subprocess.run(
                [str(venv_python), str(script_path)],
                capture_output=True, text=True, timeout=timeout_s,
                cwd=td,  # clean cwd; no PYTHONPATH leakage
            )
            elapsed = time.time() - start
            return r.returncode, r.stdout, r.stderr, elapsed
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            stderr += f"\n[verify_repro] TIMEOUT after {timeout_s}s"
            return -1, stdout, stderr, elapsed


def _parse_model_mode(original_command: str) -> tuple[str | None, str | None]:
    """Conservative extract of --models / --modes from the command."""
    parts = original_command.split()
    model = mode = None
    for i, p in enumerate(parts):
        if p == "--models" and i + 1 < len(parts):
            model = parts[i + 1]
        if p == "--modes" and i + 1 < len(parts):
            mode = parts[i + 1]
    return model, mode


def run_original_via_lookup_or_rerun(
    *,
    original_command: str,
    venv_name: str,
    venv_path: Path,
    expected_signal: dict,
    case_id: str,
    within_days: int = STALENESS_DAYS_DEFAULT,
    rerun_timeout_s: int = RERUN_TIMEOUT_DEFAULT,
) -> dict:
    """Try cache lookup first (commit 2's lookup_sweep_evidence); fall back to re-run.

    Per gap 2 disposition: re-run shells out to tools/run_experiment.py sweep
    with --workers 1 (inherits orchestrator's GPU coordination), strict timeout,
    tempdir cleanup. Runner errors classify as different-failure (no separate
    command-error class).
    """
    # Try cache first
    model, mode = _parse_model_mode(original_command)
    if model and mode:
        try:
            sys.path.insert(0, str(REPO_ROOT / "tools"))
            from lookup_sweep_evidence import lookup as _lookup  # noqa: E402
            cached = _lookup(
                model=model, mode=mode, venv_name=venv_name,
                within_days=within_days, expected_signal=expected_signal,
                case_id=case_id,
            )
            if cached is not None:
                return {**cached, "evidence_source": "sweep_results"}
        except ImportError:
            # Commit 2 hasn't shipped yet; fall through to rerun
            pass

    # Cache miss: re-run the sweep command in a tempdir
    import shutil
    import tempfile
    td = tempfile.mkdtemp(prefix="verify_repro_original_")
    try:
        cmd_parts = original_command.split()
        if "--output-dir" not in cmd_parts:
            cmd_parts.extend(["--output-dir", td])
        if "--workers" not in cmd_parts:
            cmd_parts.extend(["--workers", "1"])
        # Substitute venv_path for `python` / `python3` at start
        if cmd_parts and cmd_parts[0] in ("python", "python3"):
            cmd_parts[0] = str(venv_path)

        start = time.time()
        try:
            r = subprocess.run(
                cmd_parts, capture_output=True, text=True,
                timeout=rerun_timeout_s,
                cwd=str(REPO_ROOT),
            )
            elapsed = time.time() - start
            results_jsonl = Path(td) / "results.jsonl"
            return {
                "evidence_source": "rerun",
                "sweep_path": str(results_jsonl) if results_jsonl.exists() else None,
                "sweep_age_days": 0,
                "exit_code": r.returncode,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "elapsed_s": elapsed,
            }
        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start
            stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            stderr += f"\n[verify_repro] TIMEOUT after {rerun_timeout_s}s"
            return {
                "evidence_source": "rerun",
                "sweep_path": None,
                "sweep_age_days": 0,
                "exit_code": -1,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_s": elapsed,
            }
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ── Library entry point ─────────────────────────────────────────────────────

def verify(
    *,
    target: str,
    evidence_type: str,
    venv_name: str,
    venv_path: Path,
    case_id: str,
    body_path: Path | None = None,
    script_path: Path | None = None,
    expected_signal_json: Path | None = None,
    timeout_s: int = 600,
    skip_venv_probe: bool = False,
) -> VerificationResult:
    """Top-level library API. CLI is a thin wrapper.

    skip_venv_probe is a unit-test hook — when True, torch_version /
    torch_git_version / venv_install_age_days are stubbed (no real venv
    needed). Tests that pin extraction / classification / canonicalization
    set skip_venv_probe=True and pre-populate exit_code/stdout/stderr by
    other paths is NOT supported here — that's the test's responsibility
    via direct calls to extract_*/classify functions.
    """
    if target not in ("corpus", "upstream"):
        raise ValueError(f"target must be corpus|upstream; got {target!r}")
    if evidence_type not in ("original", "mre"):
        raise ValueError(f"evidence_type must be original|mre; got {evidence_type!r}")
    if venv_name not in ("current", "nightly"):
        raise ValueError(f"venv_name must be current|nightly; got {venv_name!r}")
    if target == "corpus" and body_path is None:
        raise ValueError("--body required when --target corpus")
    if target == "upstream" and script_path is None:
        raise ValueError("--script required when --target upstream")
    if target == "upstream" and expected_signal_json is None:
        raise ValueError("--expected-signal-json required when --target upstream")

    body_text = body_path.read_text() if body_path else None

    # Extract MRE / original_command + expected_signal
    if target == "corpus":
        if evidence_type == "mre":
            extracted = extract_mre_from_body(body_text)
            expected_signal = extract_expected_signal_from_body(body_text, "mre")
        else:
            extracted = extract_original_command_from_body(body_text)
            expected_signal = extract_expected_signal_from_body(body_text, "original")
    else:  # upstream — script IS both original and MRE (gap 1 disposition)
        extracted = script_path.read_text()
        expected_signal = json.loads(expected_signal_json.read_text())
        if "kind" not in expected_signal or "fragment" not in expected_signal:
            raise ValueError(
                f"expected_signal missing kind/fragment: {expected_signal!r}"
            )

    # Validate signal fragment stability (adversary gap 1 — refuse fragments
    # containing pointer addresses, PIDs, timestamps, UUIDs, line-number
    # anchors, or absolute home paths)
    ok, reason = validate_signal_fragment_stability(expected_signal.get("fragment", ""))
    if not ok:
        raise ValueError(f"unstable expected_signal.fragment: {reason}")

    extracted_bytes = canonicalize_extracted(extracted)
    extracted_sha = hashlib.sha256(extracted_bytes).hexdigest()

    # Probe venv
    if skip_venv_probe:
        torch_version = "stub"
        torch_git_version = "stub"
        venv_age = None
    else:
        torch_version, torch_git_version = get_venv_torch_info(venv_path)
        venv_age = get_venv_install_age_days(venv_path)
        if venv_name == "nightly" and venv_age is not None and venv_age > STALENESS_DAYS_DEFAULT:
            raise RuntimeError(
                f"nightly venv at {venv_path.parent.parent} is {venv_age} days old "
                f"(threshold: {STALENESS_DAYS_DEFAULT} days). Refresh required."
            )

    wall_clock = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if evidence_type == "mre" or target == "upstream":
        exit_code, stdout, stderr, elapsed = run_mre(extracted, venv_path, timeout_s)
        evidence_source = "rerun"
        sweep_path = None
        sweep_age_days = None
    else:
        run_result = run_original_via_lookup_or_rerun(
            original_command=extracted, venv_name=venv_name, venv_path=venv_path,
            expected_signal=expected_signal, case_id=case_id,
        )
        evidence_source = run_result["evidence_source"]
        sweep_path = run_result.get("sweep_path")
        sweep_age_days = run_result.get("sweep_age_days")
        exit_code = run_result["exit_code"]
        stdout = run_result["stdout"]
        stderr = run_result["stderr"]
        elapsed = run_result["elapsed_s"]

    classification = classify(exit_code, stdout, stderr, expected_signal)

    return VerificationResult(
        case_id=case_id, target=target, evidence_type=evidence_type,
        venv_name=venv_name, venv_path=str(venv_path),
        torch_version=torch_version, torch_git_version=torch_git_version,
        venv_install_age_days=venv_age,
        wall_clock_utc=wall_clock, elapsed_s=elapsed,
        evidence_source=evidence_source, sweep_path=sweep_path,
        sweep_age_days=sweep_age_days,
        extracted_bytes_sha256=extracted_sha,
        expected_signal=expected_signal,
        exit_code=exit_code,
        stdout_head_4k=stdout[:4096],
        stdout_tail_4k=stdout[-4096:] if len(stdout) > 4096 else "",
        stderr_head_4k=stderr[:4096],
        stderr_tail_4k=stderr[-4096:] if len(stderr) > 4096 else "",
        classification=classification,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--target", required=True, choices=["corpus", "upstream"])
    parser.add_argument("--evidence-type", required=True, choices=["original", "mre"])
    parser.add_argument("--venv-name", required=True, choices=["current", "nightly"])
    parser.add_argument("--venv-path", required=True, type=Path,
                        help="Path to venv's python interpreter")
    parser.add_argument("--body", type=Path,
                        help="Body markdown (required for --target corpus)")
    parser.add_argument("--script", type=Path,
                        help="Repro script (required for --target upstream)")
    parser.add_argument("--expected-signal-json", type=Path,
                        help="JSON with kind+fragment (required for --target upstream)")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON output here (default: "
                             "/tmp/file-issue-<case-id>-repro-<venv>-<type>.json)")
    args = parser.parse_args()

    try:
        result = verify(
            target=args.target, evidence_type=args.evidence_type,
            venv_name=args.venv_name, venv_path=args.venv_path,
            body_path=args.body, script_path=args.script,
            expected_signal_json=args.expected_signal_json,
            case_id=args.case_id, timeout_s=args.timeout_s,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_path = args.output or Path(
        f"/tmp/file-issue-{args.case_id}-repro-{args.venv_name}-{args.evidence_type}.json"
    )
    out_path.write_text(json.dumps(asdict(result), indent=2) + "\n")
    print(f"Wrote {out_path}")
    print(f"  classification:    {result.classification}")
    print(f"  evidence_source:   {result.evidence_source}")
    print(f"  torch:             {result.torch_version} ({result.torch_git_version[:12]})")
    print(f"  elapsed:           {result.elapsed_s:.1f}s")
    print(f"  extracted sha256:  {result.extracted_bytes_sha256[:16]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
