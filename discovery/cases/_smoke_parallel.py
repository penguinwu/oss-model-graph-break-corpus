"""Synthetic case for testing the parallel runner infrastructure.

Trivial MLP — no transformers, no real graph breaks, no perf measurement.
Used by `test_parallel_runs_isolated` smoke test to exercise:
  - sandbox setup with NO transformers dependency
  - prompt rewriting with no path replacements
  - validate.py shim that produces a clean schema
  - 3 parallel run_config processes don't interfere

Not a real discovery case. Don't add to `experiments/` or run via the
production launcher.
"""
from __future__ import annotations

from pathlib import Path

CASE_ID = "_smoke_parallel"
WORK_DIR = Path("/tmp/discovery-runs/_smoke_parallel")


def _ensure_work_dir():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    # Create a trivial "watched file" + .original so sandbox setup has something to copy
    fake_file = WORK_DIR / "fake_model.py"
    if not (WORK_DIR / "fake_model.py.original").exists():
        (WORK_DIR / "fake_model.py.original").write_text("# fake model placeholder\n")
    if not fake_file.exists():
        fake_file.write_text("# fake model placeholder\n")
    # Trivial validate.py that always succeeds
    validate_py = WORK_DIR / "validate.py"
    if not validate_py.exists():
        validate_py.write_text(
            "import json\n"
            "print(json.dumps({\n"
            '  "integrity": {"import_ok": True, "eager_ok": True, "compile_ok": True},\n'
            '  "fix_status": "general",\n'
            '  "details": {"gb_in_agent_run": 0, "gb_under_canonical_inputs": 0,\n'
            '              "gb_call_sites": [], "eager_self_diff": 0.0,\n'
            '              "eager_deterministic": True,\n'
            '              "max_diff_compiled_vs_eager": None,\n'
            '              "max_diff_vs_baseline": None},\n'
            '  "error": None,\n'
            "}))\n"
        )


CASE_BODY = "Smoke-test case body (no real agent work)."


def get_case_spec():
    """Build CaseSpec for the synthetic smoke case."""
    from discovery.runner import CaseSpec, WatchedFile
    _ensure_work_dir()
    return CaseSpec(
        case_id=CASE_ID,
        case_body=CASE_BODY,
        watched_files=[
            WatchedFile(
                path=WORK_DIR / "fake_model.py",
                original_backup=WORK_DIR / "fake_model.py.original",
            ),
        ],
        validate_cmd=["/home/pengwu/envs/torch211/bin/python", str(WORK_DIR / "validate.py")],
        perf_cmd=None,         # no perf for smoke
        perf_cmd_tier2=None,
        baseline_path=None,
        baseline_path_tier2=None,
    )
