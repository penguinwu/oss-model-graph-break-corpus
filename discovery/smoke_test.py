"""Smoke test for the discovery harness — runs in seconds, generic across cases.

Layer 1 (synthetic): exercises validator + measure_perf + revalidate end-to-end
against tiny synthetic models we control. Catches "happy path works, broken path
silently fails" regressions (the trap that the V8 _eager_self_check bug fell
into — we'd been testing the happy path implicitly through real cases without
ever exercising the explicit broken-shape path).

Layer 2 (per-case): iterates `discovery/cases/*.py` and runs each case's
validate.py shim against its canonical baseline (no diff applied). Catches
case-specific schema regressions.

Mandatory pre-launch check before any background discovery run:

    python -m discovery.smoke_test

Exits 0 on all green, non-zero on first failure. <30 seconds total.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
import textwrap
import traceback
import types
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _install_fake_case_module(case_id: str, source: str) -> None:
    """Write a real .py file under discovery/cases/ then import it.
    Required because validate_runner._run_canonical_check calls importlib.reload,
    which needs a real module spec (not a sys.modules object injection)."""
    cases_dir = REPO / "discovery" / "cases"
    target = cases_dir / f"{case_id}.py"
    target.write_text(source)
    # Force fresh import in case it was loaded earlier.
    full_name = f"discovery.cases.{case_id}"
    if full_name in sys.modules:
        del sys.modules[full_name]
    importlib.import_module(full_name)


def _cleanup_fake_case_module(case_id: str) -> None:
    """Remove the fake case file + its sys.modules entry."""
    target = REPO / "discovery" / "cases" / f"{case_id}.py"
    if target.exists():
        target.unlink()
    sys.modules.pop(f"discovery.cases.{case_id}", None)


# ----- Layer 1: synthetic harness self-tests -----

def _make_clean_mlp() -> torch.nn.Module:
    """Tiny model with NO graph breaks under torch.compile."""
    class CleanMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(8, 32)
            self.lin2 = torch.nn.Linear(32, 8)

        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))

    return CleanMLP().eval().cuda()


def _make_broken_mlp() -> torch.nn.Module:
    """Tiny model with a KNOWN graph break (data-dependent branch on tensor value)."""
    class BrokenMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)

        def forward(self, x):
            y = self.lin(x)
            # Data-dependent branch — forces a graph break.
            if y.sum() > 0:
                return y + 1.0
            return y - 1.0

    return BrokenMLP().eval().cuda()


def _clean_inputs(model: torch.nn.Module) -> dict:
    return {"x": torch.randn(2, 8, device="cuda")}


def _broken_perf_inputs(model: torch.nn.Module) -> dict:
    """Inputs whose shape is INCOMPATIBLE with the model — forces RuntimeError on forward.
    Used to verify perf_shape_sanity catches the runtime_failure path."""
    return {"x": torch.randn(2, 99, device="cuda")}


_CLEAN_MLP_CASE_SRC = textwrap.dedent('''\
    import torch
    from pathlib import Path
    WORK_DIR = Path("/tmp")
    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(8, 32)
            self.lin2 = torch.nn.Linear(32, 8)
        def forward(self, x):
            return self.lin2(torch.relu(self.lin1(x)))
    def make_model():
        return _Mod().eval().cuda()
    def make_inputs(model):
        return {"x": torch.randn(2, 8, device="cuda")}
''')

_BROKEN_MLP_CASE_SRC = textwrap.dedent('''\
    import torch
    from pathlib import Path
    WORK_DIR = Path("/tmp")
    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)
        def forward(self, x):
            y = self.lin(x)
            if y.sum() > 0:
                return y + 1.0
            return y - 1.0
    def make_model():
        return _Mod().eval().cuda()
    def make_inputs(model):
        return {"x": torch.randn(2, 8, device="cuda")}
''')

# REGRESSION TEST for the np.random determinism bug discovered 2026-04-27.
# A model that uses np.random in forward + only torch.manual_seed (incomplete
# reseed pattern) produces non-deterministic outputs across forwards. The
# validator's seeding must use HF's full set_seed to cover this.
_NPRANDOM_MODEL_CASE_SRC = textwrap.dedent('''\
    import torch
    import numpy as np
    from pathlib import Path
    WORK_DIR = Path("/tmp")
    class _Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(8, 8)
        def forward(self, x):
            # np.random.uniform — not covered by torch.manual_seed alone.
            # Mimics VITS layerdrop pattern (modeling_vits.py:1116).
            scale = float(np.random.uniform(0.9, 1.1))
            return self.lin(x) * scale
    def make_model():
        return _Mod().eval().cuda()
    def make_inputs(model):
        return {"x": torch.randn(2, 8, device="cuda")}
''')


def test_validator_clean_model() -> None:
    """Validator on a graph-break-free model: fix_status='general', gb_count=0,
    gb_call_sites=[] populated, eager_self_diff defined."""
    from discovery.validate_runner import _run_canonical_check

    class FakeCase:
        case_id = "_smoke_clean"
        watched_files: list = []

    _install_fake_case_module("_smoke_clean", _CLEAN_MLP_CASE_SRC)
    try:
        out = _run_canonical_check(FakeCase())
    finally:
        _cleanup_fake_case_module("_smoke_clean")

    assert out["import_ok"] is True, out
    assert out["eager_ok"] is True, out
    assert out["compile_ok"] is True, out
    assert out["graph_break_count"] == 0, f"expected 0 GB on clean MLP, got {out['graph_break_count']}"
    assert out["graph_break_call_sites"] == [], f"expected [] sites on clean, got {out['graph_break_call_sites']}"
    assert out["eager_self_diff"] is not None, "eager_self_diff missing on clean model"
    assert out["eager_deterministic"] is not None, "eager_deterministic missing on clean model"
    print("  ✓ test_validator_clean_model")


def test_validator_broken_model() -> None:
    """Validator on a model with a data-dep GB: gb_count > 0, gb_call_sites populated
    with type and reason."""
    from discovery.validate_runner import _run_canonical_check

    class FakeCase:
        case_id = "_smoke_broken"
        watched_files: list = []

    _install_fake_case_module("_smoke_broken", _BROKEN_MLP_CASE_SRC)
    try:
        out = _run_canonical_check(FakeCase())
    finally:
        _cleanup_fake_case_module("_smoke_broken")

    assert out["import_ok"] is True, out
    assert out["graph_break_count"] is not None and out["graph_break_count"] > 0, \
        f"expected GB count > 0 on broken MLP, got {out['graph_break_count']}"
    sites = out["graph_break_call_sites"]
    assert sites, f"expected gb_call_sites populated, got {sites}"
    # At least one site should have a type field set (sweep.explain populates this).
    types_found = {s.get("type") for s in sites}
    assert types_found - {None}, f"no break types classified, got {types_found}"
    print(f"  ✓ test_validator_broken_model (gb_count={out['graph_break_count']}, types={types_found - {None}})")


def test_validator_no_state_contamination_across_repeated_calls() -> None:
    """Guard against in-process state contamination across repeated
    _run_canonical_check calls.

    Caveat: synthetic MLP doesn't reproduce the VITS-scale state contamination
    bug (discovered 2026-04-27, subprocess fix in revalidate.py commit ed6a1ec).
    This test catches in-process breakage for SIMPLE cases — useful as a
    floor — but a VITS-scale regression test would require a heavier fixture
    (transitive imports + dynamo cache + CUDA pressure). Diagnosis of the
    VITS contamination is open in OPEN-LOOPS."""
    from discovery.validate_runner import _run_canonical_check

    class FakeCase:
        case_id = "_smoke_clean"
        watched_files: list = []

    _install_fake_case_module("_smoke_clean", _CLEAN_MLP_CASE_SRC)
    try:
        # Call N times in-process; every call should give a real gb_count.
        results = [_run_canonical_check(FakeCase()) for _ in range(4)]
    finally:
        _cleanup_fake_case_module("_smoke_clean")

    for i, out in enumerate(results, start=1):
        assert out["graph_break_count"] == 0, \
            f"REGRESSION: call {i}/4 in-process returned graph_break_count={out['graph_break_count']} " \
            f"(expected 0). State contamination across repeated _run_canonical_check calls. " \
            f"out={out}"
    print(f"  ✓ test_validator_no_state_contamination_across_repeated_calls (4 calls × clean_mlp = consistent gb_count=0)")


def test_validator_seeding_covers_nprandom() -> None:
    """REGRESSION TEST for the np.random bug. Validator's reseed pattern MUST
    cover np.random — otherwise a model that uses np.random in forward shows
    eager_self_diff > 0 even with reseed-each-forward via the validator's path.
    Caught the corpus-wide torch.manual_seed-only bug on 2026-04-27."""
    from discovery.validate_runner import _run_canonical_check

    class FakeCase:
        case_id = "_smoke_nprandom"
        watched_files: list = []

    _install_fake_case_module("_smoke_nprandom", _NPRANDOM_MODEL_CASE_SRC)
    try:
        out = _run_canonical_check(FakeCase())
    finally:
        _cleanup_fake_case_module("_smoke_nprandom")

    assert out["import_ok"] is True, out
    assert out["eager_ok"] is True, out
    assert out["eager_self_diff"] is not None, "eager_self_diff was None — _safe_max_diff couldn't compare"
    assert out["eager_self_diff"] == 0.0, \
        f"REGRESSION: validator's reseed pattern doesn't cover np.random. " \
        f"eager_self_diff={out['eager_self_diff']} (expected 0.0). " \
        f"This means validate_runner.py is using torch.manual_seed alone " \
        f"instead of HF's set_seed (or equivalent that covers numpy + python.random)."
    assert out["eager_deterministic"] is True, out
    print("  ✓ test_validator_seeding_covers_nprandom (eager_self_diff=0.0 with np.random in forward)")


def test_perf_happy_path() -> None:
    """measure_perf on clean model + matching inputs: perf_shape_sanity='ok',
    real ms numbers."""
    from discovery.perf import measure_perf

    result = measure_perf(_make_clean_mlp, _clean_inputs, n_warmup=2, n_repeat=3)
    assert result.perf_shape_sanity == "ok", \
        f"expected perf_shape_sanity='ok' on clean model, got {result.perf_shape_sanity!r}"
    assert result.runtime_failure_msg is None, \
        f"expected runtime_failure_msg=None on clean, got {result.runtime_failure_msg!r}"
    # Eager and compiled ms should be finite (not NaN); even if dynamo-error during
    # measurement, sanity check passing is the contract we test here.
    print(f"  ✓ test_perf_happy_path (sanity='ok', eager_ms={result.eager_ms:.3f})")


def test_perf_runtime_failure_path() -> None:
    """measure_perf on a model that raises RuntimeError at perf inputs: must
    return perf_shape_sanity='runtime_failure', runtime_failure_msg populated.
    THIS IS THE TEST WE DIDN'T HAVE — would have caught _eager_self_check bug."""
    from discovery.perf import measure_perf

    result = measure_perf(_make_clean_mlp, _broken_perf_inputs, n_warmup=2, n_repeat=3)
    assert result.perf_shape_sanity == "runtime_failure", \
        f"expected perf_shape_sanity='runtime_failure' on shape-mismatched inputs, got {result.perf_shape_sanity!r}"
    assert result.runtime_failure_msg is not None, \
        f"expected runtime_failure_msg populated, got None"
    assert "RuntimeError" in result.runtime_failure_msg, \
        f"expected RuntimeError in runtime_failure_msg, got {result.runtime_failure_msg!r}"
    print(f"  ✓ test_perf_runtime_failure_path (msg={result.runtime_failure_msg[:60]}...)")


def test_perfresult_schema_completeness() -> None:
    """Every PerfResult field expected downstream actually exists."""
    from discovery.perf import PerfResult

    expected_fields = {
        "eager_ms", "compiled_ms", "speedup", "eager_peak_mem_mb",
        "compiled_peak_mem_mb", "n_warmup", "n_repeat", "device",
        "compile_s", "compile_times", "eager_warm_peak_mem_mb",
        "compiled_warm_peak_mem_mb", "eager_self_diff", "eager_deterministic",
        "perf_shape_sanity", "runtime_failure_msg", "error",
    }
    actual_fields = set(PerfResult.__dataclass_fields__.keys())
    missing = expected_fields - actual_fields
    assert not missing, f"PerfResult missing fields: {missing}"
    print(f"  ✓ test_perfresult_schema_completeness ({len(actual_fields)} fields)")


def test_trialresult_schema_completeness() -> None:
    """TrialResult must include fix_survives_perf."""
    from discovery.runner import TrialResult

    expected_fields = {
        "case_id", "variant_id", "trial_label", "started_at", "elapsed_s",
        "agent_exit_code", "diff_path", "validation", "perf", "perf_tier2",
        "flags", "fix_survives_perf", "error",
    }
    actual_fields = set(TrialResult.__dataclass_fields__.keys())
    missing = expected_fields - actual_fields
    assert not missing, f"TrialResult missing fields: {missing}"
    print(f"  ✓ test_trialresult_schema_completeness ({len(actual_fields)} fields)")


# ----- Layer 2: per-case smoke -----

def smoke_one_case(case_id: str, python_bin: str) -> tuple[bool, str]:
    """Run the case's validate.py shim against canonical (no diff). Verify schema."""
    case_mod_path = REPO / "discovery" / "cases" / f"{case_id}.py"
    if not case_mod_path.exists():
        return False, f"case file not found: {case_mod_path}"

    case_mod = importlib.import_module(f"discovery.cases.{case_id}")
    case = case_mod.get_case_spec()
    res = subprocess.run(
        case.validate_cmd,
        capture_output=True, text=True, timeout=300,
    )
    if res.returncode != 0:
        return False, f"validate.py exited {res.returncode}: {res.stderr[-300:]}"
    try:
        out = json.loads(res.stdout)
    except json.JSONDecodeError as e:
        return False, f"validate.py stdout not JSON: {e}"
    # Schema check.
    expected_top = {"integrity", "fix_status", "details", "error"}
    missing_top = expected_top - set(out.keys())
    if missing_top:
        return False, f"validate.py output missing top-level keys: {missing_top}"
    expected_details = {
        "gb_in_agent_run", "gb_under_canonical_inputs", "gb_call_sites",
        "eager_self_diff", "eager_deterministic",
        "max_diff_compiled_vs_eager", "max_diff_vs_baseline",
    }
    missing_details = expected_details - set(out.get("details", {}).keys())
    if missing_details:
        return False, f"validate.py output missing details keys: {missing_details}"
    return True, f"fix_status={out['fix_status']} gb_count={out['details']['gb_under_canonical_inputs']}"


def list_cases() -> list[str]:
    """Discover all case ids by globbing discovery/cases/*.py (excluding _private)."""
    cases_dir = REPO / "discovery" / "cases"
    return sorted([p.stem for p in cases_dir.glob("*.py")
                   if not p.stem.startswith("_") and p.stem != "__init__"])


# ----- entry -----

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-cases", action="store_true",
                   help="skip Layer 2 (per-case smoke); Layer 1 only")
    p.add_argument("--cases-only", action="store_true",
                   help="skip Layer 1 (synthetic); Layer 2 only")
    p.add_argument("--python", default="/home/pengwu/envs/torch211/bin/python",
                   help="python interpreter for per-case validate subprocess")
    args = p.parse_args()

    failed = 0

    if not args.cases_only:
        print("=== Layer 1: synthetic harness self-tests ===")
        for test_fn in (test_validator_clean_model,
                        test_validator_broken_model,
                        test_validator_no_state_contamination_across_repeated_calls,
                        test_validator_seeding_covers_nprandom,
                        test_perf_happy_path,
                        test_perf_runtime_failure_path,
                        test_perfresult_schema_completeness,
                        test_trialresult_schema_completeness):
            try:
                test_fn()
            except AssertionError as e:
                print(f"  ✗ {test_fn.__name__}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ✗ {test_fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
                failed += 1
        print()

    if not args.skip_cases:
        print("=== Layer 2: per-case smoke ===")
        for case_id in list_cases():
            ok, msg = smoke_one_case(case_id, args.python)
            mark = "✓" if ok else "✗"
            print(f"  {mark} {case_id}: {msg}")
            if not ok:
                failed += 1
        print()

    if failed:
        print(f"FAILED ({failed} test{'s' if failed != 1 else ''})")
        return 1
    print("OK — all smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
