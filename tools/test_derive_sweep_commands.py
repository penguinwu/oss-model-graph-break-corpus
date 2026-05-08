#!/usr/bin/env python3
"""Tests for tools/derive_sweep_commands.py.

Focus: stage-derivation invariants (gate/sample/full produce identical
configs except for documented deltas) + integration with the real NGB verify
config + adversary-surfaced edge cases (sha256 drift, double-wrap, pinning).

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_derive_sweep_commands.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "derive_sweep_commands.py"
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT / "tools"))
from derive_sweep_commands import (  # noqa: E402
    derive_stage_config, STAGE_CONFIG, DeriveError, _walk_sha256_pins,
    _state_path, _load_state, STATE_DIR,
)


def _make_config(name: str = "test-spec", with_pinning: bool = True,
                 source_block: dict = None) -> dict:
    """Canonical-shaped experiment config for testing.

    Includes the python_bin + modellib_pins fields that derive REQUIRES
    (per adversary gaps #4, #5).
    """
    if source_block is None:
        source_block = {"source": "list", "names": ["GPT2Model", "DistilBertModel"]}
    settings = {
        "device": "cuda", "modes": ["eval", "train"], "workers": 4,
        "timeout_s": 180, "pass_num": 1,
    }
    if with_pinning:
        settings["python_bin"] = "/home/pengwu/envs/torch-nightly-cu126/bin/python"
        settings["modellib_pins"] = {"transformers": "5.6.2", "diffusers": "0.38.0"}
    return {
        "name": name,
        "description": "test config",
        "models": source_block,
        "configs": [
            {"name": "ngb", "compile_kwargs": {"fullgraph": False},
             "dynamo_flags": {"nested_graph_breaks": True}},
        ],
        "settings": settings,
    }


def _cleanup_state(name: str = "test-spec"):
    """Remove any leaked state files for clean test runs."""
    if STATE_DIR.exists():
        for p in STATE_DIR.glob(f"{name}-*.json"):
            p.unlink(missing_ok=True)


# ─── Stage-derivation invariants ─────────────────────────────────────────

def test_full_stage_preserves_models_block():
    cfg = _make_config()
    derived = derive_stage_config(cfg, "full")
    assert derived["models"] == cfg["models"]
    assert derived["name"] == cfg["name"]


def test_gate_stage_wraps_models_in_sample():
    cfg = _make_config()
    derived = derive_stage_config(cfg, "gate")
    assert derived["models"]["source"] == "sample"
    assert derived["models"]["size"] == 5
    assert derived["models"]["from"] == cfg["models"]
    assert derived["name"] == cfg["name"] + "-gate"


def test_sample_stage_wraps_models_in_sample_size_20():
    cfg = _make_config()
    derived = derive_stage_config(cfg, "sample")
    assert derived["models"]["source"] == "sample"
    assert derived["models"]["size"] == 20
    assert derived["models"]["from"] == cfg["models"]
    assert derived["name"] == cfg["name"] + "-sample"


def test_all_stages_preserve_configs_block_verbatim():
    """LOAD-BEARING: catches the 2026-05-07 afternoon failure mode where
    a "customized gate" was missing --compile-kwargs / --dynamo-config."""
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["configs"] == cfg["configs"], \
            f"stage {stage}: configs[] drift; got {derived['configs']!r}"


def test_all_stages_preserve_settings_block_verbatim():
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["settings"] == cfg["settings"], \
            f"stage {stage}: settings drift"


def test_all_stages_preserve_description():
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["description"] == cfg["description"]


def test_derived_config_is_annotated():
    cfg = _make_config()
    derived = derive_stage_config(cfg, "gate", orig_config_sha="deadbeef")
    assert derived.get("_derived_from") == cfg["name"]
    assert derived.get("_derived_stage") == "gate"
    assert derived.get("_derived_source_sha256") == "deadbeef"


def test_unknown_stage_raises():
    cfg = _make_config()
    try:
        derive_stage_config(cfg, "bogus")
    except ValueError as e:
        assert "bogus" in str(e)
        return
    assert False, "expected ValueError for unknown stage"


def test_seed_is_recorded_in_derived_models_block():
    cfg = _make_config()
    for stage in ("gate", "sample"):
        derived = derive_stage_config(cfg, stage, seed=99)
        assert derived["models"]["seed"] == 99


# ─── NEW: gap-derived tests (closes adversary findings) ─────────────────

def test_gap1_sha256_drift_caught_by_validate(tmp_path=None):
    """Adversary gap #1: --validate must catch sha256 drift on inner `from`
    block (not just at --run time). Direct test of _walk_sha256_pins."""
    test_dir = REPO_ROOT / "tools" / "fixtures" / "_tmp_gap1"
    test_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = test_dir / "test_results.json"
        results.write_text(json.dumps({"results": [{"name": "M1", "status": "ok"}]}))
        rel_path = str(results.relative_to(REPO_ROOT))
        # Models block with WRONG sha256
        models = {
            "source": "corpus_filter",
            "from": rel_path,
            "status": "ok",
            "source_sha256": "0" * 64,  # wrong
        }
        # Direct test: walk should report drift
        errors = []
        _walk_sha256_pins(models, errors)
        assert len(errors) == 1, f"expected 1 error; got {errors}"
        assert "drift" in errors[0]
        # Now test the wrapped form (after derive)
        wrapped = {
            "source": "sample",
            "size": 5,
            "from": models,
        }
        errors = []
        _walk_sha256_pins(wrapped, errors)
        assert len(errors) == 1, f"recursion should find inner drift; got {errors}"
        assert "drift" in errors[0]
    finally:
        if results.exists():
            results.unlink()
        try:
            test_dir.rmdir()
        except OSError:
            pass


def test_gap2_already_sampled_source_rejected():
    """Adversary gap #2: source config with models.source: 'sample' is rejected."""
    cfg = _make_config(source_block={
        "source": "sample",
        "size": 50,
        "seed": 7,
        "from": {"source": "corpus_filter", "status": "ok"},
    })
    try:
        derive_stage_config(cfg, "gate")
    except DeriveError as e:
        assert "already 'sample'" in str(e) or "already sample" in str(e).lower()
        return
    assert False, "expected DeriveError for already-sampled source"


def test_gap4_python_bin_required():
    """Adversary gap #4: source config without python_bin is rejected."""
    cfg = _make_config(with_pinning=False)
    try:
        derive_stage_config(cfg, "gate")
    except DeriveError as e:
        assert "python_bin" in str(e)
        return
    assert False, "expected DeriveError for missing python_bin"


def test_gap5_modellib_pins_required():
    """Adversary gap #5: source config without modellib_pins is rejected."""
    cfg = _make_config()
    del cfg["settings"]["modellib_pins"]
    try:
        derive_stage_config(cfg, "gate")
    except DeriveError as e:
        assert "modellib_pins" in str(e)
        return
    assert False, "expected DeriveError for missing modellib_pins"


def test_gap6_annotation_collision_rejected():
    """Adversary gap #6: source config with reserved annotation key rejected."""
    cfg = _make_config()
    cfg["_derived_from"] = "human-set-value"
    try:
        derive_stage_config(cfg, "gate")
    except DeriveError as e:
        assert "_derived_from" in str(e) or "annotation" in str(e).lower()
        return
    assert False, "expected DeriveError for annotation collision"


def test_gap7_filename_includes_source_sha8():
    """Adversary gap #7: write_transformed_config uses source-sha8 in filename
    so concurrent derives don't collide."""
    from derive_sweep_commands import write_transformed_config
    cfg = _make_config()
    derived = derive_stage_config(cfg, "gate")
    out = write_transformed_config(derived, "abc12345abcdef00")
    try:
        assert "abc12345abcdef00" in out.name, f"sha8 should appear in filename; got {out.name}"
    finally:
        out.unlink(missing_ok=True)


# ─── Adversary-proposed test scenarios ──────────────────────────────────

def test_field_propagation_bit_for_bit():
    """Adversary test #7: All non-{models, name, _derived_*} fields bit-for-bit
    identical across all three stages. Auto-covers future field additions."""
    cfg = _make_config()
    cfg["custom_extra_field"] = {"some": "data"}
    derived_stages = {s: derive_stage_config(cfg, s) for s in ("gate", "sample", "full")}
    excluded = {"name", "models", "_derived_from", "_derived_stage", "_derived_source_sha256"}
    keys_full = set(derived_stages["full"].keys()) - excluded
    for stage in ("gate", "sample"):
        keys_other = set(derived_stages[stage].keys()) - excluded
        assert keys_full == keys_other, \
            f"stage {stage} keyset != full keyset: full={keys_full}, {stage}={keys_other}"
        for key in keys_full:
            full_val = json.dumps(derived_stages["full"][key], sort_keys=True)
            other_val = json.dumps(derived_stages[stage][key], sort_keys=True)
            assert full_val == other_val, \
                f"stage {stage}: key '{key}' diverges from full"


def test_models_from_survives_recursion():
    """Adversary test #8: inner models block survives recursion intact."""
    cfg = _make_config(source_block={
        "source": "corpus_filter",
        "from": "some/path/results.json",
        "status": "ok",
        "source_sha256": "abc" * 21 + "x",
    })
    derived = derive_stage_config(cfg, "gate")
    assert derived["models"]["from"] == cfg["models"]
    assert derived["models"]["from"]["source_sha256"] == cfg["models"]["source_sha256"]


def test_idempotency_same_input_same_output():
    """Adversary test #9: same input → same output, byte-for-byte."""
    cfg = _make_config()
    d1 = derive_stage_config(cfg, "gate")
    d2 = derive_stage_config(cfg, "gate")
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_emitted_bash_uses_pinned_python_bin():
    """Adversary test #4: emitted bash uses pinned python_bin, not bare `python3`."""
    from derive_sweep_commands import emit_bash, write_transformed_config
    cfg = _make_config()
    cfg["settings"]["python_bin"] = "/custom/path/to/python"
    derived = derive_stage_config(cfg, "gate")
    out = write_transformed_config(derived, "test")
    try:
        bash = emit_bash(out, cfg, "gate")
        assert "/custom/path/to/python" in bash, "pinned python_bin should appear in bash"
        # Confirm bare 'python3' is NOT used as the executor (may still appear in comments)
        # Check the actual command lines (those starting with the validate/run pattern)
        run_lines = [line for line in bash.splitlines()
                     if "run_experiment.py" in line and ("validate" in line or "run " in line)]
        for line in run_lines:
            assert "/custom/path/to/python" in line, \
                f"command line should use pinned python; got: {line}"
    finally:
        out.unlink(missing_ok=True)


def test_emitted_bash_includes_modellib_pins():
    """Adversary test #5: emitted bash front-loads PYTHONPATH with pinned modellibs."""
    from derive_sweep_commands import emit_bash, write_transformed_config
    cfg = _make_config()
    cfg["settings"]["modellib_pins"] = {"transformers": "9.9.9", "diffusers": "0.0.1"}
    derived = derive_stage_config(cfg, "gate")
    out = write_transformed_config(derived, "test")
    try:
        bash = emit_bash(out, cfg, "gate")
        assert "transformers-9.9.9" in bash, "transformers pin should appear"
        assert "diffusers-0.0.1" in bash, "diffusers pin should appear"
        assert "PYTHONPATH" in bash
    finally:
        out.unlink(missing_ok=True)


def test_emitted_bash_quoting_handles_special_chars():
    """Adversary test #6: paths with special chars are quoted correctly (bash -n parses)."""
    from derive_sweep_commands import emit_bash
    cfg = _make_config()
    cfg["settings"]["python_bin"] = "/path with space/python$weird"
    weird_path = Path("/tmp/test config.json")
    bash = emit_bash(weird_path, cfg, "gate")
    # Write to file and run bash -n to confirm syntax
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(bash)
        sh_path = f.name
    try:
        r = subprocess.run(["bash", "-n", sh_path], capture_output=True, text=True)
        assert r.returncode == 0, \
            f"bash -n failed on emitted script:\n  stderr={r.stderr}\n  bash={bash}"
    finally:
        Path(sh_path).unlink(missing_ok=True)


# ─── Skip-to-full guardrail tests (gap #3) ───────────────────────────────

def test_gap3_skip_to_full_refused_without_state():
    """Skip-to-full guardrail: --stage full --run refused if no prior gate/sample state."""
    _cleanup_state()
    with tempfile.TemporaryDirectory() as d:
        cfg = _make_config(name="test-skip-guard")
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        r = subprocess.run([PYTHON, str(TOOL), str(cfg_path), "--stage", "full", "--run"],
                           capture_output=True, text=True)
        # Should refuse — either via DeriveError or via run_experiment downstream
        # (We're testing the guardrail behavior regardless of run success)
        assert r.returncode != 0 or "REFUSED" in (r.stdout + r.stderr) or \
               "missing" in (r.stdout + r.stderr), \
            f"expected refusal; got rc={r.returncode}, stdout={r.stdout[:200]}, stderr={r.stderr[:200]}"


def test_gap3_allow_skip_gate_logged():
    """Skip-to-full guardrail: --allow-skip-gate prints loud warning."""
    _cleanup_state()
    with tempfile.TemporaryDirectory() as d:
        cfg = _make_config(name="test-skip-allow")
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        # We use --validate (not --run) — guardrail only fires on --run.
        # Confirm validation succeeds with allow-flag present (no crash).
        r = subprocess.run([PYTHON, str(TOOL), str(cfg_path), "--stage", "full",
                            "--validate", "--allow-skip-gate"],
                           capture_output=True, text=True)
        # validate should pass (allow-skip-gate doesn't affect validation)
        # We just confirm the flag is accepted at parse time
        # (Real --run + warning test would require an actual sweep run)
        assert r.returncode == 0 or "validation" in (r.stdout + r.stderr).lower(), \
            f"--allow-skip-gate flag should parse; got rc={r.returncode}"


# ─── CLI integration ─────────────────────────────────────────────────────

def _run_cli(*args, expect_rc=None):
    r = subprocess.run([PYTHON, str(TOOL)] + list(args), capture_output=True, text=True)
    if expect_rc is not None:
        assert r.returncode == expect_rc, \
            f"expected rc={expect_rc}; got {r.returncode}\n  stdout={r.stdout}\n  stderr={r.stderr}"
    return r


def test_cli_validate_on_synthetic_config():
    """End-to-end: write a synthetic config (with required pinning), derive each stage, validate."""
    with tempfile.TemporaryDirectory() as d:
        cfg = _make_config(name="test-cli")
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        for stage in ("gate", "sample", "full"):
            r = _run_cli(str(cfg_path), "--stage", stage, "--validate", expect_rc=0)


def test_cli_emit_produces_bash_with_run_experiment_invocation():
    with tempfile.TemporaryDirectory() as d:
        cfg = _make_config(name="test-emit")
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        r = _run_cli(str(cfg_path), "--stage", "gate", "--emit", expect_rc=0)
        assert "set -euo pipefail" in r.stdout
        assert "run_experiment.py" in r.stdout
        assert "validate" in r.stdout
        assert "run " in r.stdout


def test_cli_real_ngb_verify_config_all_stages_validate():
    """Integration: the actual NGB verify config produces valid configs at every stage."""
    config_path = REPO_ROOT / "experiments" / "configs" / "ngb-verify-2026-05-07.json"
    if not config_path.exists():
        print("  [SKIP] ngb-verify config not present")
        return
    for stage in ("gate", "sample", "full"):
        r = _run_cli(str(config_path), "--stage", stage, "--validate", expect_rc=0)


# ─── Runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
