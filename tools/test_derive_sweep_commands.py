#!/usr/bin/env python3
"""Tests for tools/derive_sweep_commands.py.

Focus: stage-derivation invariants (gate/sample/full produce identical
configs except for documented deltas) + integration with the real NGB verify
config (validates end-to-end on the actual launch path).

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_derive_sweep_commands.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "derive_sweep_commands.py"
PYTHON = sys.executable

sys.path.insert(0, str(REPO_ROOT / "tools"))
from derive_sweep_commands import derive_stage_config, STAGE_CONFIG  # noqa: E402


def _make_config(name: str = "test-spec") -> dict:
    """Canonical-shaped experiment config for testing."""
    return {
        "name": name,
        "description": "test config",
        "models": {
            "source": "corpus_filter",
            "from": "sweep_results/some/explain_results.json",
            "status": "ok",
            "source_sha256": "deadbeef" * 8,
        },
        "configs": [
            {"name": "ngb", "compile_kwargs": {"fullgraph": False},
             "dynamo_flags": {"nested_graph_breaks": True}},
        ],
        "settings": {
            "device": "cuda", "modes": ["eval", "train"], "workers": 4,
            "timeout_s": 180, "pass_num": 1,
        },
    }


# Stage-derivation invariants ───────────────────────────────────────────────

def test_full_stage_preserves_models_block():
    """Full stage: models block unchanged."""
    cfg = _make_config()
    derived = derive_stage_config(cfg, "full")
    assert derived["models"] == cfg["models"]
    assert derived["name"] == cfg["name"]  # no suffix


def test_gate_stage_wraps_models_in_sample():
    """Gate stage: models wrapped in sample(size=5, from=original)."""
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
    """LOAD-BEARING: configs[] (compile_kwargs, dynamo_flags) MUST propagate.
    This catches the 2026-05-07 afternoon failure mode where a "customized
    gate" was missing --compile-kwargs and --dynamo-config.
    """
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["configs"] == cfg["configs"], \
            f"stage {stage}: configs[] drift; got {derived['configs']!r}"


def test_all_stages_preserve_settings_block_verbatim():
    """LOAD-BEARING: settings (workers, modes, timeout_s, pass_num) MUST propagate."""
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["settings"] == cfg["settings"], \
            f"stage {stage}: settings drift; got {derived['settings']!r}"


def test_all_stages_preserve_description():
    cfg = _make_config()
    for stage in ("gate", "sample", "full"):
        derived = derive_stage_config(cfg, stage)
        assert derived["description"] == cfg["description"]


def test_derived_config_is_annotated():
    """Annotation lets a reader of the transformed config find its origin."""
    cfg = _make_config()
    derived = derive_stage_config(cfg, "gate")
    assert derived.get("_derived_from") == cfg["name"]
    assert derived.get("_derived_stage") == "gate"


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


# CLI integration ─────────────────────────────────────────────────────────

def _run_cli(*args, expect_rc=None):
    r = subprocess.run([PYTHON, str(TOOL)] + list(args), capture_output=True, text=True)
    if expect_rc is not None:
        assert r.returncode == expect_rc, \
            f"expected rc={expect_rc}; got {r.returncode}\n  stdout={r.stdout}\n  stderr={r.stderr}"
    return r


def test_cli_validate_on_synthetic_config():
    """End-to-end: write a synthetic config, derive each stage, validate."""
    with tempfile.TemporaryDirectory() as d:
        # Use models.source: "list" so validate doesn't try to resolve corpus_filter
        cfg = {
            "name": "test-cli",
            "description": "synthetic for CLI test",
            "models": {"source": "list", "names": ["GPT2Model", "DistilBertModel"]},
            "configs": [{"name": "baseline", "compile_kwargs": {}, "dynamo_flags": {}}],
            "settings": {"device": "cuda", "modes": ["eval"], "workers": 4,
                         "timeout_s": 180, "pass_num": 1},
        }
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        for stage in ("gate", "sample", "full"):
            r = _run_cli(str(cfg_path), "--stage", stage, "--validate", expect_rc=0)


def test_cli_emit_produces_bash_with_run_experiment_invocation():
    with tempfile.TemporaryDirectory() as d:
        cfg = {
            "name": "test-emit",
            "description": "test",
            "models": {"source": "list", "names": ["GPT2Model"]},
            "configs": [{"name": "baseline", "compile_kwargs": {}, "dynamo_flags": {}}],
            "settings": {"device": "cuda", "modes": ["eval"], "workers": 4,
                         "timeout_s": 180, "pass_num": 1},
        }
        cfg_path = Path(d) / "test.json"
        cfg_path.write_text(json.dumps(cfg))
        r = _run_cli(str(cfg_path), "--stage", "gate", "--emit", expect_rc=0)
        assert "set -euo pipefail" in r.stdout
        assert "run_experiment.py" in r.stdout
        assert "validate" in r.stdout
        assert "run" in r.stdout


def test_cli_real_ngb_verify_config_all_stages_validate():
    """Integration: the actual NGB verify config produces valid configs at
    every stage (gate, sample, full). This is the load-bearing end-to-end
    test — if ANY stage's transformed config fails to validate, the spec
    system isn't ready for launch."""
    config_path = REPO_ROOT / "experiments" / "configs" / "ngb-verify-2026-05-07.json"
    if not config_path.exists():
        print("  [SKIP] ngb-verify config not present")
        return
    for stage in ("gate", "sample", "full"):
        r = _run_cli(str(config_path), "--stage", stage, "--validate", expect_rc=0)


# Runner ───────────────────────────────────────────────────────────────────

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
