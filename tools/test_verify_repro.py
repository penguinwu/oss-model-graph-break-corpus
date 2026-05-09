#!/usr/bin/env python3
"""Tests for tools/verify_repro.py.

Pins the load-bearing claims of the V1.0 repro-gate (Peng directives
2026-05-09 + adversary case files adv-2026-05-09-113538 + adv-2026-05-09-120800):

- Exactly ONE `python repro=true` fence per body (refuse 0 or >1)
- HTML comment extraction for original_command
- <details> block extraction for expected_signal (per evidence_type)
- Canonicalization: extracted_bytes_sha256 is whitespace + LF invariant (gap 5)
- Classification driven by expected_signal.kind, not exit_code alone (gap 4)
- Required arg matrix per target/evidence_type combination

Run: python3 tools/test_verify_repro.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import verify_repro  # noqa: E402


def _body(*, mre: str = None, original_cmd: str = None,
          mre_signal: dict = None, original_signal: dict = None,
          extra_prose: str = "") -> str:
    """Construct a synthetic corpus-shape body for testing."""
    parts = ["**Repro status:** Reproduces on torch 2.13.0.dev (verified).\n"]
    if extra_prose:
        parts.append(extra_prose + "\n")
    if original_cmd is not None:
        parts.append("\n## Original failure report\n")
        parts.append(f"<!-- original_command: {original_cmd} -->\n")
    if original_signal is not None:
        parts.append(
            "\n<details><summary>Verification signal (original)</summary>\n\n"
            f"`{json.dumps(original_signal)}`\n"
            "</details>\n"
        )
    if mre is not None:
        parts.append("\n## Minimal reproducer (MRE)\n")
        parts.append("```python repro=true\n" + mre + "\n```\n")
    if mre_signal is not None:
        parts.append(
            "\n<details><summary>Verification signal (MRE)</summary>\n\n"
            f"`{json.dumps(mre_signal)}`\n"
            "</details>\n"
        )
    return "".join(parts)


# ── Extraction tests ───────────────────────────────────────────────────────

def test_extracts_one_repro_fence_from_corpus_body():
    mre = "import torch\nprint(torch.__version__)"
    body = _body(mre=mre)
    extracted = verify_repro.extract_mre_from_body(body)
    assert extracted == mre, f"got: {extracted!r}"


def test_refuses_zero_repro_fences():
    body = "## Symptom\nNo MRE here."
    try:
        verify_repro.extract_mre_from_body(body)
        assert False, "should have raised"
    except ValueError as e:
        assert "no `python repro=true` fence" in str(e)


def test_refuses_multiple_repro_fences():
    body = (
        "```python repro=true\nprint(1)\n```\n"
        "and another:\n"
        "```python repro=true\nprint(2)\n```\n"
    )
    try:
        verify_repro.extract_mre_from_body(body)
        assert False, "should have raised"
    except ValueError as e:
        assert "expected exactly 1" in str(e) and "found 2" in str(e)


def test_extracts_original_command_from_html_comment():
    cmd = "python tools/run_experiment.py sweep --models Wav2Vec2Model --modes train"
    body = _body(original_cmd=cmd)
    extracted = verify_repro.extract_original_command_from_body(body)
    assert extracted == cmd


def test_refuses_missing_original_command_comment():
    body = "## No comment here"
    try:
        verify_repro.extract_original_command_from_body(body)
        assert False, "should have raised"
    except ValueError as e:
        assert "no `<!-- original_command:" in str(e)


def test_extracts_expected_signal_for_mre():
    sig = {"kind": "stderr_contains", "fragment": "RuntimeError: aten.bincount"}
    body = _body(mre_signal=sig)
    extracted = verify_repro.extract_expected_signal_from_body(body, "mre")
    assert extracted == sig


def test_extracts_expected_signal_for_original():
    sig = {"kind": "stdout_contains", "fragment": "max_diff=5.4"}
    body = _body(original_signal=sig)
    extracted = verify_repro.extract_expected_signal_from_body(body, "original")
    assert extracted == sig


def test_refuses_signal_with_missing_fields():
    body = (
        "<details><summary>Verification signal (MRE)</summary>\n"
        '`{"kind": "stderr_contains"}`\n'  # no fragment
        "</details>"
    )
    try:
        verify_repro.extract_expected_signal_from_body(body, "mre")
        assert False, "should have raised"
    except ValueError as e:
        assert "missing kind/fragment" in str(e)


# ── Canonicalization tests (gap 5) ─────────────────────────────────────────

def test_canonicalize_strips_leading_and_trailing_whitespace():
    a = "   import torch\nprint(1)\n   "
    b = "import torch\nprint(1)"
    assert verify_repro.canonicalize_extracted(a) == verify_repro.canonicalize_extracted(b)


def test_canonicalize_normalizes_crlf_to_lf():
    crlf = "import torch\r\nprint(1)\r\n"
    lf = "import torch\nprint(1)\n"
    assert verify_repro.canonicalize_extracted(crlf) == verify_repro.canonicalize_extracted(lf)


def test_canonicalize_normalizes_cr_only_to_lf():
    cr = "import torch\rprint(1)\r"
    lf = "import torch\nprint(1)"
    assert verify_repro.canonicalize_extracted(cr) == verify_repro.canonicalize_extracted(lf)


def test_extracted_bytes_sha256_invariant_to_body_prose_changes():
    """Gap 5 anchor: prose around the fence shouldn't change the MRE sha."""
    mre = "import torch\nprint(torch.__version__)"
    body_a = _body(mre=mre)
    body_b = _body(mre=mre, extra_prose="A bunch of additional prose paragraphs here.\n\nMore text.\n")
    extracted_a = verify_repro.extract_mre_from_body(body_a)
    extracted_b = verify_repro.extract_mre_from_body(body_b)
    sha_a = hashlib.sha256(verify_repro.canonicalize_extracted(extracted_a)).hexdigest()
    sha_b = hashlib.sha256(verify_repro.canonicalize_extracted(extracted_b)).hexdigest()
    assert sha_a == sha_b, f"prose change broke sha: {sha_a} vs {sha_b}"


# ── Classification tests (gap 4) ───────────────────────────────────────────

def test_classify_exit_nonzero_stderr_contains_reproduces():
    sig = {"kind": "exit_nonzero+stderr_contains", "fragment": "RuntimeError"}
    assert verify_repro.classify(1, "", "Traceback ... RuntimeError: foo", sig) == "reproduces"


def test_classify_exit_nonzero_stderr_contains_different_failure_when_signal_absent():
    sig = {"kind": "exit_nonzero+stderr_contains", "fragment": "RuntimeError"}
    # Errored with different exception class
    assert verify_repro.classify(1, "", "Traceback ... TypeError: bar", sig) == "different-failure"


def test_classify_exit_nonzero_stderr_contains_does_not_reproduce_on_clean_exit():
    sig = {"kind": "exit_nonzero+stderr_contains", "fragment": "RuntimeError"}
    # Clean exit means symptom didn't fire
    assert verify_repro.classify(0, "OK", "", sig) == "does-not-reproduce"


def test_classify_stderr_contains_reproduces_regardless_of_exit_code():
    """Graph-break MREs typically exit 0 but have TORCH_LOGS in stderr."""
    sig = {"kind": "stderr_contains", "fragment": "graph_break: data-dependent branching"}
    # Exit 0 + signal in stderr = reproduces (the gap-4 case)
    assert verify_repro.classify(0, "", "TORCH_LOGS: graph_break: data-dependent branching",
                                 sig) == "reproduces"


def test_classify_stderr_contains_does_not_reproduce_when_signal_absent():
    sig = {"kind": "stderr_contains", "fragment": "graph_break"}
    assert verify_repro.classify(0, "", "(no logs)", sig) == "does-not-reproduce"


def test_classify_stdout_contains_reproduces_for_numeric_divergence():
    """Numeric-divergence MREs print max_diff to stdout, exit 0."""
    sig = {"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"}
    assert verify_repro.classify(0, "Run done. numeric_max_diff=5.475 (eager vs compiled)",
                                 "", sig) == "reproduces"


def test_classify_stdout_contains_does_not_reproduce_when_value_absent():
    sig = {"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"}
    assert verify_repro.classify(0, "Run done. all values match.", "", sig) == "does-not-reproduce"


def test_classify_unknown_kind_raises():
    sig = {"kind": "unknown_signal_type", "fragment": "x"}
    try:
        verify_repro.classify(0, "", "", sig)
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown expected_signal.kind" in str(e)


# ── Library-arg validation (target/evidence_type combinations) ─────────────

def test_target_corpus_requires_body():
    try:
        verify_repro.verify(
            target="corpus", evidence_type="mre", venv_name="current",
            venv_path=Path("/tmp/fake-python"), case_id="x", body_path=None,
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "--body required" in str(e)


def test_target_upstream_requires_script():
    try:
        verify_repro.verify(
            target="upstream", evidence_type="mre", venv_name="current",
            venv_path=Path("/tmp/fake-python"), case_id="x", script_path=None,
        )
        assert False, "should have raised"
    except ValueError as e:
        assert "--script required" in str(e)


def test_target_upstream_requires_expected_signal_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("import torch")
        script_path = Path(f.name)
    try:
        try:
            verify_repro.verify(
                target="upstream", evidence_type="mre", venv_name="current",
                venv_path=Path("/tmp/fake-python"), case_id="x",
                script_path=script_path, expected_signal_json=None,
            )
            assert False, "should have raised"
        except ValueError as e:
            assert "--expected-signal-json required" in str(e)
    finally:
        script_path.unlink(missing_ok=True)


def test_invalid_target_raises():
    try:
        verify_repro.verify(
            target="invalid", evidence_type="mre", venv_name="current",
            venv_path=Path("/tmp/x"), case_id="x",
        )
        assert False
    except ValueError as e:
        assert "target must be" in str(e)


def test_invalid_evidence_type_raises():
    try:
        verify_repro.verify(
            target="corpus", evidence_type="invalid", venv_name="current",
            venv_path=Path("/tmp/x"), case_id="x",
        )
        assert False
    except ValueError as e:
        assert "evidence_type must be" in str(e)


def test_invalid_venv_name_raises():
    try:
        verify_repro.verify(
            target="corpus", evidence_type="mre", venv_name="invalid",
            venv_path=Path("/tmp/x"), case_id="x",
        )
        assert False
    except ValueError as e:
        assert "venv_name must be" in str(e)


# ── Parse model+mode (lookup-arg derivation) ───────────────────────────────

def test_parse_model_mode_from_well_formed_command():
    cmd = ("python tools/run_experiment.py sweep --models Wav2Vec2Model "
           "--modes train --workers 1 --identify-only")
    model, mode = verify_repro._parse_model_mode(cmd)
    assert model == "Wav2Vec2Model" and mode == "train"


def test_parse_model_mode_returns_none_when_absent():
    cmd = "python tools/run_experiment.py sweep --workers 1"
    model, mode = verify_repro._parse_model_mode(cmd)
    assert model is None and mode is None


# ── Doc/persona pin tests (defer to commit 5; placeholder here) ────────────

# (test_persona_documents_frozen_mre_rule, test_skill_md_documents_step_2_5,
#  test_mode_b_pre_submission_gate_lists_nine_items — all live in commit 5
#  when persona.md / SKILL.md actually carry the new content.)


# ── Runner ──────────────────────────────────────────────────────────────────

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
