#!/usr/bin/env python3
"""Tests for tools/check_doc_consistency.py.

Each rule is tested two ways:
1. positive — current repo state passes (this is the live invariant)
2. negative — synthetic input that violates the rule is caught

We can't fake the repo files easily (they're hardcoded paths in the rule
implementations), so the negative tests use the rule helpers directly OR
monkeypatch REPO_ROOT to a tempdir for end-to-end checks.

Run: python3 tools/test_check_doc_consistency.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_doc_consistency as cdc  # noqa: E402


# Positive: current repo state should be clean ─────────────────────────────────

def test_cohort_codes_passes_on_current_repo():
    assert cdc.rule_cohort_codes() == [], \
        "cohort_codes rule should pass on current repo state"


def test_apply_modes_passes_on_current_repo():
    assert cdc.rule_apply_modes() == [], \
        "apply_modes rule should pass on current repo state"


def test_results_jsonl_field_name_passes_on_current_repo():
    assert cdc.rule_results_jsonl_field_name() == [], \
        "results_jsonl_field_name rule should pass on current repo state"


def test_python_bin_precedence_passes_on_current_repo():
    assert cdc.rule_python_bin_precedence() == [], \
        "python_bin_precedence rule should pass on current repo state"


def test_d1_threshold_notation_passes_on_current_repo():
    assert cdc.rule_d1_threshold_notation() == [], \
        "d1_threshold_notation rule should pass on current repo state"


# Negative: monkeypatch REPO_ROOT to a synthetic tree that violates ───────────

def _setup_synthetic_repo(tmpdir: Path, files: dict[str, str]) -> None:
    for relpath, content in files.items():
        f = tmpdir / relpath
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)


def test_cohort_codes_catches_partial_doc_list():
    """A doc that lists 6 of 9 codes (the contributing.md pre-fix shape) must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "sweep/cohort_validator.py": """
                raise CohortValidationError("BARE_LIST_REJECTED", "x")
                raise CohortValidationError("EMPTY_SOURCE_VERSIONS", "x")
                raise CohortValidationError("PARTIAL_SOURCE_VERSIONS", "x")
                raise CohortValidationError("MISSING_METADATA_KEY", "x")
                raise CohortValidationError("VERSION_MISMATCH", "x")
                raise CohortValidationError("STALE_COHORT", "x")
                raise CohortValidationError("INVALID_MODELS_LIST", "x")
                raise CohortValidationError("FILE_NOT_FOUND", "x")
                raise CohortValidationError("INVALID_JSON", "x")
            """,
            "docs/partial.md": (
                "Codes: `BARE_LIST_REJECTED`, `EMPTY_SOURCE_VERSIONS`, "
                "`PARTIAL_SOURCE_VERSIONS`, `MISSING_METADATA_KEY`, "
                "`VERSION_MISMATCH`, `STALE_COHORT`."
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            violations = cdc.rule_cohort_codes()
            assert len(violations) == 1, \
                f"expected 1 violation for partial list; got {violations}"
            assert "INVALID_MODELS_LIST" in violations[0] and \
                   "FILE_NOT_FOUND" in violations[0] and \
                   "INVALID_JSON" in violations[0], \
                f"violation should name the missing codes; got: {violations[0]}"
        finally:
            cdc.REPO_ROOT = orig


def test_cohort_codes_passes_with_canonical_pointer():
    """A doc with a partial list BUT an explicit canonical-list pointer is OK."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "sweep/cohort_validator.py": """
                raise CohortValidationError("BARE_LIST_REJECTED", "x")
                raise CohortValidationError("FILE_NOT_FOUND", "x")
                raise CohortValidationError("INVALID_JSON", "x")
            """,
            "docs/partial_with_pointer.md": (
                "Codes include `BARE_LIST_REJECTED`. "
                "See `sweep/cohort_validator.py` for the canonical list."
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_cohort_codes() == [], \
                "doc with explicit canonical pointer should pass"
        finally:
            cdc.REPO_ROOT = orig


def test_apply_modes_catches_stale_apply_x_reference():
    """A skill citing APPLY-A when sanity_check skill defines none must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "skills/sweep_sanity_check.md": (
                "## When to apply\n\nPre-launch sample / Mid-sweep peek / Post-completion.\n"
            ),
            "skills/sweep.md": (
                "Use APPLY-A mode after the sample completes.\n"
                "For cohort expansion, see APPLY-D.\n"
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            violations = cdc.rule_apply_modes()
            assert len(violations) == 2, \
                f"expected 2 violations (APPLY-A + APPLY-D); got {violations}"
            joined = "\n".join(violations)
            assert "APPLY-A" in joined and "APPLY-D" in joined
        finally:
            cdc.REPO_ROOT = orig


def test_apply_modes_ignores_revision_log_mentions():
    """References inside revision-log entries (lines with a date) are allowed."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "skills/sweep_sanity_check.md": "## When to apply\n",
            "skills/sweep.md": (
                "| 2026-05-07 | Replaced 6-fixed smoke; added APPLY-D trigger | foo |\n"
                "| 2026-05-08 | Updated APPLY-A/C/D references to v3 names | bar |\n"
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_apply_modes() == [], \
                "revision-log lines (date prefix) should not trigger violations"
        finally:
            cdc.REPO_ROOT = orig


def test_results_jsonl_field_name_catches_model_in_example():
    """A docs/ row with `"model"` + `"config"` + `"mode"` shape must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        bad_line = '{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "ok"}'
        _setup_synthetic_repo(tmpdir, {
            "docs/example.md": "```jsonl\n" + bad_line + "\n```\n",
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            violations = cdc.rule_results_jsonl_field_name()
            assert len(violations) == 1 and "model" in violations[0], \
                f"expected violation; got: {violations}"
        finally:
            cdc.REPO_ROOT = orig


def test_results_jsonl_field_name_ignores_unrelated_model_strings():
    """A doc that mentions `"model"` outside results.jsonl shape is fine."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "docs/other.md": (
                "The corpus has `\"model\": \"X\"` in many places, but only "
                "results.jsonl rows (with `\"config\"` AND `\"mode\"`) matter.\n"
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_results_jsonl_field_name() == [], \
                "non-results.jsonl mentions of \"model\" should not trigger"
        finally:
            cdc.REPO_ROOT = orig


def test_python_bin_precedence_catches_backwards_framing():
    """Doc claiming SWEEP_PYTHON is a fallback must fail (env is PRIMARY)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "tools/run_experiment.py": (
                "python_bin = os.environ.get('SWEEP_PYTHON', "
                "settings.get('python_bin', sys.executable))\n"
            ),
            "docs/bad.md": (
                "| `python_bin` | Used as primary; uses SWEEP_PYTHON env var "
                "as fallback when unset.\n"
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            violations = cdc.rule_python_bin_precedence()
            assert len(violations) >= 1, \
                f"expected violation for backwards framing; got: {violations}"
        finally:
            cdc.REPO_ROOT = orig


def test_python_bin_precedence_passes_with_correct_framing():
    """Doc that explicitly says SWEEP_PYTHON takes precedence is OK."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "tools/run_experiment.py": (
                "python_bin = os.environ.get('SWEEP_PYTHON', "
                "settings.get('python_bin', sys.executable))\n"
            ),
            "docs/good.md": (
                "| `python_bin` | Optional; SWEEP_PYTHON env var takes precedence "
                "if set; otherwise python_bin is used.\n"
            ),
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_python_bin_precedence() == [], \
                "doc with correct precedence framing should pass"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_passes_on_current_repo():
    """Adversary impl-review gap #6: rule covers SKILL/persona/invocations/
    invocations_log.md/RETROSPECTIVE.md — current repo has all of them."""
    assert cdc.rule_subagent_required_fields() == [], \
        "subagent_required_fields rule should pass on current repo state"


def test_subagent_required_fields_catches_missing_skill_md():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # subagents/foo/ exists but has no SKILL.md
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("missing SKILL.md" in x for x in v), \
                f"should report missing SKILL.md; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_catches_missing_invocations_dir():
    """Adversary impl-review gap #6: invocations/ subdir is required."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\ndescription: x\n---\n"
        )
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        # NO invocations/ dir
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("invocations" in x and "subdir" in x for x in v), \
                f"should report missing invocations/ dir; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_catches_missing_persona():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\ndescription: x\n---\n"
        )
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        # NO persona.md
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("persona.md" in x for x in v), \
                f"should report missing persona.md; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_catches_missing_invocations_log():
    """Adversary impl-review gap #6: invocations_log.md (generated) is required."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\ndescription: x\n---\n"
        )
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        # NO invocations_log.md
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("invocations_log.md" in x for x in v), \
                f"should report missing invocations_log.md; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_catches_missing_retrospective():
    """Adversary impl-review gap #6: RETROSPECTIVE.md is required (forcing function)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\ndescription: x\n---\n"
        )
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        # NO RETROSPECTIVE.md
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("RETROSPECTIVE.md" in x for x in v), \
                f"should report missing RETROSPECTIVE.md; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_catches_skill_missing_description():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        # SKILL.md has frontmatter but no `description:` field
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\n---\n# Foo skill\n"
        )
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_subagent_required_fields()
            assert any("description" in x for x in v), \
                f"should report missing description field; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_no_fix_suggestions_passes_on_current_repo():
    """Criterion #4 redefinition (2026-05-08T20:23 ET): persona templates
    must not include Proposed fix / Possible directions / etc. as required
    sections. Current repo has been cleaned."""
    assert cdc.rule_no_fix_suggestions_in_templates() == [], \
        "no_fix_suggestions rule should pass on current repo state"


def test_no_fix_suggestions_catches_proposed_fix_section_in_template():
    """Negative case: a template that instructs the assembler to write a
    'Proposed fix' section must be flagged."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sub = tmpdir / "subagents/file-issue"
        sub.mkdir(parents=True, exist_ok=True)
        # Synthetic persona with a template containing a forbidden header.
        bad_persona = '''# persona

**Corpus bug template:**
```markdown
## Symptom

<2-4 sentences>

## Reproduction

```python
import torch
```

## Proposed fix

<what we think the fix should be>
```

The end.
'''
        (sub / "persona.md").write_text(bad_persona)
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_no_fix_suggestions_in_templates()
            assert len(v) >= 1, f"expected violation; got: {v}"
            assert any("Proposed fix" in x for x in v), \
                f"violation should name `Proposed fix`; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_no_fix_suggestions_catches_possible_directions():
    """Negative case: 'Possible directions' header in a template is forbidden."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sub = tmpdir / "subagents/file-issue"
        sub.mkdir(parents=True, exist_ok=True)
        bad_persona = '''# persona

```markdown
## Symptom

x

## Possible directions

- option a
- option b
```
'''
        (sub / "persona.md").write_text(bad_persona)
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            v = cdc.rule_no_fix_suggestions_in_templates()
            assert any("Possible directions" in x for x in v), \
                f"violation should name `Possible directions`; got: {v}"
        finally:
            cdc.REPO_ROOT = orig


def test_no_fix_suggestions_allows_anti_pattern_mentions_outside_templates():
    """The persona MAY mention forbidden phrases as anti-pattern guidance
    (necessary to enforce them). Only INSIDE template code-fence blocks
    should the patterns be banned."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        sub = tmpdir / "subagents/file-issue"
        sub.mkdir(parents=True, exist_ok=True)
        # Persona with anti-pattern phrases mentioned in PROSE (not in templates)
        ok_persona = '''# persona

The forbidden section headers are: Proposed fix, Possible directions,
Suggested fix, Recommendations, How to fix.

The persona must reject any draft containing these.

**Corpus bug template:**
```markdown
## Symptom

<x>

## Reproduction

<y>
```
'''
        (sub / "persona.md").write_text(ok_persona)
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_no_fix_suggestions_in_templates() == [], \
                "anti-pattern mentions in PROSE (outside template blocks) should be allowed"
        finally:
            cdc.REPO_ROOT = orig


def test_subagent_required_fields_baseline_passes_with_all_files():
    """All required pieces present → rule reports no violations."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        (tmpdir / "subagents" / "foo" / "invocations").mkdir(parents=True)
        (tmpdir / "subagents" / "foo" / "SKILL.md").write_text(
            "---\nname: foo\ndescription: a sub-agent for testing\n---\n# Foo\n"
        )
        (tmpdir / "subagents" / "foo" / "persona.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "invocations_log.md").write_text("x")
        (tmpdir / "subagents" / "foo" / "RETROSPECTIVE.md").write_text("x")
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            assert cdc.rule_subagent_required_fields() == [], \
                "baseline (all files present) should produce no violations"
        finally:
            cdc.REPO_ROOT = orig


def test_d1_threshold_notation_catches_geq_notation():
    """`D1 ... ≥1e-3` or `D1 ... >=1e-3` in any doc must fail."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        _setup_synthetic_repo(tmpdir, {
            "tools/check_cohort_invariants.py": (
                "D1_THRESHOLD = 1e-3\n"
                "if diff > D1_THRESHOLD:\n"
                "    return 'STRICT_FAIL'\n"
            ),
            "docs/bad.md": "D1 catastrophic divergence ≥1e-3 is STRICT_FAIL.\n",
            "docs/bad2.md": "D1 catastrophic divergence >=1e-3 is STRICT_FAIL.\n",
        })
        orig = cdc.REPO_ROOT
        try:
            cdc.REPO_ROOT = tmpdir
            violations = cdc.rule_d1_threshold_notation()
            assert len(violations) == 2, \
                f"expected 2 violations (≥ + >=); got: {violations}"
        finally:
            cdc.REPO_ROOT = orig


# Runner ─────────────────────────────────────────────────────────────────────

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
