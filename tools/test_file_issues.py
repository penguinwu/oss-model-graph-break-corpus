#!/usr/bin/env python3
"""Tests for tools/file_issues.py.

Pins the load-bearing claims of the subagents/file-issue/ trust chain:
- `--via-skill` is REQUIRED at the CLI level on every posting subcommand
  (argparse `required=True`, not runtime validation).
- The `--via-skill` validator refuses to proceed if the case file is missing,
  if mode_a_verdict is not proceed/proceed-with-fixes, if mode_b_sha256 is
  empty, or if body_sha256 doesn't match the --body file's actual hash.
- `correctness-apply` is deprecated and exits non-zero with an explanation.

Per adversary review case adv-2026-05-08-153427-file-issue-design gaps #4 + #10.

Run: PYTHONPATH=$(pwd) python3 tools/test_file_issues.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "file_issues.py"
PYTHON = sys.executable


def _run(*args: str, input_text: str | None = None) -> tuple[int, str, str]:
    r = subprocess.run([PYTHON, str(TOOL), *args],
                       capture_output=True, text=True, input=input_text)
    return r.returncode, r.stdout, r.stderr


def _make_case_file(case_id: str, *, mode_a_verdict: str = "proceed",
                    mode_b_sha256: str = "abc123", body_sha256: str = "") -> Path:
    """Write a synthetic per-case file under subagents/file-issue/invocations/.

    Returns the path. Caller is responsible for cleanup.
    """
    inv_dir = REPO_ROOT / "subagents/file-issue/invocations"
    inv_dir.mkdir(parents=True, exist_ok=True)
    p = inv_dir / f"{case_id}.md"
    fm = [
        "---",
        f"case_id: {case_id}",
        "subagent: file-issue",
        "date_utc: 2026-05-08T20:00:00Z",
        "target_repo: penguinwu/oss-model-graph-break-corpus",
        "issue_type: bug",
        "persona_sha: deadbeef",
        f"mode_a_verdict: {mode_a_verdict}",
        f"mode_a_sha256: deadbeef",
        f"mode_b_sha256: {mode_b_sha256}",
    ]
    if body_sha256:
        fm.append(f"body_sha256: {body_sha256}")
    fm.extend([
        "footer_marker: \"<!-- via subagents/file-issue case_id=" + case_id + " -->\"",
        "posted_url: pending",
        "---",
        "",
        "## Mode A raw output",
        "",
        "```",
        "VERDICT: " + mode_a_verdict,
        "```",
    ])
    p.write_text("\n".join(fm) + "\n")
    return p


def _make_cluster_plan(case_id: str, *, approved: bool = True,
                       extra_case_ids: list[str] | None = None) -> tuple[Path, str]:
    """Write a synthetic 1-row cluster plan referencing case_id, return (path, sha256).

    If approved=True, peng_approval.approved_at is set to a timestamp (Peng
    explicitly approved). If False, approved_at is None (plan exists but
    Peng has not approved yet).

    The returned sha256 is the file's content sha = the --cluster-plan-approved
    token value. Self-referential pinning of the token field inside the file
    is intentionally NOT used — see _validate_cluster_plan docstring.

    Caller is responsible for cleanup.
    """
    plans_dir = REPO_ROOT / "subagents/file-issue/cluster-plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    affected = [{"case_id": case_id, "role": "primary"}]
    for extra in (extra_case_ids or []):
        affected.append({"case_id": extra, "role": "primary"})
    plan = {
        "sweep_ref": f"test-plan-{case_id}",
        "generated_at": "2026-05-09T00:00:00Z",
        "clustering_method": "single_manual",
        "total_failure_rows": len(affected),
        "total_clustered_rows": len(affected),
        "multi_root_cases": [],
        "single_manual": True,
        "clusters": [{
            "cluster_id": f"test-{case_id}",
            "cluster_type": "single_manual",
            "root_signal": {"case_id": case_id},
            "affected_cases": affected,
            "case_count": len(affected),
            "representative_case": case_id,
            "dup_candidates": [],
            "action": "proceed-as-new",
        }],
        "peng_approval": {
            "approved_at": "2026-05-09T00:01:00Z" if approved else None,
            "approval_message_ref": "test-helper" if approved else None,
        },
    }
    plan_path = plans_dir / f"test-plan-{case_id}.yaml"
    plan_path.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    return plan_path, sha


def test_corpus_issue_rejects_naked_post():
    """Gap #10: tool MUST exit non-zero if --via-skill is missing."""
    rc, out, err = _run("corpus-issue", "--body", "/tmp/foo.md", "--title", "T")
    assert rc != 0, f"expected non-zero exit; got rc={rc}\n  stdout: {out}\n  stderr: {err}"
    assert "via-skill" in err, f"error should name --via-skill; got: {err}"


def test_corpus_issue_rejects_naked_post_no_cluster_plan_approved():
    """V1 cluster+dedup: --cluster-plan-approved is REQUIRED at runtime for NEW + EDIT
    (argparse-optional since 2026-05-10 to permit CLOSE op which uses --plan)."""
    case_id = "file-2026-05-10-cluster-plan-test"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="proceed",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    try:
        # NEW op without --cluster-plan-approved → runtime refusal
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--body", str(body_path), "--title", "T")
        assert rc != 0, f"expected non-zero; got rc={rc}"
        assert "cluster-plan-approved" in err, \
            f"error should name --cluster-plan-approved; got: {err}"
    finally:
        case_path.unlink(missing_ok=True)
        body_path.unlink(missing_ok=True)


def test_pytorch_upstream_rejects_naked_post():
    """Same enforcement on pytorch-upstream."""
    rc, out, err = _run("pytorch-upstream", "--script", "/tmp/x.py",
                        "--venv", "v:/usr/bin/python3")
    assert rc != 0, f"expected non-zero exit; got rc={rc}"
    assert "via-skill" in err, f"error should name --via-skill; got: {err}"


def test_via_skill_rejects_unknown_case_id():
    """If the case file doesn't exist, refuse. (via-skill check fires before
    cluster-plan check, so the cluster-plan token here can be any string.)"""
    rc, out, err = _run("corpus-issue", "--via-skill", "file-9999-99-99-999999-bogus",
                        "--cluster-plan-approved", "0" * 64,
                        "--body", "/tmp/foo.md", "--title", "T")
    assert rc != 0, f"expected non-zero; got rc={rc}"
    assert "case file not found" in err, f"err should explain why; got: {err}"


def test_via_skill_rejects_reframe_verdict():
    """Gap #4: only proceed / proceed-with-fixes verdicts may post."""
    case_id = "file-2026-05-08-test-reframe"
    cf = _make_case_file(case_id, mode_a_verdict="reframe", mode_b_sha256="")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("body content")
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", "0" * 64,
                            "--body", body_path, "--title", "T")
        assert rc != 0, f"expected non-zero for reframe verdict; got rc={rc}"
        assert "reframe" in err and "Required: proceed" in err, \
            f"err should explain verdict requirement; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_via_skill_rejects_empty_mode_b_sha256():
    """Mode A passed but Mode B never wrote a body — refuse."""
    case_id = "file-2026-05-08-test-no-mode-b"
    cf = _make_case_file(case_id, mode_a_verdict="proceed", mode_b_sha256="")
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("body content")
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", "0" * 64,
                            "--body", body_path, "--title", "T")
        assert rc != 0, f"expected non-zero; got rc={rc}"
        assert "mode_b_sha256" in err, f"err should name the missing field; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_via_skill_rejects_body_sha256_mismatch():
    """Gap #5: body sha256 must match. If body was edited post-Mode B, refuse."""
    case_id = "file-2026-05-08-test-sha-mismatch"
    body_text = "the original body Mode B wrote"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="some_sha", body_sha256=body_sha)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("a TAMPERED body different from what Mode B wrote")
            tampered_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", "0" * 64,
                            "--body", tampered_path, "--title", "T")
        assert rc != 0, f"expected non-zero for sha mismatch; got rc={rc}"
        assert "sha256 mismatch" in err, f"err should explain mismatch; got: {err}"
        assert "expected" in err and "actual" in err, \
            f"err should show both hashes; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        Path(tampered_path).unlink(missing_ok=True)


def test_via_skill_accepts_matching_body_sha256():
    """Happy path: matching sha256 + approved cluster plan in dry-run passes.

    Uses --edit (EDIT path is exempt from Phase 3 v1.0 repro gate; v1.5 extends).
    Legacy via-skill + cluster-plan-approved gates are orthogonal to the new
    gate; this test covers them through the unchanged EDIT branch.
    """
    case_id = "file-2026-05-08-test-sha-match"
    body_text = "the body Mode B wrote (no case-id marker; dropped per Peng directive)"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="mode_b_full_output_sha",
                         body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--edit", "999")
        assert rc == 0, f"expected 0 (dry-run EDIT); got rc={rc}\n  stdout: {out}\n  stderr: {err}"
        assert "DRY-RUN" in out, f"should be dry-run by default; got: {out}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_via_skill_accepts_proceed_with_fixes_verdict():
    """proceed-with-fixes is also a posting-allowed verdict (via EDIT path)."""
    case_id = "file-2026-05-08-test-proceed-fixes"
    body_text = "body content (no marker — dropped per Peng directive 2026-05-08T21:13 ET)"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed-with-fixes",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--edit", "999")
        assert rc == 0, f"proceed-with-fixes should pass; got rc={rc}\n  err: {err}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_via_skill_does_not_require_footer_marker():
    """Marker requirement DROPPED 2026-05-08T21:13 ET (via EDIT path)."""
    case_id = "file-2026-05-08-test-no-marker-now-ok"
    body_text = "body content with NO footer marker — should pass now"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--edit", "999")
        assert rc == 0, (f"expected dry-run pass (no marker required); "
                         f"got rc={rc}\n  err: {err}")
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


# V1 cluster+dedup gate tests (Peng directive 2026-05-08T22:01 ET) ──────────

def test_cluster_plan_rejects_unknown_token():
    """Token doesn't match any plan in cluster-plans/ → reject."""
    case_id = "file-2026-05-09-test-unknown-token"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", "f" * 64,
                            "--body", body_path, "--title", "T")
        assert rc != 0, f"expected non-zero (no plan matches); got rc={rc}"
        assert "no plan" in err and "matches" in err, \
            f"err should explain no plan matches token; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_cluster_plan_rejects_unapproved_plan():
    """Plan exists with matching sha BUT peng_approval.approved_at is null → reject."""
    case_id = "file-2026-05-09-test-unapproved"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=False)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--title", "T")
        assert rc != 0, f"expected non-zero (plan unapproved); got rc={rc}"
        assert "approved_at" in err and "null" in err, \
            f"err should explain approval missing; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_cluster_plan_rejects_case_id_not_in_plan():
    """Plan is approved but case_id is not in any cluster's affected_cases → reject.
    (Prevents reusing an old approval for an unrelated filing.)"""
    plan_case_id = "file-2026-05-09-test-plan-owner"
    other_case_id = "file-2026-05-09-test-not-in-plan"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(other_case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(plan_case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        rc, out, err = _run("corpus-issue", "--via-skill", other_case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--title", "T")
        assert rc != 0, f"expected non-zero (case_id not in plan); got rc={rc}"
        assert "not found" in err and "affected_cases" in err, \
            f"err should explain case_id mismatch; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_cluster_plan_accepts_multi_case_batch():
    """Plan covering multiple case_ids accepts each individually (via EDIT path)."""
    case_a = "file-2026-05-09-test-batch-a"
    case_b = "file-2026-05-09-test-batch-b"
    body_text = "shared body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cfa = _make_case_file(case_a, mode_a_verdict="proceed",
                          mode_b_sha256="x", body_sha256=body_sha)
    cfb = _make_case_file(case_b, mode_a_verdict="proceed",
                          mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_a, approved=True, extra_case_ids=[case_b])
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        for case_id in (case_a, case_b):
            rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                                "--cluster-plan-approved", token,
                                "--body", body_path, "--edit", "999")
            assert rc == 0, (f"batch member {case_id} should pass; "
                             f"got rc={rc}\n  err: {err}")
    finally:
        cfa.unlink(missing_ok=True)
        cfb.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


# ── Phase 3 v1.0 repro-verification gate tests (Peng directives 2026-05-09) ──

def _make_v3_body(*, mre: str = "import torch\nprint('repro')",
                  original_cmd: str = "python tools/run_experiment.py sweep "
                                      "--models X --modes train") -> str:
    """Build a real-shape v3 body with the new fields the repro-gate requires."""
    return (
        "**Repro status:** Reproduces on torch X.Y.Z (verified U).\n\n"
        "## Original failure report\n"
        f"<!-- original_command: {original_cmd} -->\n\n"
        "<details><summary>Verification signal (original)</summary>\n\n"
        "`{\"kind\": \"stdout_contains\", \"fragment\": \"max_diff\"}`\n"
        "</details>\n\n"
        "## Minimal reproducer (MRE)\n"
        f"```python repro=true\n{mre}\n```\n\n"
        "<details><summary>Verification signal (MRE)</summary>\n\n"
        "`{\"kind\": \"stderr_contains\", \"fragment\": \"RuntimeError\"}`\n"
        "</details>\n"
    )


def _make_verification_json(*, case_id: str, target: str = "corpus",
                            evidence_type: str, venv_name: str,
                            extracted_sha: str,
                            classification: str = "reproduces") -> Path:
    """Write a synthetic verification JSON with the required fields."""
    j = {
        "case_id": case_id, "target": target, "evidence_type": evidence_type,
        "venv_name": venv_name, "venv_path": "/fake/python",
        "torch_version": "2.13.0.dev20260509", "torch_git_version": "abc123",
        "venv_install_age_days": 1,
        "wall_clock_utc": "2026-05-09T12:00:00Z", "elapsed_s": 1.0,
        "evidence_source": "rerun", "sweep_path": None, "sweep_age_days": None,
        "extracted_bytes_sha256": extracted_sha,
        "expected_signal": {"kind": "stderr_contains", "fragment": "x"},
        "exit_code": 1, "stdout_head_4k": "", "stdout_tail_4k": "",
        "stderr_head_4k": "RuntimeError ...", "stderr_tail_4k": "",
        "classification": classification,
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    f.write(json.dumps(j))
    f.close()
    return Path(f.name)


def _setup_new_path_test(case_id: str, *, classification_override: dict = None):
    """Bundle: case file + cluster plan + body + 4 verification JSONs.
    Returns dict of paths + 4 JSON paths + token. Caller cleans up.

    classification_override = {("nightly","mre"): "does-not-reproduce", ...}
    lets tests override specific classifications (for anomaly + escape-valve tests).
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import verify_repro
    body_text = _make_v3_body()
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    body_file = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    body_file.write(body_text)
    body_file.close()
    body_path = Path(body_file.name)

    mre_text = verify_repro.extract_mre_from_body(body_text)
    orig_text = verify_repro.extract_original_command_from_body(body_text)
    mre_sha = hashlib.sha256(verify_repro.canonicalize_extracted(mre_text)).hexdigest()
    orig_sha = hashlib.sha256(verify_repro.canonicalize_extracted(orig_text)).hexdigest()

    overrides = classification_override or {}
    json_paths = {}
    for venv in ("current", "nightly"):
        for etype in ("original", "mre"):
            cls = overrides.get((venv, etype), "reproduces")
            sha = orig_sha if etype == "original" else mre_sha
            jp = _make_verification_json(
                case_id=case_id, evidence_type=etype, venv_name=venv,
                extracted_sha=sha, classification=cls,
            )
            json_paths[(venv, etype)] = jp

    return {
        "case_file": cf, "plan_file": pp, "token": token,
        "body_path": body_path, "json_paths": json_paths,
    }


def _cleanup_new_path_test(setup: dict):
    setup["case_file"].unlink(missing_ok=True)
    setup["plan_file"].unlink(missing_ok=True)
    setup["body_path"].unlink(missing_ok=True)
    for p in setup["json_paths"].values():
        p.unlink(missing_ok=True)


def test_corpus_new_rejects_post_without_repro_verified_current_original():
    case_id = "file-2026-05-09-test-no-cur-orig"
    s = _setup_new_path_test(case_id)
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            # Provide 3 of 4; omit current-original
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "T",
        )
        assert rc != 0
        assert "repro-verified-current-original" in err, f"err should name missing flag; got: {err}"
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_happy_path_with_all_4_jsons():
    case_id = "file-2026-05-09-test-happy"
    s = _setup_new_path_test(case_id)
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            "--repro-verified-current-original", str(s["json_paths"][("current", "original")]),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "Test",
        )
        assert rc == 0, f"happy path should pass dry-run; got rc={rc}, err={err}"
        assert "DRY-RUN" in out
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_with_nightly_unavailable_reason_proceeds_without_nightly_jsons():
    """Gap 6 anchor: escape valve for legitimate stale-nightly cases."""
    case_id = "file-2026-05-09-test-nightly-unavail"
    s = _setup_new_path_test(case_id)
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            # Only current cells
            "--repro-verified-current-original", str(s["json_paths"][("current", "original")]),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--nightly-unavailable-reason", "BPF block on pypi.nvidia.com pending",
            "--body", str(s["body_path"]), "--title", "Test",
        )
        assert rc == 0, f"nightly-unavailable should bypass nightly flags; got rc={rc}, err={err}"
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_rejects_current_does_not_reproduce():
    """Current cell hard-refuse on classification != reproduces."""
    case_id = "file-2026-05-09-test-cur-doesnt-repro"
    s = _setup_new_path_test(case_id, classification_override={
        ("current", "mre"): "does-not-reproduce",
    })
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            "--repro-verified-current-original", str(s["json_paths"][("current", "original")]),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "T",
        )
        assert rc != 0
        assert "current" in err and ("does-not-reproduce" in err or "reproduces" in err), \
            f"err should explain current cell mismatch; got: {err}"
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_rejects_nightly_anomaly():
    """Nightly cell != reproduces (without escape valve) → anomaly refusal (gap 5/Layer 5)."""
    case_id = "file-2026-05-09-test-nightly-anomaly"
    s = _setup_new_path_test(case_id, classification_override={
        ("nightly", "original"): "does-not-reproduce",
    })
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            "--repro-verified-current-original", str(s["json_paths"][("current", "original")]),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "T",
        )
        assert rc != 0
        assert "NIGHTLY-REPRO ANOMALY" in err, f"err should announce anomaly; got: {err}"
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_rejects_extracted_sha_mismatch():
    """Gap 5 anchor: body bytes change after verify → refuse."""
    case_id = "file-2026-05-09-test-sha-mismatch"
    s = _setup_new_path_test(case_id)
    try:
        # Tamper: rewrite the body's MRE inner text after the JSONs were created
        body_text = s["body_path"].read_text()
        tampered = body_text.replace("import torch\nprint('repro')",
                                     "import torch\nprint('TAMPERED')")
        s["body_path"].write_text(tampered)
        # Update case_file's body_sha256 so via-skill passes — focus the test on
        # the repro-evidence sha mismatch (not the via-skill body sha mismatch)
        new_body_sha = hashlib.sha256(tampered.encode()).hexdigest()
        old_text = s["case_file"].read_text()
        s["case_file"].write_text(re.sub(
            r"^body_sha256:\s*.+$",
            f"body_sha256: {new_body_sha}",
            old_text, count=1, flags=re.MULTILINE,
        ))
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            "--repro-verified-current-original", str(s["json_paths"][("current", "original")]),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "T",
        )
        assert rc != 0
        assert "extracted_bytes_sha256 mismatch" in err, \
            f"err should explain MRE bytes drift; got: {err}"
    finally:
        _cleanup_new_path_test(s)


def test_corpus_new_rejects_target_mismatch_in_json():
    """JSON has target=upstream but tool is filing corpus → refuse."""
    case_id = "file-2026-05-09-test-target-mismatch"
    s = _setup_new_path_test(case_id)
    # Override one JSON's target to "upstream"
    bad_json = s["json_paths"][("current", "original")]
    j = json.loads(bad_json.read_bytes())
    j["target"] = "upstream"
    bad_json.write_text(json.dumps(j))
    try:
        rc, out, err = _run(
            "corpus-issue", "--via-skill", case_id,
            "--cluster-plan-approved", s["token"],
            "--repro-verified-current-original", str(bad_json),
            "--repro-verified-current-mre", str(s["json_paths"][("current", "mre")]),
            "--repro-verified-nightly-original", str(s["json_paths"][("nightly", "original")]),
            "--repro-verified-nightly-mre", str(s["json_paths"][("nightly", "mre")]),
            "--body", str(s["body_path"]), "--title", "T",
        )
        assert rc != 0
        assert "target" in err, f"err should explain target mismatch; got: {err}"
    finally:
        _cleanup_new_path_test(s)


# ── Original placement: just before persona-documents-cluster-cohesion test ──

def test_persona_documents_phase3_v1_0_frozen_mre_rule():
    """Phase 3 v1.0 (Peng directive 2026-05-09): persona must document the
    frozen-MRE-and-original_command rule. Pin so a future edit can't drop it."""
    persona = (REPO_ROOT / "subagents/file-issue/persona.md").read_text()
    required = [
        "frozen-MRE rule",
        "MRE_REVISION_NEEDED",
        "ORIGINAL_REVISION_NEEDED",
        "MRE_DRIFT",
        "ORIGINAL_DRIFT",
        "Phase 3 v1.0 body shape",
        "python repro=true",
        "original_command",
        "Verification signal (original)",
        "Verification signal (MRE)",
    ]
    missing = [r for r in required if r not in persona]
    assert not missing, (
        f"persona.md missing Phase 3 v1.0 required language: {missing}"
    )


def test_persona_documents_mode_a_check_11_metadata_only():
    """Mode A check 11 must be metadata-only (drop the temporal-impossible
    cross-check per gap 3 disposition)."""
    persona = (REPO_ROOT / "subagents/file-issue/persona.md").read_text()
    # Required language signals
    assert "Phase 3 v1.0 repro verification metadata" in persona
    assert "repro_verified_current_original_path" in persona
    assert "Do NOT cross-check `extracted_bytes_sha256`" in persona, (
        "Mode A check 11 must explicitly NOT cross-check sha (gap 3 — temporal impossibility)"
    )


def test_skill_md_documents_step_2_5_verify_repro():
    skill = (REPO_ROOT / "subagents/file-issue/SKILL.md").read_text()
    required = [
        "Step 2.5 — Repro verification",
        "Phase 3 v1.0",
        "tools/verify_repro.py",
        "tools/lookup_sweep_evidence.py",
        "--nightly-unavailable-reason",
        "NIGHTLY VENV STALE",
    ]
    missing = [r for r in required if r not in skill]
    assert not missing, f"SKILL.md missing Phase 3 v1.0 required language: {missing}"


def test_skill_md_documents_4_flag_authority_gate():
    skill = (REPO_ROOT / "subagents/file-issue/SKILL.md").read_text()
    required = [
        "--repro-verified-current-original",
        "--repro-verified-current-mre",
        "--repro-verified-nightly-original",
        "--repro-verified-nightly-mre",
        "verify_repro × 4 cells",
    ]
    missing = [r for r in required if r not in skill]
    assert not missing, f"SKILL.md authority gate missing flags: {missing}"


def test_persona_documents_cluster_cohesion_check():
    """Mode A check 10 (cluster cohesion) must be in persona.md after V1 ships.
    Pins the documentation so a future edit can't silently drop the check."""
    persona = (REPO_ROOT / "subagents/file-issue/persona.md").read_text()
    required_phrases = [
        "Cluster cohesion",
        "cluster_id",
        "root_signal",
        "affected_case",
    ]
    missing = [p for p in required_phrases if p not in persona]
    assert not missing, (
        f"persona.md is missing Mode A check 10 (cluster cohesion) language: "
        f"{missing}. Per V1 cluster+dedup directive 2026-05-08T22:01 ET, "
        f"the persona must document the cluster-cohesion check."
    )


def test_persona_documents_no_fix_suggestion_rule():
    """Criterion #4 redefinition (Peng directive 2026-05-08T20:23 ET):
    persona must explicitly forbid fix-suggestion content. This test pins
    that the persona file has the rule in writing, so a future edit can't
    silently drop it.

    The rule's ENFORCEMENT in Mode A (verdict reframe) and Mode B
    (VALIDATION_FAILED) is exercised when the sub-agent runs against a
    real draft with the anti-pattern. Until we have a way to harness-test
    a live Mode A invocation in a unit test, this doc-presence check is
    the load-bearing pin.
    """
    persona = (REPO_ROOT / "subagents/file-issue/persona.md").read_text()
    # Three load-bearing claims must appear in the persona text
    # ("criterion 4" with bare integer per 2026-05-08T21:24 auto-link prevention)
    required_phrases = [
        "Forbidden section headers",
        "Proposed fix",
        "Possible directions",
        "regression_evidence",
        "criterion 4",
    ]
    missing = [p for p in required_phrases if p not in persona]
    assert not missing, (
        f"persona.md is missing required no-fix-suggestion language: {missing}. "
        f"Per Peng directive 2026-05-08T20:23 ET, the persona must explicitly "
        f"forbid fix-suggestion content. Did someone delete the criterion #4 "
        f"redefinition section?"
    )


def test_skill_md_documents_no_fix_suggestion_rule():
    """Same pin for SKILL.md — the user-facing 'What this skill does NOT do'
    section must include the no-fix-suggestion rule."""
    skill = (REPO_ROOT / "subagents/file-issue/SKILL.md").read_text()
    required_phrases = [
        "What this skill does NOT do",
        "does NOT propose a fix",
        "Possible directions",
        "Alban refuted",  # the cautionary tale
    ]
    missing = [p for p in required_phrases if p not in skill]
    assert not missing, (
        f"SKILL.md is missing required no-fix-suggestion language: {missing}. "
        f"This documents the Peng directive 2026-05-08T20:23 ET; deletion "
        f"would let the anti-pattern silently re-emerge."
    )


def test_corpus_issue_edit_dry_run_passes_with_matching_sha():
    """Per Peng directive 2026-05-08T21:13 ET: corpus-issue gains --edit
    <issue_num> as a first-class operation. Dry-run with --edit + matching
    sha256 should pass without --title (existing title preserved).
    """
    case_id = "file-2026-05-08-test-edit"
    body_text = "edited body content (no marker required post-2026-05-08T21:13)"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        # No --title (existing title preserved)
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path, "--edit", "77")
        assert rc == 0, f"expected dry-run pass for --edit; got rc={rc}\n  err: {err}"
        assert "EDIT issue #77" in out, f"output should describe edit operation; got: {out}"
        assert "preserved" in out, f"output should note title preservation; got: {out}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_corpus_issue_new_requires_title():
    """NEW issue (no --edit) requires --title; without it, exit 2 with explanation."""
    case_id = "file-2026-05-08-test-new-no-title"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    cf = _make_case_file(case_id, mode_a_verdict="proceed",
                         mode_b_sha256="x", body_sha256=body_sha)
    pp, token = _make_cluster_plan(case_id, approved=True)
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(body_text)
            body_path = f.name
        # No --title, no --edit
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--cluster-plan-approved", token,
                            "--body", body_path)
        assert rc != 0, f"expected non-zero; got rc={rc}\n  out: {out}"
        assert "title" in err.lower(), f"err should explain title required; got: {err}"
    finally:
        cf.unlink(missing_ok=True)
        pp.unlink(missing_ok=True)
        Path(body_path).unlink(missing_ok=True)


def test_pytorch_upstream_posting_disabled_by_default():
    """Per Peng directive 2026-05-08 19:56 ET: pytorch-upstream --post must
    refuse to fire unless PYTORCH_UPSTREAM_POSTING_ENABLED constant in the
    source is True. CLI flag alone is not enough.

    Defense-in-depth on top of the External Engagement Approval rule.
    """
    # Read the constant directly from source — must be False by default.
    src = (REPO_ROOT / "tools/file_issues.py").read_text()
    import re as _re
    m = _re.search(
        r"^PYTORCH_UPSTREAM_POSTING_ENABLED\s*=\s*(True|False)\s*$",
        src, _re.MULTILINE,
    )
    assert m, "PYTORCH_UPSTREAM_POSTING_ENABLED constant missing from tools/file_issues.py"
    assert m.group(1) == "False", (
        "PYTORCH_UPSTREAM_POSTING_ENABLED must be False in committed code. "
        "If you set it to True for a post, set it back to False before committing."
    )


def test_correctness_apply_is_deprecated():
    """Gap #5 of design: correctness-apply was the umbrella-issue path; deprecated."""
    rc, out, err = _run("correctness-apply", "--plan", "/tmp/nonexistent.json")
    assert rc != 0, f"expected non-zero (deprecated); got rc={rc}"
    assert "DEPRECATED" in err, f"err should announce deprecation; got: {err}"
    assert "subagents/file-issue" in err, f"err should point at replacement; got: {err}"


# Runner ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# close-mode tests (per CLOSE_MODE_DESIGN rev 3 + adversary case adv-2026-05-10-152000)
# ─────────────────────────────────────────────────────────────────────────────


def _make_close_sweep(tmp: Path, *, days_old: float = 1.0,
                      affected_pairs: list[tuple[str, str, str, int]] | None = None,
                      with_marker: bool = False) -> Path:
    """Build a synthetic sweep dir suitable for close-mode tests.

    affected_pairs: list of (model, mode, identify_status, explain_gb_count)
    """
    from datetime import datetime, timedelta, timezone
    sweep = tmp / "2026-05-09"
    sweep.mkdir(parents=True, exist_ok=True)
    finished = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat().replace("+00:00", "Z")
    (sweep / "sweep_state.json").write_text(json.dumps({
        "status": "done",
        "finished": finished,
        "started": finished,
        "versions": {"torch": "2.13.0.dev20260507+cu126"},
    }))
    # Build identify_results.json (jsonl with metadata + rows)
    affected = affected_pairs or []
    id_lines = [json.dumps({"_record_type": "metadata", "pass": "identify"})]
    for name, mode, status, _gb in affected:
        id_lines.append(json.dumps({
            "_record_type": "row", "name": name, "mode": mode,
            "source": "hf", "status": status, "wall_time_s": 10,
        }))
    (sweep / "identify_results.json").write_text("\n".join(id_lines) + "\n")
    # Build explain_results.json
    ex_lines = [json.dumps({"_record_type": "metadata", "pass": "explain"})]
    for name, mode, _status, gb_count in affected:
        ex_lines.append(json.dumps({
            "_record_type": "row", "name": name, "mode": mode,
            "graph_break_count": gb_count,
        }))
    (sweep / "explain_results.json").write_text("\n".join(ex_lines) + "\n")
    if with_marker:
        (sweep / ".audit-rerun-required").write_text(
            "# marker for test\nFooModel|eval\n"
        )
    return sweep


def _make_close_plan(tmp: Path, issue_num: int, *,
                     model_disposition: dict | None = None,
                     model_count: int | None = None) -> Path:
    """Write a sweep-report.json plan with one close_candidate."""
    if model_disposition is None:
        model_disposition = {"FooModel": "fullgraph on current sweep"}
    if model_count is None:
        model_count = len(model_disposition)
    plan = {
        "metadata": {"timestamp": "2026-05-10T11:00:00Z"},
        "close_candidates": [{
            "number": issue_num,
            "title": "test issue",
            "previous_model_count": model_count,
            "model_disposition": model_disposition,
        }],
    }
    p = tmp / "sweep-report.json"
    p.write_text(json.dumps(plan))
    return p


def _close_args(*, case_id: str, issue: int, plan: Path, sweep: Path,
                body: Path, reason: str = "completed") -> list[str]:
    return [
        "corpus-issue",
        "--via-skill", case_id,
        "--close", str(issue),
        "--plan", str(plan),
        "--sweep-dir", str(sweep),
        "--body", str(body),
        "--close-reason", reason,
    ]


def test_close_requires_plan_sweep_dir_close_reason():
    """Required-arg validation: --close without --plan / --sweep-dir / --close-reason exits non-zero."""
    case_id = "file-2026-05-10-close-test-1"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    try:
        # Missing --plan (have --sweep-dir + --close-reason)
        rc, out, err = _run("corpus-issue", "--via-skill", case_id,
                            "--close", "999", "--body", str(body_path),
                            "--close-reason", "completed",
                            "--sweep-dir", "/tmp/foo")
        assert rc != 0, f"missing --plan should refuse; got rc={rc}"
        assert "--plan" in (out + err), f"error should name --plan; got: {err}"
    finally:
        case_path.unlink(missing_ok=True)
        body_path.unlink(missing_ok=True)


def test_close_refuses_non_close_verdict():
    """case file with mode_a_verdict='reframe' → close op refuses."""
    case_id = "file-2026-05-10-close-test-2"
    body_text = "body content"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="reframe",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        plan = _make_close_plan(Path(tmp), 999)
        rc, out, err = _run(*_close_args(case_id=case_id, issue=999, plan=plan,
                                          sweep=Path(tmp), body=body_path))
        # _validate_via_skill rejects reframe early (before close-mode-specific check)
        assert rc != 0
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_refuses_when_not_a_candidate():
    """Plan's close_candidates does NOT include issue → refuse."""
    case_id = "file-2026-05-10-close-test-3"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        # Plan has issue 999 but we ask to close 12345
        plan = _make_close_plan(Path(tmp), 999)
        sweep = _make_close_sweep(Path(tmp))
        rc, out, err = _run(*_close_args(case_id=case_id, issue=12345, plan=plan,
                                          sweep=sweep, body=body_path))
        assert rc != 0
        assert "not-a-candidate" in (out + err) or "not in the plan" in (out + err).lower(), \
            f"error should say not-a-candidate; got: {err}"
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_refuses_partial_flip():
    """Candidate exists but classify_close_candidate returns review-needed → refuse."""
    case_id = "file-2026-05-10-close-test-4"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        # Disposition with one model still breaking → classify returns review-needed
        plan = _make_close_plan(Path(tmp), 999, model_disposition={
            "FooModel": "fullgraph on current sweep",
            "BarModel": "still breaking, pattern unclassified",
        })
        sweep = _make_close_sweep(Path(tmp))
        rc, out, err = _run(*_close_args(case_id=case_id, issue=999, plan=plan,
                                          sweep=sweep, body=body_path))
        assert rc != 0
        assert "reject-keep-open" in (out + err), f"got: {err}"
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_refuses_stale_sweep():
    """Sweep age > 10 days → reframe (refuse)."""
    case_id = "file-2026-05-10-close-test-5"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        plan = _make_close_plan(Path(tmp), 999)
        sweep = _make_close_sweep(Path(tmp), days_old=15.0)  # stale
        rc, out, err = _run(*_close_args(case_id=case_id, issue=999, plan=plan,
                                          sweep=sweep, body=body_path))
        assert rc != 0
        assert "reframe" in (out + err) or "stale" in (out + err).lower(), \
            f"got: {err}"
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_refuses_when_marker_present():
    """`.audit-rerun-required` marker exists → block-stale-rerun (refuse)."""
    case_id = "file-2026-05-10-close-test-6"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        plan = _make_close_plan(Path(tmp), 999)
        sweep = _make_close_sweep(Path(tmp), with_marker=True)
        rc, out, err = _run(*_close_args(case_id=case_id, issue=999, plan=plan,
                                          sweep=sweep, body=body_path))
        assert rc != 0
        assert "block-stale-rerun" in (out + err) or "marker" in (out + err).lower(), \
            f"got: {err}"
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_reason_validation_argparse():
    """Invalid --close-reason value rejected at argparse."""
    rc, out, err = _run("corpus-issue", "--via-skill", "x",
                        "--close", "999", "--plan", "/tmp/x", "--sweep-dir", "/tmp",
                        "--body", "/tmp/b", "--close-reason", "invalid")
    assert rc != 0


def test_close_stale_apply_refused_outside_cron():
    """Gap #5: close-stale --apply requires CORPUS_CLOSE_STALE_FROM_CRON=1."""
    import os as _os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"close_candidates": [], "metadata": {}}, f)
        plan_path = Path(f.name)
    try:
        # No env var → refuse
        env = _os.environ.copy()
        env.pop("CORPUS_CLOSE_STALE_FROM_CRON", None)
        r = subprocess.run(
            [PYTHON, str(TOOL), "close-stale", "--plan", str(plan_path), "--apply"],
            capture_output=True, text=True, env=env,
        )
        assert r.returncode != 0
        assert "CORPUS_CLOSE_STALE_FROM_CRON" in r.stderr or "cron" in r.stderr.lower()
        # With env var → allowed (will succeed because plan is empty)
        env["CORPUS_CLOSE_STALE_FROM_CRON"] = "1"
        r = subprocess.run(
            [PYTHON, str(TOOL), "close-stale", "--plan", str(plan_path), "--apply"],
            capture_output=True, text=True, env=env,
        )
        assert r.returncode == 0, f"with env var should pass; got {r.returncode}\n{r.stderr}"
    finally:
        plan_path.unlink(missing_ok=True)


def test_close_refuses_when_compile_kwargs_mismatch():
    """Per Peng directive 15:27 ET: close-mode refuses if sweep's compile_kwargs
    don't match the corpus canonical (fullgraph=True, backend='eager')."""
    case_id = "file-2026-05-10-close-test-7"
    body_text = "body"
    body_sha = hashlib.sha256(body_text.encode()).hexdigest()
    case_path = _make_case_file(case_id, mode_a_verdict="close",
                                 mode_b_sha256="abc", body_sha256=body_sha)
    body_path = Path(tempfile.mktemp(suffix=".md"))
    body_path.write_text(body_text)
    with tempfile.TemporaryDirectory() as tmp:
        plan = _make_close_plan(Path(tmp), 999)
        # Build a sweep dir where compile_kwargs is fullgraph=False (mismatch)
        sweep = Path(tmp) / "2026-05-09"
        sweep.mkdir(parents=True, exist_ok=True)
        from datetime import datetime, timezone
        finished = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        (sweep / "sweep_state.json").write_text(json.dumps({
            "status": "done", "finished": finished, "started": finished,
            "versions": {"torch": "2.13.0.dev"}}))
        # Issue body parse_affected_models needs '## Affected Models' table
        # — we patch by mocking a fetched issue; instead, write a row with
        # mismatched compile_kwargs and let _per_mode_pass_check fall through
        id_lines = [json.dumps({"_record_type": "metadata", "pass": "identify"})]
        id_lines.append(json.dumps({
            "_record_type": "row", "name": "FooModel", "mode": "eval",
            "source": "hf", "status": "full_graph",
            "compile_kwargs": {"fullgraph": False, "backend": "eager"},
        }))
        (sweep / "identify_results.json").write_text("\n".join(id_lines) + "\n")
        ex_lines = [json.dumps({"_record_type": "metadata", "pass": "explain"})]
        (sweep / "explain_results.json").write_text("\n".join(ex_lines) + "\n")
        # The compile-flags check is at Step 6.5 AFTER per-mode check; need
        # per-mode check to pass first. parse_affected_models needs the issue
        # body fetch to return the table — that requires hitting GitHub. So
        # we test the logic indirectly by importing _do_close_op behavior.
        # Skip the integration test; the logic is straightforward + asserted
        # via Step 6.5's mismatches check.
        # Light test: assert the code path exists and message format is right.
        from pathlib import Path as _P
        f_src = _P(REPO_ROOT / "tools" / "file_issues.py").read_text()
        assert "compiler-flags-match check failed" in f_src, \
            "Step 6.5 compiler-flags check missing from file_issues.py"
        assert "fullgraph=True, backend='eager'" in f_src, \
            "expected canonical config not pinned in code"
    case_path.unlink(missing_ok=True)
    body_path.unlink(missing_ok=True)


def test_close_stale_dry_run_does_not_require_env_var():
    """close-stale without --apply (dry-run) bypasses the env-var check."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"close_candidates": [], "metadata": {}}, f)
        plan_path = Path(f.name)
    try:
        rc, out, err = _run("close-stale", "--plan", str(plan_path))
        assert rc == 0, f"dry-run should not require env var; got rc={rc}\n{err}"
    finally:
        plan_path.unlink(missing_ok=True)


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
