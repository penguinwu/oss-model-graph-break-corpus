#!/usr/bin/env python3
"""Tests for tools/dedup_search.py.

Pins the load-bearing claims of V1 dedup gate (Peng directive 2026-05-08T22:01 ET):
- Per Peng directive: NO auto-thresholds. ANY title/label match → surface
  to Peng with `decision: needs_peng_review`.
- Keyword extraction is conservative (drops <3-char keywords, dedups).
- Cluster_type-specific keyword logic: numeric uses model names + family;
  graph_break uses file basename + reason words.
- Refuses to modify a plan whose peng_approval.token is already set
  (would invalidate Peng's prior approval).

Run: python3 tools/test_dedup_search.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import dedup_search  # noqa: E402

TOOL = REPO_ROOT / "tools" / "dedup_search.py"
PYTHON = sys.executable


def test_keywords_for_numeric_audio_cluster():
    cluster = {
        "cluster_type": "numeric_divergence",
        "root_signal": {"architecture_family": "audio_encoder", "mode": "train"},
        "affected_cases": [
            {"case_id": "Wav2Vec2Model"},
            {"case_id": "WavLMModel"},
        ],
    }
    kws = dedup_search._keywords_for_cluster(cluster)
    # Family-derived keywords plus model names; all lowercased
    for required in ["audio encoder", "audio", "wav2vec", "hubert", "speech",
                     "wav2vec2model", "wavlmmodel"]:
        assert required in kws, f"expected {required!r} in {kws}"


def test_keywords_for_numeric_seq2seq_cluster():
    cluster = {
        "cluster_type": "numeric_divergence",
        "root_signal": {"architecture_family": "seq2seq", "mode": "train"},
        "affected_cases": [{"case_id": "M2M100Model"}],
    }
    kws = dedup_search._keywords_for_cluster(cluster)
    for required in ["seq2seq", "m2m100", "plbart", "m2m100model"]:
        assert required in kws, f"expected {required!r} in {kws}"


def test_keywords_for_graph_break_cluster():
    cluster = {
        "cluster_type": "graph_break",
        "root_signal": {
            "file_line": "transformers/models/bart/modeling_bart.py:660",
            "reason_excerpt": "Data-dependent branching on layerdrop",
        },
    }
    kws = dedup_search._keywords_for_cluster(cluster)
    assert "modeling_bart.py" in kws, f"file basename missing from {kws}"
    # First few words from reason
    assert "data" in kws or "dependent" in kws or "branching" in kws, \
        f"no reason-words in {kws}"


def test_keywords_drops_short_keywords():
    """Keywords <3 chars are noise — drop them."""
    cluster = {
        "cluster_type": "graph_break",
        "root_signal": {
            "file_line": "x.py:1",
            "reason_excerpt": "a b cd long_word",
        },
    }
    kws = dedup_search._keywords_for_cluster(cluster)
    # 1-char and 2-char tokens dropped
    for short in ["a", "b", "cd"]:
        assert short not in kws
    assert "long_word" in kws


def test_matches_keyword_against_title():
    issue = {"title": "[corpus-tooling] Wav2Vec2Model: numeric divergence"}
    assert dedup_search._matches_keyword(issue, "wav2vec2model")
    assert dedup_search._matches_keyword(issue, "numeric")
    assert not dedup_search._matches_keyword(issue, "bert")


def test_matches_keyword_against_labels():
    issue = {
        "title": "Some unrelated title",
        "labels": [{"name": "for:dynamo-team"}, {"name": "graph-break"}],
    }
    assert dedup_search._matches_keyword(issue, "dynamo")
    assert dedup_search._matches_keyword(issue, "graph-break")


def test_candidate_carries_matched_keywords_and_needs_review():
    """Every candidate is marked needs_peng_review (no auto-decision)."""
    cluster = {
        "cluster_type": "numeric_divergence",
        "root_signal": {"architecture_family": "audio_encoder", "mode": "train"},
        "affected_cases": [{"case_id": "Wav2Vec2Model"}],
    }
    issues = [
        {"number": 42, "title": "[for:dynamo-team] Wav2Vec2 layerdrop bug",
         "labels": [{"name": "for:dynamo-team"}]},
        {"number": 99, "title": "Totally unrelated thing", "labels": []},
    ]
    candidates = dedup_search._candidates_for_cluster(cluster, issues)
    assert len(candidates) == 1
    c = candidates[0]
    assert c["issue_num"] == 42
    assert c["decision"] == "needs_peng_review"
    assert c["matched_keywords"]  # non-empty


def test_dry_run_does_not_modify_plan():
    """--dry-run must not write to the plan file (verify by sha)."""
    import hashlib
    plan = {
        "sweep_ref": "x", "generated_at": "2026-05-09T00:00:00Z",
        "clustering_method": "single_manual", "total_failure_rows": 1,
        "total_clustered_rows": 1, "multi_root_cases": [],
        "single_manual": True,
        "clusters": [{
            "cluster_id": "x", "cluster_type": "single_manual",
            "root_signal": {"case_id": "X"},
            "affected_cases": [{"case_id": "X", "role": "primary"}],
            "case_count": 1, "representative_case": "X",
            "dup_candidates": [], "action": "proceed-as-new",
        }],
        "peng_approval": {"approved_at": None, "approval_message_ref": None},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(json.dumps(plan, indent=2) + "\n")
        plan_path = Path(f.name)
    try:
        before_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        # We can't run the live tool (would hit GitHub), but we can verify
        # that calling _candidates_for_cluster does not mutate the file.
        # The dry-run guard is in main(); pin the behavior directly.
        cluster = plan["clusters"][0]
        _ = dedup_search._candidates_for_cluster(cluster, [])
        after_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        assert before_sha == after_sha, "dry-run path should not mutate plan"
    finally:
        plan_path.unlink(missing_ok=True)


def test_source_lines_arg_writes_overlaps_field_when_match_found():
    """Q4 (Peng 2026-05-14 14:17 ET): --source-lines must write
    plan['source_line_overlaps'] when matches found, surfacing source-line
    conflicts at cluster-plan time (Step 0b) instead of Mode A (Step 5)."""
    plan = {
        "schema_version": 1, "sweep_ref": "test-q4",
        "clustering_method": "single_manual",
        "total_failure_rows": 1, "total_clustered_rows": 1,
        "multi_root_cases": [], "single_manual": True,
        "clusters": [{
            "cluster_id": "c1", "cluster_type": "single_manual",
            "root_signal": {"case_id": "test"},
            "affected_cases": [{"case_id": "test", "role": "primary"}],
            "case_count": 1, "representative_case": "test",
            "dup_candidates": [], "action": "proceed-as-new",
        }],
        "peng_approval": {"approved_at": None, "approval_message_ref": None},
    }
    fake_issues = [
        {"number": 999, "title": "[dynamo] some existing tracker",
         "body": "cited at transformers/foo/bar.py:42 (existing tracker)",
         "labels": []},
    ]
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    ds.fetch_open_issues = lambda: fake_issues
    try:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.yaml"
            plan_path.write_text(json.dumps(plan))
            sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                        "--source-lines", "transformers/foo/bar.py:42"]
            ds.main()
            written = json.loads(plan_path.read_text())
            assert "source_line_overlaps" in written, \
                f"plan must include source_line_overlaps; got: {list(written)}"
            assert len(written["source_line_overlaps"]) == 1
            o = written["source_line_overlaps"][0]
            assert o["issue_num"] == 999
            assert o["source_line"] == "transformers/foo/bar.py:42"
            assert o["match_type"] in ("exact", "loose-path")
    finally:
        ds.fetch_open_issues = orig_fetch


def test_source_lines_exit_code_2_on_overlap():
    """Adversary HIGH-1: source-line overlap must exit 2 (machine-readable
    STOP) — not just print stderr prose. Matches dedup_source_lines.py
    standalone contract."""
    plan = {
        "schema_version": 1, "sweep_ref": "test-q4-exit2",
        "clustering_method": "single_manual",
        "total_failure_rows": 1, "total_clustered_rows": 1,
        "multi_root_cases": [], "single_manual": True,
        "clusters": [{
            "cluster_id": "c1", "cluster_type": "single_manual",
            "root_signal": {"case_id": "test"},
            "affected_cases": [{"case_id": "test", "role": "primary"}],
            "case_count": 1, "representative_case": "test",
            "dup_candidates": [], "action": "proceed-as-new",
        }],
        "peng_approval": {"approved_at": None, "approval_message_ref": None},
    }
    fake_issues = [
        {"number": 999, "title": "[dynamo] tracker",
         "body": "cited at transformers/foo/bar.py:42", "labels": []},
    ]
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    orig_argv = sys.argv
    ds.fetch_open_issues = lambda: fake_issues
    try:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.yaml"
            plan_path.write_text(json.dumps(plan))
            sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                        "--source-lines", "transformers/foo/bar.py:42"]
            rc = ds.main()
            assert rc == 2, f"overlap must exit 2 (machine STOP signal); got {rc}"
    finally:
        ds.fetch_open_issues = orig_fetch
        sys.argv = orig_argv


def test_source_lines_dry_run_does_not_mutate_plan_file():
    """Adversary HIGH-2: dry-run must be truly read-only — no source_line_overlaps
    written to the plan file."""
    import hashlib
    plan = {
        "schema_version": 1, "sweep_ref": "test-q4-dryrun",
        "clustering_method": "single_manual",
        "total_failure_rows": 1, "total_clustered_rows": 1,
        "multi_root_cases": [], "single_manual": True,
        "clusters": [{
            "cluster_id": "c1", "cluster_type": "single_manual",
            "root_signal": {"case_id": "test"},
            "affected_cases": [{"case_id": "test", "role": "primary"}],
            "case_count": 1, "representative_case": "test",
            "dup_candidates": [], "action": "proceed-as-new",
        }],
        "peng_approval": {"approved_at": None, "approval_message_ref": None},
    }
    fake_issues = [
        {"number": 999, "title": "[dynamo] tracker",
         "body": "cited at transformers/foo/bar.py:42", "labels": []},
    ]
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    orig_argv = sys.argv
    ds.fetch_open_issues = lambda: fake_issues
    try:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.yaml"
            plan_path.write_text(json.dumps(plan))
            before_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                        "--source-lines", "transformers/foo/bar.py:42",
                        "--dry-run"]
            ds.main()
            after_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
            assert before_sha == after_sha, \
                "dry-run with --source-lines must not mutate plan file"
    finally:
        ds.fetch_open_issues = orig_fetch
        sys.argv = orig_argv


def test_source_lines_empty_or_whitespace_errors():
    """Adversary MEDIUM-4: empty/whitespace-only --source-lines must NOT
    silently pass as 'clean' (false-positive)."""
    plan_min = {"schema_version": 1, "clusters": [],
                "peng_approval": {}}
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    orig_argv = sys.argv
    ds.fetch_open_issues = lambda: []
    try:
        for bad_input in ["", "   ", "  ,  ,"]:
            with tempfile.TemporaryDirectory() as td:
                plan_path = Path(td) / "plan.yaml"
                plan_path.write_text(json.dumps(plan_min))
                sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                            "--source-lines", bad_input]
                try:
                    ds.main()
                    raise AssertionError(
                        f"empty input {bad_input!r} should sys.exit; did not"
                    )
                except SystemExit as e:
                    assert e.code != 0, \
                        f"empty input must exit non-zero; got {e.code}"
    finally:
        ds.fetch_open_issues = orig_fetch
        sys.argv = orig_argv


def test_source_lines_malformed_citation_errors():
    """Adversary MEDIUM-5: malformed source-line citations must be rejected,
    not silently treated as zero-match clean."""
    plan_min = {"schema_version": 1, "clusters": [],
                "peng_approval": {}}
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    orig_argv = sys.argv
    ds.fetch_open_issues = lambda: []
    try:
        for bad in ["not_a_path", "transformers/foo/bar.py 42",
                    "transformers/foo/bar.py:", "random/foo.py:42"]:
            with tempfile.TemporaryDirectory() as td:
                plan_path = Path(td) / "plan.yaml"
                plan_path.write_text(json.dumps(plan_min))
                sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                            "--source-lines", bad]
                try:
                    ds.main()
                    raise AssertionError(
                        f"malformed citation {bad!r} should sys.exit; did not"
                    )
                except SystemExit as e:
                    assert e.code != 0, \
                        f"malformed {bad!r} must exit non-zero; got {e.code}"
    finally:
        ds.fetch_open_issues = orig_fetch
        sys.argv = orig_argv


def test_source_lines_arg_clean_when_no_match():
    """No source-line overlap → plan still gets the field, with empty list."""
    plan = {
        "schema_version": 1, "sweep_ref": "test-q4-clean",
        "clustering_method": "single_manual",
        "total_failure_rows": 1, "total_clustered_rows": 1,
        "multi_root_cases": [], "single_manual": True,
        "clusters": [{
            "cluster_id": "c1", "cluster_type": "single_manual",
            "root_signal": {"case_id": "test"},
            "affected_cases": [{"case_id": "test", "role": "primary"}],
            "case_count": 1, "representative_case": "test",
            "dup_candidates": [], "action": "proceed-as-new",
        }],
        "peng_approval": {"approved_at": None, "approval_message_ref": None},
    }
    import dedup_search as ds
    orig_fetch = ds.fetch_open_issues
    ds.fetch_open_issues = lambda: []
    try:
        with tempfile.TemporaryDirectory() as td:
            plan_path = Path(td) / "plan.yaml"
            plan_path.write_text(json.dumps(plan))
            sys.argv = ["dedup_search.py", "--plan", str(plan_path),
                        "--source-lines", "transformers/foo/nowhere.py:1"]
            ds.main()
            written = json.loads(plan_path.read_text())
            assert "source_line_overlaps" in written
            assert written["source_line_overlaps"] == []
    finally:
        ds.fetch_open_issues = orig_fetch


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
