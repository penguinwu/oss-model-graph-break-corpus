#!/usr/bin/env python3
"""Cluster sweep failures by root pattern → produce a cluster plan.

Phase 2 V1 of the file-issue skill (per Peng directive 2026-05-08T22:01 ET).
The cluster plan goes to subagents/file-issue/cluster-plans/<sweep-ref>.yaml
and is the artifact Peng approves before any per-cluster issue work.

Two modes:
- --single-manual <case_id>     emit a 1-row "single_manual: true" plan;
                                use for one-off non-sweep filings (closes the
                                dead-air window between gate-ship and
                                full-clusterer-ship)
- --from-sweep <results.jsonl>  emit a multi-cluster plan from sweep results;
                                cluster types: numeric, graph-break, fallback

Per Peng directive: NO automated dedup-action thresholds. Whenever
tools/dedup_search.py surfaces ANY candidate, the user surfaces to Peng
for human judgment. Threshold-based auto-action will be designed by
example after the first few real surfaces.

Run examples:
    python3 tools/cluster_failures.py --single-manual file-2026-05-08-foo
    python3 tools/cluster_failures.py --from-sweep \\
        experiments/results/<run>/results.jsonl --cluster-type numeric

Requires Python 3.9+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("ERROR: cluster_failures.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent
PLANS_DIR = REPO_ROOT / "subagents/file-issue/cluster-plans"

# Audio-encoder family — based on actual NGB D1 cluster (corpus-grounded, not
# invented from intuition). Add models as new sweeps surface them.
AUDIO_ENCODER_FAMILY = {
    "Wav2Vec2Model", "Wav2Vec2ConformerModel", "WavLMModel",
    "UniSpeechModel", "UniSpeechSatModel", "HubertModel",
    "Data2VecAudioModel", "SEWModel", "SpeechEncoderDecoderModel",
}
SEQ2SEQ_FAMILY = {
    "M2M100Model", "M2M100ForConditionalGeneration",
    "PLBartModel", "PLBartForConditionalGeneration",
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _arch_family(model_name: str) -> str:
    """Map a model name to a coarse architecture-family bucket.

    Conservative: only buckets we have observed in actual sweep data. Unknown
    models map to "other". The family axis is one signal among several; the
    clusterer uses it for numeric divergences only (where family + magnitude
    + mode is the empirical grouping that fell out of the NGB D1 case).
    """
    if model_name in AUDIO_ENCODER_FAMILY:
        return "audio_encoder"
    if model_name in SEQ2SEQ_FAMILY:
        return "seq2seq"
    return "other"


def _magnitude_bucket(max_diff: float) -> str:
    """Coarse magnitude bucket for numeric divergences. Three buckets only,
    chosen because the NGB D1 cases naturally fell into them."""
    if max_diff > 4.0:
        return ">4.0"
    if max_diff > 2.0:
        return "2.0-4.0"
    return "≤2.0"


def _cluster_numeric(rows: list[dict]) -> tuple[list[dict], int]:
    """Cluster numeric divergences by (family, mode).

    Returns (clusters, n_failure_rows). n_failure_rows is the count of
    rows that ARE D1 divergences (NOT total rows in the file) — fixes the
    misleading "total_failure_rows: 380" output from the first smoke test.

    Empirical decision (post-smoke-test on the 14 NGB D1 case): magnitude
    bucket was REMOVED as a primary axis because it artificially split the
    audio-encoder family (7 models with max_diff >4.0, 2 with max_diff in
    2.0-4.0) — but those 9 models share the SAME root cause (NGB feature ×
    audio encoder forward). Magnitude is now METADATA in the root_signal,
    not a grouping key.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    n_failure_rows = 0
    for r in rows:
        max_diff = r.get("numeric_max_diff")
        if max_diff is None or max_diff <= 1e-3:
            continue
        n_failure_rows += 1
        key = (_arch_family(r.get("name", "")), r.get("mode", "?"))
        groups[key].append(r)

    clusters = []
    for (family, mode), members in sorted(groups.items()):
        rep = max(members, key=lambda m: m.get("numeric_max_diff", 0))
        max_diffs = [m.get("numeric_max_diff", 0) for m in members]
        clusters.append({
            "cluster_id": f"numeric-{family}-{mode}",
            "cluster_type": "numeric_divergence",
            "root_signal": {
                "architecture_family": family,
                "mode": mode,
                # Magnitude is metadata, not a clustering key:
                "magnitude_range": f"{min(max_diffs):.3f}-{max(max_diffs):.3f}",
            },
            "affected_cases": [
                {
                    "case_id": m.get("name"),
                    "role": "primary",
                    "sweep_evidence_excerpt": f"max_diff={m.get('numeric_max_diff'):.4f}",
                }
                for m in sorted(members, key=lambda x: -x.get("numeric_max_diff", 0))
            ],
            "case_count": len(members),
            "representative_case": (
                f"{rep.get('name')} {rep.get('mode')} (max_diff "
                f"{rep.get('numeric_max_diff'):.3f})"
            ),
            "dup_candidates": [],
            "action": "pending",
        })
    return clusters, n_failure_rows


def _cluster_graph_break(rows: list[dict]) -> tuple[list[dict], int]:
    """Cluster graph breaks by (break_reason fingerprint, file:line).

    Returns (clusters, n_failure_rows). n_failure_rows = count of rows
    whose status == "graph_break" (the failure-row population for this
    cluster type).

    Fingerprint = first 80 chars of the break_reason text (after stripping
    user-specific prefixes). Conservative — the empirical layerdrop case
    showed all 89 affected models share an identical break_reason at
    `transformers/.../modeling_*.py:NNN` for the layerdrop block.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    n_failure_rows = 0
    for r in rows:
        if r.get("status") != "graph_break":
            continue
        n_failure_rows += 1
        breaks = r.get("break_reasons", []) or []
        for br in breaks:
            reason = (br.get("reason", "") or "")[:80]
            # Extract file:line if present
            m = re.search(r"(\w+/[\w/]+\.py):(\d+)", reason)
            file_line = f"{m.group(1)}:{m.group(2)}" if m else "unknown_location"
            # Fingerprint = first 80 chars hash
            fp = hashlib.sha256(reason.encode()).hexdigest()[:12]
            groups[(fp, file_line)].append({
                "row": r,
                "reason_excerpt": reason,
            })

    clusters = []
    for (fp, file_line), members in sorted(groups.items()):
        if len(members) < 2:
            continue  # singletons handled separately or via fallback
        rep = members[0]
        clusters.append({
            "cluster_id": f"graph-break-{file_line.replace('/', '-').replace(':', '-')}-{fp}",
            "cluster_type": "graph_break",
            "root_signal": {
                "break_reason_fingerprint": fp,
                "file_line": file_line,
                "reason_excerpt": rep["reason_excerpt"],
            },
            "affected_cases": [
                {
                    "case_id": m["row"].get("name"),
                    "role": "primary",
                    "sweep_evidence_excerpt": m["reason_excerpt"][:80],
                }
                for m in members
            ],
            "case_count": len(members),
            "representative_case": (
                f"{rep['row'].get('name')} {rep['row'].get('mode', '?')}"
            ),
            "dup_candidates": [],
            "action": "pending",
        })
    return clusters, n_failure_rows


def _emit_single_manual(case_id: str) -> dict:
    """Emit a 1-row single_manual cluster plan (closes the dead-air window).

    Use for one-off filings that don't come from a sweep. The plan is a
    structural placeholder — Peng's plan-approval token is implicit-via-
    existing-channels (Phase 1 corpus-issue surface still applies).
    """
    return {
        "sweep_ref": f"single-manual-{case_id}",
        "generated_at": _now_utc(),
        "clustering_method": "single_manual",
        "total_failure_rows": 1,
        "total_clustered_rows": 1,
        "multi_root_cases": [],
        "single_manual": True,
        "clusters": [{
            "cluster_id": f"single-{case_id}",
            "cluster_type": "single_manual",
            "root_signal": {"case_id": case_id},
            "affected_cases": [{"case_id": case_id, "role": "primary"}],
            "case_count": 1,
            "representative_case": case_id,
            "dup_candidates": [],
            "action": "proceed-as-new",
        }],
        "peng_approval": {
            "token": None,
            "approved_at": None,
            "approval_message_ref": None,
        },
    }


def _emit_from_sweep(results_path: Path, cluster_type: str) -> dict:
    """Emit a multi-cluster plan from a sweep results.jsonl file."""
    if not results_path.is_file():
        sys.exit(f"ERROR: sweep results file not found: {results_path}")

    rows = []
    with open(results_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("_record_type") == "metadata":
                    continue
                rows.append(r)
            except json.JSONDecodeError:
                pass

    if cluster_type == "numeric":
        clusters, n_failure_rows = _cluster_numeric(rows)
    elif cluster_type == "graph-break":
        clusters, n_failure_rows = _cluster_graph_break(rows)
    elif cluster_type == "fallback":
        # Each case becomes its own cluster — explicit warning surface
        clusters = [{
            "cluster_id": f"fallback-{i}",
            "cluster_type": "unknown_type_fallback",
            "root_signal": {"raw_row": r},
            "affected_cases": [{"case_id": r.get("name", f"row_{i}"), "role": "primary"}],
            "case_count": 1,
            "representative_case": r.get("name", f"row_{i}"),
            "dup_candidates": [],
            "action": "pending",
        } for i, r in enumerate(rows)]
        n_failure_rows = len(rows)
    else:
        sys.exit(f"ERROR: unknown --cluster-type: {cluster_type}")

    # Audit invariant: total_clustered_rows = sum of case_counts (may exceed
    # total_failure_rows if multi-root cases exist; for V1 we don't yet
    # detect multi-root, so they'll be equal in practice).
    total_clustered = sum(c["case_count"] for c in clusters)

    return {
        "sweep_ref": str(results_path.relative_to(REPO_ROOT) if results_path.is_relative_to(REPO_ROOT) else results_path),
        "generated_at": _now_utc(),
        "clustering_method": (
            f"heuristic_v1_{cluster_type}" if cluster_type != "fallback"
            else "unknown_type_fallback"
        ),
        "total_rows_in_sweep": len(rows),
        "total_failure_rows": n_failure_rows,
        "total_clustered_rows": total_clustered,
        "multi_root_cases": [],
        "single_manual": False,
        "clusters": clusters,
        "peng_approval": {
            "token": None,
            "approved_at": None,
            "approval_message_ref": None,
        },
    }


def _write_plan(plan: dict, out_path: Path | None) -> Path:
    """Write the plan as YAML-ish (the schema is small enough to manually emit)."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", plan["sweep_ref"])[:80]
        out_path = PLANS_DIR / f"{slug}.yaml"
    # Use json.dumps for simplicity (valid YAML subset for our shapes)
    out_path.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    return out_path


def _print_token(plan_path: Path) -> str:
    """Compute and print the sha256 of the plan file (= Peng's approval token)."""
    sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    return sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single-manual",
                              help="Emit a 1-row plan for a single non-sweep filing")
    p_single.add_argument("case_id",
                          help="Case ID for the single manual filing")

    p_sweep = sub.add_parser("from-sweep",
                             help="Cluster sweep results into a plan")
    p_sweep.add_argument("results_path", type=Path,
                         help="Path to sweep results.jsonl")
    p_sweep.add_argument("--cluster-type", required=True,
                         choices=["numeric", "graph-break", "fallback"],
                         help="Which clusterer to apply")

    args = parser.parse_args()

    if args.mode == "single-manual":
        plan = _emit_single_manual(args.case_id)
    elif args.mode == "from-sweep":
        plan = _emit_from_sweep(args.results_path, args.cluster_type)
    else:
        parser.print_help()
        return 2

    out_path = _write_plan(plan, None)
    token = _print_token(out_path)
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")
    print(f"  sweep_ref: {plan['sweep_ref']}")
    print(f"  clustering_method: {plan['clustering_method']}")
    print(f"  total_failure_rows: {plan['total_failure_rows']}")
    print(f"  clusters: {len(plan['clusters'])}")
    print(f"  approval token: {token}")
    print()
    if plan["clustering_method"] == "unknown_type_fallback":
        print(f"  ⚠️  WARNING: unknown_type_fallback — every row is its own cluster.")
        print(f"      Recommend authoring a cluster heuristic for this failure type")
        print(f"      before batch-filing.")
        print()
    print(f"To use the token: pass to `tools/file_issues.py corpus-issue --cluster-plan-approved {token[:12]}...`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
