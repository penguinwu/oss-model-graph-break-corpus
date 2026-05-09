#!/usr/bin/env python3
"""Search the corpus repo for open issues that may overlap with a cluster plan.

Phase 2 V1 of the file-issue skill (per Peng directive 2026-05-08T22:01 ET):
> "if there is any dedup candidate, surface to me for human judgement and we
>  will figure out a better system by examples."

So this tool DOES NOT compute overlap thresholds and DOES NOT auto-decide
the action. It surfaces ANY GitHub issue whose title or labels match a
cluster's keywords, and writes them into the plan's `dup_candidates[*]`
list with `decision: needs_peng_review`. The user (Otter) then surfaces
the per-cluster dup_candidates list to Peng before any per-cluster post.

Run examples:
    python3 tools/dedup_search.py --plan subagents/file-issue/cluster-plans/<slug>.yaml
    python3 tools/dedup_search.py --plan <plan> --dry-run    # don't modify

Important: any modification to the plan content INVALIDATES a prior approval
token (sha256 changes). This is by design — the dedup-search output
materially changes what Peng is approving. Run dedup_search BEFORE Peng
approves, never after.

Requires Python 3.9+.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("ERROR: dedup_search.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
# Reuse the proxy-API plumbing from file_issues.py (token, _proxy_api, REPO_SLUG)
from file_issues import _proxy_api, REPO_SLUG, fetch_open_issues  # noqa: E402


def _keywords_for_cluster(cluster: dict) -> list[str]:
    """Derive a small set of search keywords from a cluster's root_signal.

    V1 heuristic — corpus-grounded, conservative:
    - numeric clusters: model names in affected_cases + architecture family
      ("audio", "seq2seq") if applicable.
    - graph_break clusters: basename of file_line + first 4 words of reason
      excerpt.
    - fallback: just the case_id.
    Returns a deduplicated list of keyword strings (lowercase, trimmed).
    """
    kws: list[str] = []
    ctype = cluster.get("cluster_type", "")
    root = cluster.get("root_signal", {}) or {}
    if ctype == "numeric_divergence":
        family = root.get("architecture_family")
        if family and family != "other":
            kws.append(family.replace("_", " "))
            # Family-specific keywords (corpus-grounded):
            if family == "audio_encoder":
                kws.extend(["audio", "wav2vec", "hubert", "speech"])
            elif family == "seq2seq":
                kws.extend(["seq2seq", "m2m100", "plbart"])
        for case in cluster.get("affected_cases", []):
            cid = case.get("case_id")
            if cid:
                kws.append(cid)
    elif ctype == "graph_break":
        file_line = root.get("file_line", "")
        if file_line and file_line != "unknown_location":
            base = file_line.split("/")[-1].split(":")[0]
            kws.append(base)
        excerpt = root.get("reason_excerpt", "")
        if excerpt:
            words = re.findall(r"\b[A-Za-z][A-Za-z_]+\b", excerpt)[:4]
            kws.extend(words)
    else:
        # fallback / single_manual / unknown — use representative_case
        rep = cluster.get("representative_case", "")
        if rep:
            kws.append(rep.split()[0])  # first token

    # Dedup, lowercase, drop empties + very-short keywords (<3 chars)
    seen: set[str] = set()
    out: list[str] = []
    for k in kws:
        kk = k.strip().lower()
        if not kk or len(kk) < 3 or kk in seen:
            continue
        seen.add(kk)
        out.append(kk)
    return out


def _matches_keyword(issue: dict, kw: str) -> bool:
    """Conservative substring match against issue title + labels.

    NOT body — body content is much noisier and would surface low-quality
    matches. Title + labels is the high-signal surface.
    """
    title = (issue.get("title") or "").lower()
    if kw in title:
        return True
    for label in issue.get("labels", []) or []:
        name = (label.get("name") if isinstance(label, dict) else str(label)) or ""
        if kw in name.lower():
            return True
    return False


def _candidates_for_cluster(cluster: dict, all_issues: list[dict]) -> list[dict]:
    """Find any open issues that match ANY keyword from the cluster.

    Per Peng directive: surface ANY match. No thresholds, no
    decisions. Each candidate carries the keywords it matched
    so Peng can quickly judge relevance.
    """
    kws = _keywords_for_cluster(cluster)
    candidates: dict[int, dict] = {}  # by issue number, dedup multi-keyword hits
    for issue in all_issues:
        matched_kws = [k for k in kws if _matches_keyword(issue, k)]
        if not matched_kws:
            continue
        num = issue.get("number")
        if num is None:
            continue
        candidates[num] = {
            "issue_num": int(num),
            "title": issue.get("title", ""),
            "labels": [
                (lbl.get("name") if isinstance(lbl, dict) else str(lbl))
                for lbl in (issue.get("labels") or [])
            ],
            "matched_keywords": matched_kws,
            # Per Peng directive: no auto-decision. Otter surfaces to Peng.
            "decision": "needs_peng_review",
        }
    return sorted(candidates.values(), key=lambda c: c["issue_num"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plan", required=True, type=Path,
                        help="Path to cluster plan YAML/JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print findings; do NOT modify the plan file")
    args = parser.parse_args()

    if not args.plan.is_file():
        sys.exit(f"ERROR: plan not found: {args.plan}")

    plan = json.loads(args.plan.read_bytes())

    # Refuse to write into an already-approved plan (would invalidate the
    # token). Safer to surface explicitly than silently break trust.
    approval = (plan.get("peng_approval") or {}).get("token")
    if approval and not args.dry_run:
        sys.exit(
            f"ERROR: plan already has peng_approval.token={approval[:12]}.... "
            f"Modifying it now would invalidate that approval. Either run with "
            f"--dry-run to inspect, or clear peng_approval.token first and "
            f"re-surface to Peng after dedup search."
        )

    print(f"Fetching open issues from {REPO_SLUG}...", file=sys.stderr)
    all_issues = fetch_open_issues()
    print(f"  {len(all_issues)} open issues fetched", file=sys.stderr)

    total_candidates = 0
    for cluster in plan.get("clusters", []):
        candidates = _candidates_for_cluster(cluster, all_issues)
        cluster["dup_candidates"] = candidates
        total_candidates += len(candidates)
        cid = cluster.get("cluster_id", "<unnamed>")
        print(f"\nCluster {cid}: {len(candidates)} candidate(s)")
        for c in candidates:
            print(f"  #{c['issue_num']}: {c['title'][:70]}")
            print(f"    matched: {c['matched_keywords']}")
            print(f"    labels: {c['labels']}")

    print(f"\nTotal candidates across all clusters: {total_candidates}")
    if total_candidates > 0:
        print(
            "\nPer Peng directive 2026-05-08T22:01 ET: ANY candidate triggers "
            "human review.\nSurface the per-cluster dup_candidates list to Peng "
            "before posting."
        )

    if args.dry_run:
        print(f"\n[DRY-RUN] plan {args.plan} NOT modified.", file=sys.stderr)
        return 0

    args.plan.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    print(f"\nUpdated {args.plan}", file=sys.stderr)
    print(
        f"  NOTE: plan content changed → any prior approval token is INVALID. "
        f"Re-surface to Peng for fresh approval (new sha256).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
