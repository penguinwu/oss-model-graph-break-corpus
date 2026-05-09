#!/usr/bin/env python3
"""analyze_ledger.py — V1 minimal analyzer for subagents/mre/ledger.jsonl.

Reads the ledger, prints per-strategy + per-error-class metrics, surfaces
deep-dive candidates per the persona's surfacing rule.

V1 is intentionally minimal (per Peng directive 2026-05-09 ~11:29 ET).
Augment when patterns warrant it.

Usage: python3 subagents/mre/analyze_ledger.py [path-to-ledger.jsonl]
       (defaults to subagents/mre/ledger.jsonl alongside this script)
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_LEDGER = Path(__file__).parent / "ledger.jsonl"

# Per persona surfacing rule (V1.0.1):
# Any class with >=5 attempts AND <50% success rate is flagged
# as a "deep-dive candidate." (Median-minutes signal deferred to V2.)
DEEP_DIVE_MIN_ATTEMPTS = 5
DEEP_DIVE_MAX_SUCCESS_RATE = 0.5


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists() or path.stat().st_size == 0:
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: skipping malformed row at {path}: {e}", file=sys.stderr)
    return rows


def median(values: list[int | float]) -> float:
    return statistics.median(values) if values else 0.0


def fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.0f}%)" if total else f"{n}/0"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LEDGER
    rows = load_rows(path)

    print(f"# mre ledger analysis — {path}")
    print(f"# {len(rows)} row(s)")
    if not rows:
        print("\nLedger is empty. Use the mre subagent (see SKILL.md) to populate.")
        return

    # --- Overall ---
    print("\n## Overall")
    outcomes = Counter(r["outcome"] for r in rows)
    for outcome, n in outcomes.most_common():
        print(f"  {outcome:25s} {fmt_pct(n, len(rows))}")
    minutes = [r["minutes_spent"] for r in rows if r.get("minutes_spent") is not None]
    if minutes:
        print(f"  minutes_spent             min={min(minutes)} median={median(minutes):.1f} max={max(minutes)}")

    # --- Per error class ---
    print("\n## Per error_class")
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["error_class"]].append(r)
    for ec, ec_rows in sorted(by_class.items()):
        verified = sum(1 for r in ec_rows if r["outcome"] == "verified")
        ec_minutes = [r["minutes_spent"] for r in ec_rows if r.get("minutes_spent") is not None]
        print(f"  {ec:25s} attempts={len(ec_rows)} verified={fmt_pct(verified, len(ec_rows))} median_min={median(ec_minutes):.1f}")

    # --- Per strategy ---
    print("\n## Per strategy_used")
    by_strat = defaultdict(list)
    for r in rows:
        by_strat[r.get("strategy_used") or "(none)"].append(r)
    for s, s_rows in sorted(by_strat.items()):
        verified = sum(1 for r in s_rows if r["outcome"] == "verified")
        s_minutes = [r["minutes_spent"] for r in s_rows if r.get("minutes_spent") is not None]
        print(f"  {s:5s} attempts={len(s_rows)} verified={fmt_pct(verified, len(s_rows))} median_min={median(s_minutes):.1f}")

    # --- Same-file clusters ---
    print("\n## Same-file clusters (provenance_anchor file shared across cases)")
    by_file = defaultdict(list)
    for r in rows:
        anchor = r.get("provenance_anchor") or ""
        # Strip line number to group by file
        f = anchor.rsplit(":", 1)[0] if ":" in anchor else anchor
        if f:
            by_file[f].append(r)
    clusters = [(f, rs) for f, rs in by_file.items() if len(rs) > 1]
    if clusters:
        for f, rs in sorted(clusters, key=lambda x: -len(x[1])):
            ids = ",".join(f"#{r.get('issue_num')}" for r in rs)
            print(f"  {f}  ({len(rs)} cases: {ids})")
    else:
        print("  (none — all anchors are unique files)")

    # --- Failure modes (when outcome != verified) ---
    print("\n## Failure modes (rows with outcome != verified)")
    failures = [r for r in rows if r["outcome"] != "verified"]
    if failures:
        fm_counts = Counter(r.get("failure_mode") or "(unset)" for r in failures)
        for fm, n in fm_counts.most_common():
            print(f"  {fm:30s} {n}")
    else:
        print(f"  (none — {len(rows)}/{len(rows)} verified)")

    # --- Deep-dive candidates per persona surfacing rule ---
    print("\n## Deep-dive candidates per persona surfacing rule (V1.0.1)")
    print(f"  Rule: error_class with >={DEEP_DIVE_MIN_ATTEMPTS} attempts AND <{DEEP_DIVE_MAX_SUCCESS_RATE*100:.0f}% success rate")
    candidates = []
    for ec, ec_rows in by_class.items():
        if len(ec_rows) < DEEP_DIVE_MIN_ATTEMPTS:
            continue
        verified = sum(1 for r in ec_rows if r["outcome"] == "verified")
        rate = verified / len(ec_rows)
        if rate < DEEP_DIVE_MAX_SUCCESS_RATE:
            candidates.append((ec, len(ec_rows), rate))
    if candidates:
        for ec, n, rate in candidates:
            print(f"  → {ec}  attempts={n}  success_rate={rate*100:.0f}%")
    else:
        print("  (none flagged — either insufficient sample size per class, or success rates >=50%)")


if __name__ == "__main__":
    main()
