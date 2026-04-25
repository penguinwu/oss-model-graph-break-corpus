#!/usr/bin/env python3
"""Walk the corpus repo for plan files, parse frontmatter, surface state.

A plan file is any markdown file whose path matches `**/plan*.md` and whose
frontmatter contains `plan:` and `status:` keys.

Usage:
    python3 tools/check_plan.py                  # human-readable summary
    python3 tools/check_plan.py --json           # JSON for daily_brief.py
    python3 tools/check_plan.py --stale-days 3   # override staleness threshold
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STALE_DAYS = 3
PLAN_GLOBS = ["**/plan.md", "**/*_plan.md", "**/plan_*.md"]
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", "sweep_results", "correctness-resweep", ".trash"}


def find_plan_files() -> list[Path]:
    seen: set[Path] = set()
    for pattern in PLAN_GLOBS:
        for path in REPO.glob(pattern):
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            seen.add(path)
    return sorted(seen)


def parse_frontmatter(path: Path) -> dict | None:
    text = path.read_text()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    block = text[4:end]
    fm: dict = {}
    for line in block.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip()
    if "plan" not in fm or "status" not in fm:
        return None
    return fm


def days_since(date_str: str, today: dt.date) -> int | None:
    try:
        d = dt.date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return None
    return (today - d).days


def collect(stale_days: int, today: dt.date) -> dict:
    plans = []
    for path in find_plan_files():
        fm = parse_frontmatter(path)
        if fm is None:
            continue
        last_check = fm.get("last_check", "")
        age = days_since(last_check, today)
        plans.append({
            "path": str(path.relative_to(REPO)),
            "name": fm.get("plan", "<unnamed>"),
            "status": fm.get("status", "<unknown>"),
            "owner": fm.get("owner", ""),
            "last_check": last_check,
            "age_days": age,
            "stale": (age is not None and age > stale_days and fm.get("status", "") == "active"),
            "missing_last_check": age is None,
        })
    plans.sort(key=lambda p: (-(p["age_days"] or 0), p["path"]))
    return {
        "today": today.isoformat(),
        "stale_threshold_days": stale_days,
        "plan_count": len(plans),
        "active": sum(1 for p in plans if p["status"] == "active"),
        "stale": sum(1 for p in plans if p["stale"]),
        "missing_last_check": sum(1 for p in plans if p["missing_last_check"]),
        "plans": plans,
    }


def render_human(report: dict) -> str:
    lines = [
        f"Plan check — {report['today']}",
        f"  total plans: {report['plan_count']}  active: {report['active']}  stale (>{report['stale_threshold_days']}d): {report['stale']}",
        "",
    ]
    for p in report["plans"]:
        marker = "STALE" if p["stale"] else ("MISSING last_check" if p["missing_last_check"] else "ok")
        age = "?" if p["age_days"] is None else f"{p['age_days']}d"
        lines.append(f"  [{marker:5s}] {p['status']:7s} {age:>4s}  {p['path']}  ({p['name']})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    args = parser.parse_args()

    report = collect(args.stale_days, dt.date.today())
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        sys.stdout.write(render_human(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
