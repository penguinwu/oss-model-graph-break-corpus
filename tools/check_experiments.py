#!/usr/bin/env python3
"""Drift detector for the discovery experiment convention.

Walks `discovery/experiments/*/` and validates:

  * Every directory has a `plan.md`.
  * `plan.md` has the required header lines (Slug, Title, Owner, Umbrella issue, Status, Created, Last updated).
  * The slug in `plan.md` matches the directory name.
  * The directory is listed in `discovery/experiments/README.md`.
  * `reports/` subdirectory exists.

Usage:
    python3 tools/check_experiments.py                  # human summary
    python3 tools/check_experiments.py --json           # JSON for daily_brief

Returns nonzero exit if any drift is detected.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO / "discovery" / "experiments"
README = EXPERIMENTS_DIR / "README.md"

REQUIRED_HEADERS = ("slug", "title", "owner", "umbrella_issue", "status", "created", "last_updated")


def _parse_plan_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\*\*([A-Za-z _]+):\*\*\s*(.+?)\s*$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            headers[key] = m.group(2).strip()
    return headers


def collect() -> dict:
    """Return a structured drift report."""
    report = {
        "experiment_count": 0,
        "ok": [],
        "drift": [],
    }

    if not EXPERIMENTS_DIR.is_dir():
        report["drift"].append({"item": "experiments-dir-missing", "detail": str(EXPERIMENTS_DIR)})
        return report

    readme_text = README.read_text() if README.exists() else ""

    for child in sorted(EXPERIMENTS_DIR.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        report["experiment_count"] += 1
        slug = child.name
        problems: list[str] = []

        plan = child / "plan.md"
        if not plan.exists():
            problems.append("missing plan.md")
        else:
            headers = _parse_plan_headers(plan.read_text())
            for required in REQUIRED_HEADERS:
                if required not in headers:
                    problems.append(f"plan.md missing **{required.replace('_', ' ').title()}:**")
            slug_in_plan = headers.get("slug", "").strip("`")
            if slug_in_plan and slug_in_plan != slug:
                problems.append(
                    f"plan.md slug mismatch: dir={slug!r}, plan={slug_in_plan!r}"
                )

        if not (child / "reports").is_dir():
            problems.append("missing reports/ subdir")

        if slug not in readme_text:
            problems.append("not listed in README.md")

        entry = {"slug": slug, "path": str(child.relative_to(REPO))}
        if problems:
            entry["problems"] = problems
            report["drift"].append(entry)
        else:
            report["ok"].append(entry)

    return report


def render_human(report: dict) -> str:
    lines = [
        f"Experiment-convention check — {report['experiment_count']} experiments found",
        f"  ok:    {len(report['ok'])}",
        f"  drift: {len(report['drift'])}",
        "",
    ]
    if not report["drift"]:
        lines.append("All experiments conform to the convention.")
    else:
        for d in report["drift"]:
            lines.append(f"DRIFT: {d.get('slug', d.get('item'))}")
            for p in d.get("problems", []):
                lines.append(f"   - {p}")
            if "detail" in d:
                lines.append(f"   - detail: {d['detail']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    report = collect()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        sys.stdout.write(render_human(report))
    return 1 if report["drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
