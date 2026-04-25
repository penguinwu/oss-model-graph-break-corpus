#!/usr/bin/env python3
"""Create a new per-case issue inside an existing experiment.

Usage:
    python3 tools/new_case_issue.py <experiment-slug> <case-id> "<Model display name>"

Example:
    python3 tools/new_case_issue.py 2026-04-cross-case-skill-discovery \
        vits_model_train "VitsModel (train mode)"

What it does:
  1. Validates the experiment slug exists under discovery/experiments/.
  2. Reads the experiment's plan.md to look up the umbrella issue number.
  3. Renders tools/templates/case_issue.md with the case-specific values.
  4. Creates a GitHub issue with the rendered body + title shape
     "[<Experiment Title>] <Model name> case".
  5. Adds the issue to project board #1.
  6. Prints the issue URL + node id.

Does NOT mutate the umbrella issue. Caller is responsible for running
`tools/update_umbrella_queue.py` (or hand-editing) afterward.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

# Make repo tools importable when called as script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._gh_proxy import add_issue_to_project, create_issue  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "discovery" / "experiments"
TEMPLATE = REPO_ROOT / "tools" / "templates" / "case_issue.md"


def _read_plan_metadata(experiment_dir: Path) -> dict[str, str]:
    plan_path = experiment_dir / "plan.md"
    if not plan_path.exists():
        raise SystemExit(f"missing plan.md in {experiment_dir}")
    text = plan_path.read_text()
    meta: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"\*\*([A-Za-z _]+):\*\*\s*(.+?)\s*$", line)
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            meta[key] = m.group(2).strip()
    for required in ("umbrella_issue", "slug", "title"):
        if required not in meta:
            raise SystemExit(
                f"plan.md is missing required `**{required.replace('_', ' ').title()}:** ...` line"
            )
    return meta


def _extract_canonical_checklist(experiment_dir: Path) -> str:
    """Pull the `## Pre-launch checklist (canonical ...)` section from plan.md.

    Returns the bullet block (everything between the header and the next `## ` header
    or the explicit "If any item" footer). Returns a stub if the section is missing.
    """
    plan_text = (experiment_dir / "plan.md").read_text()
    # Find the canonical checklist header
    m = re.search(r"^## Pre-launch checklist \(canonical[^)]*\)\s*$(.+?)(?=^## |\Z)", plan_text, re.MULTILINE | re.DOTALL)
    if not m:
        return "(no canonical checklist found in master plan — please add `## Pre-launch checklist (canonical ...)` section to plan.md)"
    block = m.group(1).strip()
    # Drop the leading prose paragraph (everything before the first `- [ ]`)
    bullets_start = block.find("- [ ]")
    if bullets_start == -1:
        return block
    # Take from first bullet through the "If any item" line if present, else through end
    body = block[bullets_start:]
    return body.strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("experiment_slug", help="e.g. 2026-04-cross-case-skill-discovery")
    p.add_argument("case_id", help="case identifier, e.g. vits_model_train")
    p.add_argument("model_name", help="display name, e.g. 'VitsModel (train mode)'")
    p.add_argument("--dry-run", action="store_true", help="print body, do not create issue")
    args = p.parse_args()

    experiment_dir = EXPERIMENTS_DIR / args.experiment_slug
    if not experiment_dir.is_dir():
        raise SystemExit(f"experiment dir not found: {experiment_dir}")
    meta = _read_plan_metadata(experiment_dir)
    umbrella_issue = meta["umbrella_issue"].lstrip("#")
    experiment_title = meta["title"]

    canonical_checklist = _extract_canonical_checklist(experiment_dir)
    snapshot_date = dt.date.today().isoformat()

    body = TEMPLATE.read_text().format(
        experiment_title=experiment_title,
        experiment_slug=args.experiment_slug,
        umbrella_issue=umbrella_issue,
        case_id=args.case_id,
        model_name=args.model_name,
        canonical_checklist=canonical_checklist,
        snapshot_date=snapshot_date,
    )
    title = f"[{experiment_title}] {args.model_name} case"

    if args.dry_run:
        print(f"=== TITLE ===\n{title}\n")
        print(f"=== BODY ===\n{body}")
        return

    issue = create_issue(title=title, body=body)
    item_id = add_issue_to_project(issue["node_id"])
    print(f"created #{issue['number']}: {issue['html_url']}")
    print(f"  title:    {title}")
    print(f"  node_id:  {issue['node_id']}")
    print(f"  board id: {item_id}")
    print(f"  next:     update umbrella #{umbrella_issue} case queue table to link this issue")


if __name__ == "__main__":
    main()
