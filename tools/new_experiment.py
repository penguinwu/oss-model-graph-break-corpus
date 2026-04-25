#!/usr/bin/env python3
"""Scaffold a new discovery experiment.

Usage:
    python3 tools/new_experiment.py "<descriptive slug>" [--title "<Display Title>"]

Example:
    python3 tools/new_experiment.py "skill-comparison-easy-cases" \
        --title "Skill Comparison on Easy Cases"

What it does:
  1. Computes the experiment dir name = "YYYY-MM-<slug>" (date prefix auto).
  2. Creates discovery/experiments/<dirname>/{,reports/}.
  3. Renders tools/templates/experiment_plan.md to plan.md (methodology TODOs).
  4. Renders tools/templates/umbrella_issue.md and creates the umbrella GitHub
     issue.
  5. Patches plan.md with the actual umbrella issue number.
  6. Adds the umbrella to project board #1.
  7. Updates discovery/experiments/README.md table to list the new experiment.
  8. Prints next-step instructions.

Does NOT create per-case issues — use tools/new_case_issue.py for those after
authoring case files.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._gh_proxy import add_issue_to_project, create_issue  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "discovery" / "experiments"
PLAN_TEMPLATE = REPO_ROOT / "tools" / "templates" / "experiment_plan.md"
UMBRELLA_TEMPLATE = REPO_ROOT / "tools" / "templates" / "umbrella_issue.md"
README = EXPERIMENTS_DIR / "README.md"


def _slug_to_title(slug: str) -> str:
    """Best-effort title from slug. Better: pass --title explicitly."""
    return " ".join(w.capitalize() for w in slug.split("-"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slug", help="descriptive slug (no date prefix); e.g. 'skill-vs-no-skill-easy-cases'")
    p.add_argument("--title", help="display title (default: slug -> Title Case). Set this if slug has compound words like 'cross-case'.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    today = dt.date.today()
    full_slug = f"{today.year}-{today.month:02d}-{args.slug}"
    title = args.title or _slug_to_title(args.slug)
    experiment_dir = EXPERIMENTS_DIR / full_slug

    if experiment_dir.exists():
        raise SystemExit(f"experiment dir already exists: {experiment_dir}")

    plan_text = PLAN_TEMPLATE.read_text().format(
        title=title,
        slug=full_slug,
        umbrella_issue="<TBD>",
        created=today.isoformat(),
    )
    umbrella_text = UMBRELLA_TEMPLATE.read_text().format(
        title=title,
        slug=full_slug,
    )

    if args.dry_run:
        print(f"=== DIRECTORY ===\n{experiment_dir}\n")
        print(f"=== PLAN.MD (first 600 chars) ===\n{plan_text[:600]}\n")
        print(f"=== UMBRELLA ISSUE TITLE ===\n[Umbrella] {title}\n")
        print(f"=== UMBRELLA ISSUE BODY (first 600 chars) ===\n{umbrella_text[:600]}\n")
        return

    # 1. Create dir + plan.md (with umbrella TBD).
    (experiment_dir / "reports").mkdir(parents=True)
    plan_path = experiment_dir / "plan.md"
    plan_path.write_text(plan_text)

    # 2. Create the umbrella issue.
    issue = create_issue(
        title=f"[Umbrella] {title}",
        body=umbrella_text,
    )
    item_id = add_issue_to_project(issue["node_id"])

    # 3. Patch plan.md with the real umbrella issue number.
    plan_text_v2 = plan_text.replace(
        "**Umbrella issue:** #<TBD>",
        f"**Umbrella issue:** #{issue['number']}",
    )
    plan_path.write_text(plan_text_v2)

    # 4. Update README.md table.
    readme_text = README.read_text()
    new_row = (
        f"| `{full_slug}` | active | "
        f"[plan.md]({full_slug}/plan.md) | #{issue['number']} |"
    )
    # Insert under the "active experiments" header table.
    if new_row in readme_text:
        pass  # idempotent
    else:
        # Find the active experiments table; append a row before the next blank line after the table.
        marker = "## Active experiments"
        if marker in readme_text:
            # find the blank line after the table marker
            lines = readme_text.splitlines()
            for i, line in enumerate(lines):
                if line.strip() == marker:
                    # find the table; its first blank line after marker is the insertion point
                    j = i + 1
                    while j < len(lines) and lines[j].strip() != "":
                        j += 1
                    # j points to a blank line; insert just before it
                    lines.insert(j, new_row)
                    break
            README.write_text("\n".join(lines) + "\n")
        else:
            print("WARNING: README.md missing '## Active experiments' header; skipping table update")

    print(f"created experiment dir: {experiment_dir}")
    print(f"created plan: {plan_path}")
    print(f"created umbrella issue #{issue['number']}: {issue['html_url']}")
    print(f"  board id: {item_id}")
    print(f"updated README.md: {README}")
    print()
    print("next steps:")
    print(f"  1. Edit {plan_path} — fill in TBDs (methodology, axes, what we record)")
    print(f"  2. Author per-case files in discovery/cases/")
    print(f"  3. Create per-case issues: tools/new_case_issue.py {full_slug} <case_id> <model_name>")
    print(f"  4. Commit + push")


if __name__ == "__main__":
    main()
