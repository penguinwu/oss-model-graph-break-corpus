#!/usr/bin/env python3
"""Scaffold a new discovery experiment (lightweight by default).

Usage:
    python3 tools/new_experiment.py "<descriptive slug>" [--title "<Display Title>"]

Example:
    python3 tools/new_experiment.py "skill-comparison-easy-cases" \\
        --title "Skill Comparison on Easy Cases"

Default behavior (lightweight, local):
  1. Computes the experiment dir name = "YYYY-MM-<slug>" (date prefix auto).
  2. Creates discovery/experiments/<dirname>/{,reports/}.
  3. Renders tools/templates/experiment_plan.md to plan.md (methodology TODOs).
  4. Updates discovery/experiments/README.md table to list the new experiment.
  5. Prints next-step instructions.

Per Peng 2026-04-27: GitHub-issue creation is ORTHOGONAL to scaffolding an
experiment. Most local experiments don't need an AutoDev-visible umbrella
issue — flooding GitHub with issues nobody tracks defeats the point. Use
--with-umbrella-issue to opt in for experiments that warrant team-visible
tracking (e.g. cross-case studies that produce shipped findings, or work
that needs cross-team comments / status visibility).

When --with-umbrella-issue is set:
  6. Renders tools/templates/umbrella_issue.md and creates the umbrella
     GitHub issue.
  7. Patches plan.md with the actual umbrella issue number.
  8. Adds the umbrella to project board #1.
  9. Updates the README.md table row with the issue link.

Does NOT create per-case issues — use tools/new_case_issue.py for those
after authoring case files (also opt-in per case).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "discovery" / "experiments"
PLAN_TEMPLATE = REPO_ROOT / "tools" / "templates" / "experiment_plan.md"
UMBRELLA_TEMPLATE = REPO_ROOT / "tools" / "templates" / "umbrella_issue.md"
README = EXPERIMENTS_DIR / "README.md"


def _slug_to_title(slug: str) -> str:
    """Best-effort title from slug. Better: pass --title explicitly."""
    return " ".join(w.capitalize() for w in slug.split("-"))


def _update_readme_active_table(full_slug: str, issue_num: str) -> None:
    """Insert a row into the '## Active experiments' table.
    `issue_num` is "—" when no umbrella issue exists, else "#NN"."""
    readme_text = README.read_text()
    new_row = (
        f"| `{full_slug}` | active | "
        f"[plan.md]({full_slug}/plan.md) | {issue_num} |"
    )
    if new_row in readme_text:
        return  # idempotent
    marker = "## Active experiments"
    if marker not in readme_text:
        print("WARNING: README.md missing '## Active experiments' header; skipping table update")
        return
    lines = readme_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == marker:
            j = i + 1
            while j < len(lines) and lines[j].strip() != "":
                j += 1
            lines.insert(j, new_row)
            break
    README.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("slug", help="descriptive slug (no date prefix); e.g. 'skill-vs-no-skill-easy-cases'")
    p.add_argument("--title", help="display title (default: slug -> Title Case). Set this if slug has compound words like 'cross-case'.")
    p.add_argument("--with-umbrella-issue", action="store_true",
                   help="ALSO create a tracking GitHub issue + add to project board #1. "
                        "Default: off (most local experiments don't need it).")
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
        umbrella_issue="none (local experiment; pass --with-umbrella-issue to file one)",
        created=today.isoformat(),
    )

    if args.dry_run:
        print(f"=== DIRECTORY ===\n{experiment_dir}\n")
        print(f"=== PLAN.MD (first 600 chars) ===\n{plan_text[:600]}\n")
        if args.with_umbrella_issue:
            umbrella_text = UMBRELLA_TEMPLATE.read_text().format(title=title, slug=full_slug)
            print(f"=== UMBRELLA ISSUE TITLE ===\n[Umbrella] {title}\n")
            print(f"=== UMBRELLA ISSUE BODY (first 600 chars) ===\n{umbrella_text[:600]}\n")
        else:
            print("=== UMBRELLA ISSUE === (skipped — no --with-umbrella-issue)")
        return

    # 1. Create dir + plan.md.
    (experiment_dir / "reports").mkdir(parents=True)
    plan_path = experiment_dir / "plan.md"
    plan_path.write_text(plan_text)

    issue_num_str = "—"
    if args.with_umbrella_issue:
        # Lazy import — only pull in the GitHub proxy machinery if we're using it.
        from tools._gh_proxy import add_issue_to_project, create_issue
        umbrella_text = UMBRELLA_TEMPLATE.read_text().format(title=title, slug=full_slug)
        issue = create_issue(
            title=f"[Umbrella] {title}",
            body=umbrella_text,
        )
        item_id = add_issue_to_project(issue["node_id"])
        issue_num_str = f"#{issue['number']}"
        # Patch plan.md with the actual umbrella number
        plan_text_v2 = plan_text.replace(
            "**Umbrella issue:** none (local experiment; pass --with-umbrella-issue to file one)",
            f"**Umbrella issue:** #{issue['number']}",
        )
        plan_path.write_text(plan_text_v2)
        print(f"created umbrella issue #{issue['number']}: {issue['html_url']}")
        print(f"  board id: {item_id}")

    # Update README.md table.
    _update_readme_active_table(full_slug, issue_num_str)

    print(f"created experiment dir: {experiment_dir}")
    print(f"created plan: {plan_path}")
    print(f"updated README.md: {README}")
    if not args.with_umbrella_issue:
        print("(no umbrella issue — pass --with-umbrella-issue if you want one)")
    print()
    print("next steps:")
    print(f"  1. Edit {plan_path} — fill in TBDs (methodology, axes, what we record)")
    print(f"  2. Author per-case files in discovery/cases/")
    print(f"  3. (optional) Create per-case issues: tools/new_case_issue.py {full_slug} <case_id> <model_name>")
    print(f"  4. Commit + push")


if __name__ == "__main__":
    main()
