#!/usr/bin/env python3
"""Add a Backlog card to the project board in one command.

Use this whenever you (the agent or Peng) commit to deferred work — work that
isn't being executed right now but mustn't be forgotten. The board's Backlog
column is the canonical source of "agreed but not started." TodoWrite is
in-conversation only; OPEN-LOOPS.md is project-level facts; the board is
indefinite-lifetime, visible without local access.

⚠️ DO NOT use queue_task.py if an existing per-case issue (e.g. one created by
new_case_issue.py) already covers the work. In that case, update THAT issue's
Status field on the board instead. Creating a separate "queued" card is
duplication — it splits the conversation across two issues. The lesson cost us
one duplicate (#64 superseded by #59) on 2026-04-24.

Usage:
    python3 tools/queue_task.py "<title>" [--body "<text>"] [--umbrella <issue#>] [--label <label>]

Examples:
    python3 tools/queue_task.py "Mistral3 relaunch w/ new runner + skip skill arm" \
        --body "Use --skills none. Wait for runner change to land first." \
        --umbrella 60

    python3 tools/queue_task.py "Investigate diff-promoted-post-validate flag rate" \
        --body "If most timed-out trials trigger this flag, the runner change is paying off."

The created issue auto-includes a `**Queued at:**` line so the convention
checker can flag aged Backlog items in the daily brief.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools._gh_proxy import add_issue_to_project, create_issue  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("title", help="short task title (becomes the issue title)")
    p.add_argument("--body", default="", help="optional details")
    p.add_argument("--umbrella", type=int, help="related umbrella issue number (linked in body)")
    p.add_argument("--label", action="append", default=[], help="GitHub label (can repeat)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    queued_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body_lines = [
        f"**Queued at:** {queued_at}",
    ]
    if args.umbrella:
        body_lines.append(f"**Umbrella:** #{args.umbrella}")
    body_lines.append(f"**Status:** Backlog (queued — not started)\n")
    body_lines.append("---")
    body_lines.append("")
    if args.body:
        body_lines.append(args.body)
    else:
        body_lines.append(
            "(No additional details. Add a comment when picking up so the context is "
            "captured before work starts.)"
        )

    title = f"[Queued] {args.title}" if not args.title.startswith("[") else args.title
    body = "\n".join(body_lines)

    if args.dry_run:
        print(f"=== TITLE ===\n{title}\n")
        print(f"=== BODY ===\n{body}")
        return

    issue = create_issue(title=title, body=body, labels=args.label or None)
    item_id = add_issue_to_project(issue["node_id"])
    print(f"queued #{issue['number']}: {issue['html_url']}")
    print(f"  title:    {title}")
    print(f"  board id: {item_id}")


if __name__ == "__main__":
    main()
