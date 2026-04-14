#!/usr/bin/env python3
"""Daily Summary — gather mechanical activity data for the cron job.

Collects commits, GitHub issues, and repo stats since the last summary.
Outputs structured data that Claude uses alongside conversation logs and
memory to compose a rich daily summary for the feedback space.

The cron job prompt handles the narrative (design work, decisions, blockers).
This script handles the mechanical data that's tedious to gather by hand.

Usage:
    python3 tools/daily_summary.py              # JSON output for cron job
    python3 tools/daily_summary.py --since 72h  # Override lookback window
    python3 tools/daily_summary.py --post       # Compose & post (legacy)
    python3 tools/daily_summary.py --dry-run    # Preview post mode
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_SPACE = "spaces/AAQABmB_3Is"
DB_PATH = os.path.expanduser("~/.myclaw/spaces/AAQANraxXE4/myclaw.db")
PROXY = "http://localhost:7824/fetch"
REPO_SLUG = "penguinwu/oss-model-graph-break-corpus"
API_BASE = f"https://api.github.com/repos/{REPO_SLUG}"
GH_CONFIG = os.path.expanduser("~/.config/gh/hosts.yml")

STATE_KEY = "daily_summary_last_post"


def db_get(key):
    """Read a value from the MyClaw state DB."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def db_set(key, value):
    """Write a value to the MyClaw state DB."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)",
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_gh_token():
    """Read GitHub token from gh CLI config."""
    try:
        with open(GH_CONFIG) as f:
            for line in f:
                if "oauth_token:" in line:
                    return line.split("oauth_token:")[1].strip()
    except FileNotFoundError:
        pass
    return None


def proxy_get(url, token, max_size=150000):
    """GET a URL via the web proxy. Returns parsed JSON or None."""
    import urllib.request
    payload = {
        "url": url,
        "max_size": max_size,
        "headers": {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
    }
    req = urllib.request.Request(
        PROXY,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=15).read())
        return json.loads(resp.get("content", "null"))
    except Exception as e:
        print(f"  proxy_get failed for {url}: {e}", file=sys.stderr)
        return None


def get_since_timestamp(args):
    """Determine the lookback window.

    Priority: --since flag > state DB > day-of-week heuristic.
    Monday looks back 48h (Sat morning) — Friday's standup covers Thu-Fri.
    """
    now = datetime.now(timezone.utc)

    if args.since:
        # Parse duration like "72h" or "48h"
        val = args.since.rstrip("hm")
        unit = args.since[-1]
        delta = timedelta(hours=int(val)) if unit == "h" else timedelta(minutes=int(val))
        return now - delta

    # Check last summary time from DB
    last = db_get(STATE_KEY)
    if last:
        try:
            return datetime.fromtimestamp(float(last), tz=timezone.utc)
        except (ValueError, OSError):
            pass

    # Day-of-week heuristic: Monday looks back 48h (Sat morning), others 24h
    # Friday's standup already covers Thu-Fri, so Monday only needs Sat-Sun
    weekday = now.weekday()  # 0=Monday
    if weekday == 0:
        return now - timedelta(hours=48)
    else:
        return now - timedelta(hours=24)


def get_commits_since(since_dt):
    """Get git commits since a datetime."""
    since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%S")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log",
         f"--since={since_str}", "--oneline", "--no-merges"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return []
    lines = result.stdout.strip().split("\n")
    return [l for l in lines if l.strip()]


def get_unpushed_commits():
    """Count commits ahead of origin/main."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-list",
         "--count", "origin/main..HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip())
    except ValueError:
        return 0


def paginated_get(url_template, token, per_page=10, max_items=100):
    """Fetch GitHub issues in small pages to stay under proxy size limits.

    GitHub REST always returns full issue objects including body (often 5-15KB
    each with stack traces and code blocks). Fetching in small pages of 10
    keeps each response under ~100KB, avoiding proxy truncation.
    """
    all_items = []
    page = 1
    while len(all_items) < max_items:
        url = f"{url_template}&page={page}&per_page={per_page}"
        # Small per_page → small response → 100KB is plenty
        data = proxy_get(url, token, max_size=150000)
        if not data:
            break
        all_items.extend(data)
        if len(data) < per_page:
            break  # Last page
        page += 1
    return all_items[:max_items]


def _parse_rest_issues(raw_issues, state_str):
    """Extract only the fields we need from REST API issue objects."""
    result = []
    for issue in raw_issues:
        if "pull_request" in issue:
            continue
        result.append({
            "number": issue["number"],
            "title": issue["title"][:80],
            "state": state_str,
            "comments": issue["comments"],
            "labels": [l["name"] for l in issue.get("labels", [])],
            "updated_at": issue.get("updated_at", ""),
        })
    return result


def get_github_issues(token):
    """Get open issues and recently closed issues with comment counts.

    Paginates in small batches (10 per page) to avoid proxy truncation.
    GitHub REST returns full issue bodies (stack traces, code blocks) which
    makes each issue 5-15KB — at per_page=100 that's 500KB-1.5MB.
    """
    open_data = paginated_get(
        f"{API_BASE}/issues?state=open", token, per_page=10, max_items=100,
    )
    if not open_data:
        print("  WARNING: Failed to fetch open issues from GitHub", file=sys.stderr)
        return [], []

    open_issues = _parse_rest_issues(open_data, "open")

    closed_data = paginated_get(
        f"{API_BASE}/issues?state=closed&sort=updated&direction=desc",
        token, per_page=10, max_items=30,
    )
    recently_closed = _parse_rest_issues(closed_data, "closed") if closed_data else []

    return open_issues, recently_closed


def get_recent_issue_comments(token, since_dt):
    """Check which issues got new comments since the lookback window."""
    since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    data = proxy_get(
        f"{API_BASE}/issues/comments?since={since_str}&per_page=50&sort=updated",
        token,
    )
    if not data:
        return {}

    # Group by issue number, filtering out self-comments from our bot account
    comments_by_issue = {}
    for comment in data:
        # Extract issue number from issue_url
        issue_url = comment.get("issue_url", "")
        try:
            issue_num = int(issue_url.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            continue
        user = comment.get("user", {}).get("login", "unknown")
        # Skip comments by our own bot/owner account to avoid echo
        if user == "penguinwu":
            continue
        if issue_num not in comments_by_issue:
            comments_by_issue[issue_num] = []
        comments_by_issue[issue_num].append(user)

    return comments_by_issue


def compose_summary(since_dt, commits, unpushed, open_issues, recent_comments):
    """Build the daily summary message."""
    now = datetime.now(timezone.utc)

    # Derive period label from actual lookback window, not day-of-week
    gap_hours = (now - since_dt).total_seconds() / 3600
    if gap_hours > 36:
        period = "the weekend"
    else:
        period = "yesterday"

    lines = [f"[🦦 Otter]: Good morning! Here's what happened since {period}.\n"]

    # Commits
    if commits:
        lines.append(f"*Commits ({len(commits)}):*")
        for c in commits[:8]:  # Cap at 8 to keep it concise
            # Strip hash prefix for cleaner display
            msg = c.split(" ", 1)[1] if " " in c else c
            lines.append(f"  - {msg}")
        if len(commits) > 8:
            lines.append(f"  - ...and {len(commits) - 8} more")
        lines.append("")

    # GitHub Issues
    if open_issues:
        lines.append(f"*Open Issues ({len(open_issues)}):*")
        for issue in open_issues:
            labels = ""
            audience_labels = [l for l in issue["labels"] if l.startswith("for:")]
            if audience_labels:
                labels = f" [{', '.join(audience_labels)}]"
            comment_info = ""
            if issue["number"] in recent_comments:
                users = recent_comments[issue["number"]]
                comment_info = f" — new comments from {', '.join(set(users))}"
            lines.append(
                f"  - #{issue['number']}: {issue['title']}{labels}{comment_info}"
            )
        lines.append("")

    # New comments on issues
    comment_issues = [n for n in recent_comments if n not in {i["number"] for i in open_issues}]
    if comment_issues:
        lines.append("*New activity on closed issues:*")
        for n in comment_issues:
            users = recent_comments[n]
            lines.append(f"  - #{n}: comments from {', '.join(set(users))}")
        lines.append("")

    # Blockers / waiting on humans
    blockers = []
    if unpushed > 0:
        blockers.append(f"{unpushed} unpushed commit(s) — need Peng to `git push`")

    if blockers:
        lines.append("*Waiting on humans:*")
        for b in blockers:
            lines.append(f"  - {b}")
        lines.append("")

    # No activity case
    if not commits and not recent_comments:
        lines.append("Quiet period — no commits or issue activity.")
        lines.append("")

    return "\n".join(lines).strip()


def post_to_feedback_space(message):
    """Post summary to feedback space as bot."""
    result = subprocess.run(
        ["gchat", "send", FEEDBACK_SPACE, message, "--as-bot"],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Daily summary data for feedback space")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview composed summary without posting (--post mode)")
    parser.add_argument("--post", action="store_true",
                        help="Compose and post a basic summary (legacy mode)")
    parser.add_argument("--since", help="Override lookback window (e.g., 72h, 48h)")
    args = parser.parse_args()

    since_dt = get_since_timestamp(args)

    # Gather mechanical data
    commits = get_commits_since(since_dt)
    unpushed = get_unpushed_commits()

    token = get_gh_token()
    open_issues = []
    recent_comments = {}
    if token:
        open_issues, _ = get_github_issues(token)
        recent_comments = get_recent_issue_comments(token, since_dt)

    if args.post or args.dry_run:
        # Legacy mode: compose and post a basic summary
        summary = compose_summary(since_dt, commits, unpushed, open_issues, recent_comments)
        if args.dry_run:
            print("--- DRY RUN ---")
            print(summary)
            print("--- END ---")
        else:
            if post_to_feedback_space(summary):
                print("Summary posted.")
                db_set(STATE_KEY, str(datetime.now(timezone.utc).timestamp()))
            else:
                print("ERROR: Failed to post.", file=sys.stderr)
                sys.exit(1)
        return

    # Default: output structured JSON for the cron job prompt
    now = datetime.now(timezone.utc)
    data = {
        "since": since_dt.isoformat(),
        "now": now.isoformat(),
        "is_monday": now.weekday() == 0,
        "commits": commits,
        "commits_count": len(commits),
        "unpushed_count": unpushed,
        "open_issues": open_issues,
        "recent_comments": {str(k): v for k, v in recent_comments.items()},
    }
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
