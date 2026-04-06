#!/usr/bin/env python3
"""Monitor GitHub issues for new activity and alert via GChat."""

import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

PROXY = "http://localhost:7824/fetch"
REPO = "penguinwu/oss-model-graph-break-corpus"
API_BASE = f"https://api.github.com/repos/{REPO}"
DB_PATH = os.path.expanduser("~/.myclaw/spaces/AAQANraxXE4/myclaw.db")
SPACE_ID = "spaces/AAQANraxXE4"
GH_CONFIG = os.path.expanduser("~/.config/gh/hosts.yml")

# State keys
KEY_LAST_CHECK = "github_monitor_last_check"
KEY_KNOWN_ISSUES = "github_monitor_known_issues"
KEY_KNOWN_COMMENTS = "github_monitor_known_comment_counts"


def get_token():
    """Read GitHub token from gh CLI config."""
    with open(GH_CONFIG) as f:
        for line in f:
            if "oauth_token:" in line:
                return line.split("oauth_token:")[1].strip()
    return None


def proxy_get(url, token):
    """GET a URL via the web proxy."""
    payload = {
        "url": url,
        "max_size": 100000,
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
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                return json.loads(result["content"])
            else:
                print(f"API error: {result.get('error', 'unknown')}", file=sys.stderr)
                return None
    except Exception as e:
        print(f"Proxy error: {e}", file=sys.stderr)
        return None


def proxy_available():
    """Check if web proxy is running."""
    try:
        req = urllib.request.Request("http://localhost:7824/status")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode()).get("ok", False)
    except Exception:
        return False


def db_get(key):
    """Read a value from the state table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def db_set(key, value):
    """Write a value to the state table."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value)
        )
        conn.commit()
    finally:
        conn.close()


def send_gchat(message):
    """Send alert to Otter's GChat space."""
    subprocess.run(
        ["gchat", "send", SPACE_ID, message, "--as-bot"],
        capture_output=True,
        timeout=30,
    )


def main():
    # Check proxy
    if not proxy_available():
        print("Web proxy not running, skipping check")
        return

    token = get_token()
    if not token:
        print("No GitHub token found", file=sys.stderr)
        return

    # Fetch open issues
    issues = proxy_get(f"{API_BASE}/issues?state=all&sort=updated&direction=desc&per_page=50", token)
    if issues is None:
        print("Failed to fetch issues", file=sys.stderr)
        return

    # Load known state
    known_issues_raw = db_get(KEY_KNOWN_ISSUES)
    known_issues = set(json.loads(known_issues_raw)) if known_issues_raw else set()

    known_comments_raw = db_get(KEY_KNOWN_COMMENTS)
    known_comments = json.loads(known_comments_raw) if known_comments_raw else {}

    alerts = []

    for issue in issues:
        num = issue["number"]
        title = issue["title"]
        user = issue["user"]["login"]
        comments_count = issue["comments"]
        html_url = issue["html_url"]
        state = issue["state"]

        # New issue?
        if num not in known_issues:
            # Skip if this is the first run (don't alert on existing issues)
            if known_issues_raw is not None:
                alerts.append(
                    f"New issue #{num} by {user}: {title}\n{html_url}"
                )
            known_issues.add(num)

        # New comments?
        prev_count = known_comments.get(str(num), 0)
        if comments_count > prev_count:
            new_count = comments_count - prev_count
            if known_comments_raw is not None and prev_count > 0:
                # Fetch the latest comments
                comments = proxy_get(
                    f"{API_BASE}/issues/{num}/comments?per_page={new_count}&sort=created&direction=desc",
                    token,
                )
                if comments:
                    for c in comments[:3]:  # Max 3 comments per alert
                        commenter = c["user"]["login"]
                        body_preview = c["body"][:200].replace("\n", " ")
                        alerts.append(
                            f"New comment on #{num} ({title}) by {commenter}: {body_preview}\n{html_url}"
                        )
            elif known_comments_raw is not None and prev_count == 0 and comments_count > 0:
                alerts.append(
                    f"{comments_count} new comment(s) on #{num}: {title}\n{html_url}"
                )

        known_comments[str(num)] = comments_count

    # Save state
    db_set(KEY_KNOWN_ISSUES, json.dumps(sorted(known_issues)))
    db_set(KEY_KNOWN_COMMENTS, json.dumps(known_comments))
    db_set(KEY_LAST_CHECK, datetime.now(timezone.utc).isoformat())

    # Send alerts
    if alerts:
        header = f"[🦦 Otter]: GitHub activity on {REPO}:\n\n"
        message = header + "\n\n".join(alerts)
        # Truncate to GChat limit
        if len(message) > 3900:
            message = message[:3900] + "\n... (truncated)"
        send_gchat(message)
        print(f"Sent {len(alerts)} alert(s)")
    else:
        print("No new activity")


if __name__ == "__main__":
    main()
