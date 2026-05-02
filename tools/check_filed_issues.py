#!/usr/bin/env python3
"""Check status of issues we've filed across external repos.

Queries GitHub for issues authored by `penguinwu` across repos we file to,
tracks new comments since last check via a state file, and emits JSON for
the daily brief to surface in the 🚨 Needs your attention section.

Usage:
    python3 tools/check_filed_issues.py                    # JSON to stdout
    python3 tools/check_filed_issues.py --pretty           # human-readable

State file: ~/.cache/check_filed_issues_state.json (tracks last_seen_per_issue)

Required: web_proxy at localhost:7824 (per ~/.myclaw-shared/recipes/github-access.md)
GitHub token: ~/.config/gh/hosts.yml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Otter's primary project repos — track ALL issue activity (regardless of author)
PRIMARY_REPOS = [
    "penguinwu/oss-model-graph-break-corpus",
    "penguinwu/pt2-skill-discovery",
]

# External repos where Otter has filed issues — track only Otter-filed entries
# Use date cutoff = Otter's birthday (2026-03-01). Issues authored by penguinwu
# BEFORE this date are pre-Otter (TorchScript era, Peng's personal filings) and
# are someone else's responsibility.
EXTERNAL_REPOS = [
    "pytorch/pytorch",
    "huggingface/transformers",
]
OTTER_BIRTHDAY = "2026-03-01"  # ISO date — issues created before this are pre-Otter
AUTHOR = "penguinwu"

PROXY = "http://localhost:7824/fetch"
STATE_FILE = Path.home() / ".cache" / "check_filed_issues_state.json"


def _gh_token() -> str | None:
    """Read GitHub OAuth token from gh CLI config."""
    hosts_yml = Path.home() / ".config" / "gh" / "hosts.yml"
    if not hosts_yml.exists():
        return None
    for line in hosts_yml.read_text().splitlines():
        line = line.strip()
        if line.startswith("oauth_token:"):
            return line.split(":", 1)[1].strip()
    return None


def _gh_get(url: str, token: str) -> Any:
    """GET a GitHub API URL via the local web proxy."""
    import urllib.request

    payload = json.dumps({
        "url": url,
        "method": "GET",
        "headers": {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
        },
    }).encode()
    req = urllib.request.Request(
        PROXY,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        wrapper = json.loads(resp.read())
    if not wrapper.get("ok"):
        raise RuntimeError(f"proxy error for {url}: {wrapper}")
    content = wrapper.get("content", "")
    return json.loads(content) if content else None


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _fetch_issues(token: str, query: str) -> list[dict]:
    """Run a single GitHub issue-search query, return raw items."""
    url = f"https://api.github.com/search/issues?q={quote(query, safe=':+')}&per_page=100"
    try:
        return (_gh_get(url, token) or {}).get("items", [])
    except Exception as e:
        print(f"warning: query failed [{query[:80]}...]: {e}", file=sys.stderr)
        return []


def fetch_tracked_issues(token: str) -> list[dict]:
    """Return tracked issues, scoped per Otter's responsibility:
    - PRIMARY_REPOS (corpus, pt2-skill-discovery): ALL issues regardless of author
    - EXTERNAL_REPOS (pytorch/pytorch, transformers): only issues authored by Otter
      (proxied via author=penguinwu created on/after OTTER_BIRTHDAY).
    """
    issues: list[dict] = []

    # Primary repos: track every issue (open + closed in last 90d for response monitoring)
    for repo in PRIMARY_REPOS:
        for q in [
            f"is:issue+repo:{repo}+state:open+sort:updated",
            f"is:issue+repo:{repo}+state:closed+closed:>=2026-02-01+sort:updated",
        ]:
            for item in _fetch_issues(token, q):
                issues.append({
                    "repo": repo, "scope": "primary",
                    "number": item["number"], "title": item["title"],
                    "state": item["state"], "html_url": item["html_url"],
                    "updated_at": item["updated_at"],
                    "created_at": item.get("created_at", ""),
                    "comments_count": item.get("comments", 0),
                    "labels": [lbl["name"] for lbl in item.get("labels", [])],
                    "author": item.get("user", {}).get("login", ""),
                })

    # External repos: only Otter-authored (created on/after OTTER_BIRTHDAY)
    for repo in EXTERNAL_REPOS:
        q = f"is:issue+author:{AUTHOR}+repo:{repo}+created:>={OTTER_BIRTHDAY}+sort:updated"
        for item in _fetch_issues(token, q):
            issues.append({
                "repo": repo, "scope": "external",
                "number": item["number"], "title": item["title"],
                "state": item["state"], "html_url": item["html_url"],
                "updated_at": item["updated_at"],
                "created_at": item.get("created_at", ""),
                "comments_count": item.get("comments", 0),
                "labels": [lbl["name"] for lbl in item.get("labels", [])],
                "author": item.get("user", {}).get("login", ""),
            })

    # Dedupe by (repo, number) — open + closed queries can overlap
    seen = set()
    unique: list[dict] = []
    for iss in issues:
        key = (iss["repo"], iss["number"])
        if key not in seen:
            seen.add(key)
            unique.append(iss)
    return unique


def diff_against_state(issues: list[dict], state: dict) -> list[dict]:
    """Mark which issues have NEW activity since last check.

    `state` is keyed `<repo>#<number>` → {"last_updated_at": ..., "last_comments_count": ...}
    """
    flagged: list[dict] = []
    for iss in issues:
        key = f"{iss['repo']}#{iss['number']}"
        prev = state.get(key, {})
        prev_updated = prev.get("last_updated_at", "")
        prev_count = prev.get("last_comments_count", 0)

        new_activity = False
        signal = []
        if iss["updated_at"] > prev_updated:
            new_activity = True
            signal.append(f"updated {iss['updated_at'][:10]}")
        if iss["comments_count"] > prev_count:
            new_activity = True
            signal.append(f"+{iss['comments_count'] - prev_count} comments")

        flagged.append({
            **iss,
            "new_activity": new_activity,
            "signal": "; ".join(signal) if signal else "no change",
        })
    return flagged


def update_state(issues: list[dict], state: dict) -> dict:
    """After processing, persist current state for next run."""
    new_state = dict(state)
    for iss in issues:
        key = f"{iss['repo']}#{iss['number']}"
        new_state[key] = {
            "last_updated_at": iss["updated_at"],
            "last_comments_count": iss["comments_count"],
        }
    return new_state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pretty", action="store_true", help="human-readable output")
    p.add_argument("--no-update", action="store_true",
                   help="don't update state file (useful for repeated runs)")
    p.add_argument("--changes-only", action="store_true",
                   help="emit only issues with NEW activity since last check (default for daily brief)")
    args = p.parse_args()

    token = _gh_token()
    if not token:
        print("ERROR: no GitHub token at ~/.config/gh/hosts.yml", file=sys.stderr)
        return 1

    state = _load_state()
    issues = fetch_tracked_issues(token)
    flagged = diff_against_state(issues, state)

    if args.changes_only:
        flagged = [f for f in flagged if f["new_activity"]]

    # Sort: new activity first, then by updated_at desc
    flagged.sort(key=lambda x: (not x["new_activity"], x["updated_at"]), reverse=False)
    flagged.reverse()  # most recent first

    if not args.no_update:
        new_state = update_state(issues, state)
        _save_state(new_state)

    if args.pretty:
        new_count = sum(1 for f in flagged if f["new_activity"])
        primary = sum(1 for f in flagged if f.get("scope") == "primary")
        external = sum(1 for f in flagged if f.get("scope") == "external")
        print(f"Tracked issues: {len(flagged)} ({primary} primary-repo, {external} external) — {new_count} with NEW activity\n")
        for f in flagged:
            marker = "🆕" if f["new_activity"] else "  "
            scope = "📦" if f.get("scope") == "primary" else "🔗"
            print(f"  {marker} {scope} {f['repo']}#{f['number']} [{f['state']}] — {f['title'][:60]}")
            print(f"      {f['signal']} | {f['html_url']}")
    else:
        out = {
            "checked_at": int(time.time()),
            "primary_repos": PRIMARY_REPOS,
            "external_repos": EXTERNAL_REPOS,
            "otter_birthday": OTTER_BIRTHDAY,
            "total_count": len(flagged),
            "new_activity_count": sum(1 for f in flagged if f["new_activity"]),
            "issues": flagged,
        }
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
