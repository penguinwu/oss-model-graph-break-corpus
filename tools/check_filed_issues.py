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

# Bot/self markers — mirror jobs/github_issue_monitor.py BOT_MARKERS.
# A comment whose body contains any of these is treated as "self-noise" and
# does NOT count as new activity (used to de-noise the daily brief, since
# Otter posts redirect/closure/cross-link comments routinely).
BOT_MARKERS = ["[🦦 Otter]", "<!-- otter-bot -->", "## tlparse Trace Reports"]

# Priority authors — comments by these accounts get tagged for Peng's attention
# even when they're embedded in higher-volume noise. Start narrow; grow as we
# identify external stakeholders by github handle.
PRIORITY_AUTHORS = {
    "anijain2305": "Animesh (dynamo team)",
}

# Mention detection — if a comment body @-mentions Peng, surface as priority
# regardless of who the commenter is.
MENTION_HANDLES = {"penguinwu"}


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


def _fetch_events_since(repo: str, number: int, prev_updated_at: str,
                        token: str) -> list[dict]:
    """Fetch issue events strictly newer than `prev_updated_at`. Returns [] on error.

    Used to detect SELF-CAUSED issue-level activity (body edits, label/state
    changes, new-issue creation) that bumps `updated_at` without adding a
    comment. The `_classify_comments` filter only catches self-comments —
    this catches the rest.

    Per Peng directive 2026-05-11: filter ALL Otter-caused activity from the
    GChat alert path (a) self-comments, (b) self body edits, (c) self issue
    creation, (d) self labels/state changes. (a) is handled by
    _classify_comments; (b)+(c)+(d) by this function.

    See `https://docs.github.com/en/rest/issues/events`. Endpoint returns
    actor + event type for each timeline event. We filter to events strictly
    after `prev_updated_at` (defensive against the boundary-comment leak that
    bit us 2026-05-06; see _fetch_new_comments docstring).
    """
    url = f"https://api.github.com/repos/{repo}/issues/{number}/events?per_page=100"
    try:
        items = _gh_get(url, token) or []
    except Exception as e:
        print(f"warning: fetching events for {repo}#{number}: {e}", file=sys.stderr)
        return []
    if prev_updated_at:
        items = [e for e in items if (e.get("created_at") or "") > prev_updated_at]
    return items


def _fetch_new_comments(repo: str, number: int, n_new: int,
                        prev_updated_at: str, token: str) -> list[dict]:
    """Fetch comments strictly newer than `prev_updated_at`. Returns [] on error.

    Bug history (2026-05-06): two interacting bugs caused self-comment leak.
    (a) The previous implementation used `?per_page=N&sort=created&direction=desc`,
        but GitHub's `/issues/N/comments` endpoint silently ignores `sort` and
        `direction` (only `since` and `per_page` are supported per the docs).
        That returned the OLDEST N comments instead of the newest.
    (b) After fixing (a), the API's `since` parameter is INCLUSIVE — it returns
        comments where updated_at >= since, including the boundary comment that
        was the last one we saw. Without a local strict-greater filter, the
        boundary (often an external comment) re-leaks as "new activity" each run.

    Result of the chain: self-replies on upstream issues we filed leaked through
    the filter (Alban's original re-counted as new activity even though we hadn't
    received any new external comments — see #182116 follow-up to commit 4ced364).

    Fix: use `since=prev_updated_at` to narrow the API response (efficient), then
    locally filter to `created_at > prev_updated_at` (strict, defensive).
    """
    if n_new <= 0:
        return []
    url = f"https://api.github.com/repos/{repo}/issues/{number}/comments?per_page=100"
    if prev_updated_at:
        url += f"&since={prev_updated_at}"
    try:
        items = _gh_get(url, token) or []
    except Exception as e:
        print(f"warning: fetching comments for {repo}#{number}: {e}", file=sys.stderr)
        return []
    if prev_updated_at:
        # Strict-greater local filter — `since` API param is inclusive.
        items = [c for c in items if (c.get("created_at") or "") > prev_updated_at]
    return items


def _classify_comments(comments: list[dict]) -> dict:
    """Classify a batch of comments into self-noise vs real activity vs priority.

    Returns:
      - non_self_count: int (comments not posted by self/bot)
      - priority_signals: list[str] (one per priority comment, prefix with handle)
    """
    non_self_count = 0
    priority_signals: list[str] = []
    for c in comments:
        commenter = (c.get("user") or {}).get("login", "")
        body = c.get("body") or ""
        # Skip self
        if commenter == AUTHOR:
            continue
        # Skip bot-marker-bearing comments (regardless of author — could be
        # a bot account with the marker, or self-comment with the marker)
        if any(m in body for m in BOT_MARKERS):
            continue
        non_self_count += 1
        # Priority signals
        why = []
        if commenter in PRIORITY_AUTHORS:
            why.append(PRIORITY_AUTHORS[commenter])
        if any(f"@{h}" in body for h in MENTION_HANDLES):
            why.append("mentions @penguinwu")
        if why:
            preview = body[:120].replace("\n", " ").strip()
            priority_signals.append(f"{commenter}: {' + '.join(why)} — \"{preview}\"")
    return {
        "non_self_count": non_self_count,
        "priority_signals": priority_signals,
    }


def diff_against_state(issues: list[dict], state: dict, token: str) -> list[dict]:
    """Mark which issues have NEW activity since last check.

    `state` is keyed `<repo>#<number>` → {"last_updated_at": ..., "last_comments_count": ...}

    Three-stage self-noise filter (Peng directive 2026-05-11 — the GChat alert
    path must not surface activity caused by Otter; sweep-report and
    weekly-brief paths intentionally still summarize self-actions because
    those have a different audience):
      Stage 1 — count + updated_at delta. Potential activity if either changed.
      Stage 2 — fetch new comments + filter self/bot. If 0 non-self comments
                remain, the comment delta is pure self-noise.
      Stage 3 — fire WHENEVER updated_advanced (body edit / label / state /
                new-issue creation), regardless of comments_increased. Without
                this gating, a self-comment co-occurring with an external rename
                in the same poll window would shadow the rename. Returns
                `non_self_issue_event: True` if an external actor caused any
                issue-level action; False if all-self.

                Empty-events heuristic: GitHub's `/issues/N/events` endpoint
                does NOT log pure-description edits. When `events == []` AND
                `updated_advanced` is true, the bump came from a body edit
                (or similar untracked action). Heuristic: if the issue's
                author is Otter, only Otter could have body-edited without
                leaving an event → suppress. If author is external, we can't
                tell → fail-open and surface (rare but possible scenario).
    """
    flagged: list[dict] = []
    for iss in issues:
        key = f"{iss['repo']}#{iss['number']}"
        prev = state.get(key, {})
        prev_updated = prev.get("last_updated_at", "")
        prev_count = prev.get("last_comments_count", 0)
        is_brand_new = (key not in state)

        # Stage 1 — coarse delta
        comments_increased = iss["comments_count"] > prev_count
        # prev_updated="" sentinel for new issues; lexicographic > works because
        # ISO-8601 strings sort chronologically and any non-empty timestamp > "".
        updated_advanced = iss["updated_at"] > prev_updated

        # Stage 2 — fetch + filter comments (only when comments increased)
        non_self_count = 0
        priority_signals: list[str] = []
        if comments_increased:
            new_comments = _fetch_new_comments(
                iss["repo"], iss["number"],
                iss["comments_count"] - prev_count,
                prev_updated,
                token,
            )
            cls = _classify_comments(new_comments)
            non_self_count = cls["non_self_count"]
            priority_signals = cls["priority_signals"]

        # Stage 3 — issue-level external-actor detection. Fires whenever
        # updated_advanced is true (NOT gated on comments_increased — that
        # gating was the Gap-1 false-suppression bug).
        non_self_issue_event = False
        if updated_advanced:
            new_events = _fetch_events_since(
                iss["repo"], iss["number"], prev_updated, token,
            )
            actors = [
                (e.get("actor") or {}).get("login", "")
                for e in new_events
            ]
            if is_brand_new:
                # Treat the creator as an implicit actor for brand-new issues
                # (we just learned about it, so its existence is itself activity).
                actors.append(iss.get("author", ""))

            if actors:
                non_self_issue_event = any(a and a != AUTHOR for a in actors)
            else:
                # Empty events: pure body edit (no /events log) or similar.
                # Heuristic above — only the author can edit body silently.
                non_self_issue_event = (iss.get("author") != AUTHOR)

        # Compose signal. Suppress when comments delta is all-self AND any
        # issue-level update is all-self. Surface either signal independently.
        signal: list[str] = []
        if comments_increased and non_self_count > 0:
            signal.append(f"+{non_self_count} non-self comments")
        if updated_advanced and non_self_issue_event:
            signal.append(f"updated {iss['updated_at'][:10]}")

        new_activity = bool(signal) or bool(priority_signals)
        priority = bool(priority_signals)

        flagged.append({
            **iss,
            "new_activity": new_activity,
            "priority": priority,
            "priority_signals": priority_signals,
            "non_self_comment_count": non_self_count,
            "self_only_issue_update": updated_advanced and not non_self_issue_event,
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
    flagged = diff_against_state(issues, state, token)

    if args.changes_only:
        flagged = [f for f in flagged if f["new_activity"]]

    # Sort: priority first, then new activity, then by updated_at desc
    flagged.sort(key=lambda x: (not x.get("priority", False), not x["new_activity"], x["updated_at"]), reverse=False)
    flagged.reverse()  # most recent / highest priority first

    if not args.no_update:
        new_state = update_state(issues, state)
        _save_state(new_state)

    if args.pretty:
        new_count = sum(1 for f in flagged if f["new_activity"])
        priority_count = sum(1 for f in flagged if f.get("priority", False))
        primary = sum(1 for f in flagged if f.get("scope") == "primary")
        external = sum(1 for f in flagged if f.get("scope") == "external")
        print(f"Tracked issues: {len(flagged)} ({primary} primary-repo, {external} external) — "
              f"{new_count} with NEW activity, {priority_count} PRIORITY\n")
        for f in flagged:
            marker = "⚡" if f.get("priority") else ("🆕" if f["new_activity"] else "  ")
            scope = "📦" if f.get("scope") == "primary" else "🔗"
            print(f"  {marker} {scope} {f['repo']}#{f['number']} [{f['state']}] — {f['title'][:60]}")
            print(f"      {f['signal']} | {f['html_url']}")
            for ps in f.get("priority_signals", []):
                print(f"      ⚡ {ps}")
    else:
        out = {
            "checked_at": int(time.time()),
            "primary_repos": PRIMARY_REPOS,
            "external_repos": EXTERNAL_REPOS,
            "otter_birthday": OTTER_BIRTHDAY,
            "total_count": len(flagged),
            "new_activity_count": sum(1 for f in flagged if f["new_activity"]),
            "priority_count": sum(1 for f in flagged if f.get("priority", False)),
            "issues": flagged,
        }
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
