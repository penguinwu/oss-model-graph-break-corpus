#!/usr/bin/env python3
"""File GitHub issues with duplicate detection.

Prevents duplicate issue creation by checking existing issues (open + recently
closed) for title similarity before creating. All issue creation for this repo
should go through this script.

Usage:
  # File a single issue
  python file_issue.py --title "Bug title" --body "Description" --labels dynamo

  # File from a JSON file (batch)
  python file_issue.py --from-json issues.json

  # Dry run — check for duplicates without creating
  python file_issue.py --from-json issues.json --dry-run

  # Force create even if duplicate detected
  python file_issue.py --title "Bug title" --body "Desc" --force

JSON format for --from-json:
  [{"title": "...", "body": "...", "labels": ["dynamo"]}]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from difflib import SequenceMatcher

PROXY_URL = "http://localhost:7824/fetch"
REPO = "penguinwu/oss-model-graph-break-corpus"
SIMILARITY_THRESHOLD = 0.8  # titles >= 80% similar are considered duplicates


def _load_token():
    """Load GitHub token from gh CLI config."""
    config_path = os.path.expanduser("~/.config/gh/hosts.yml")
    with open(config_path) as f:
        for line in f:
            if "oauth_token:" in line:
                return line.split("oauth_token:")[1].strip()
    raise RuntimeError("GitHub token not found in ~/.config/gh/hosts.yml")


def _api_request(url, method="GET", body=None, token=None):
    """Make a GitHub API request via the web proxy."""
    payload = {
        "url": url,
        "method": method,
        "headers": {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
        },
    }
    if body is not None:
        payload["body"] = body

    req = urllib.request.Request(
        PROXY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())

    if not data.get("ok", False) and data.get("status", 0) >= 400:
        raise RuntimeError(f"API error: HTTP {data.get('status')} — {data.get('content', '')[:200]}")

    content = data.get("content", "")
    if isinstance(content, str):
        return json.loads(content) if content else {}
    return content


def fetch_existing_issues(token, state="all", per_page=100):
    """Fetch existing issues (open + closed) for duplicate checking."""
    issues = []
    page = 1
    while True:
        url = (f"https://api.github.com/repos/{REPO}/issues"
               f"?state={state}&per_page={per_page}&page={page}")
        batch = _api_request(url, token=token)
        if not batch:
            break
        # Filter out pull requests (GitHub API returns PRs in issues endpoint)
        issues.extend(i for i in batch if "pull_request" not in i)
        if len(batch) < per_page:
            break
        page += 1
    return issues


def find_duplicates(title, existing_issues):
    """Find existing issues with similar titles. Returns list of (issue, similarity)."""
    dupes = []
    title_lower = title.lower().strip()
    for issue in existing_issues:
        existing_lower = issue["title"].lower().strip()
        # Exact match
        if title_lower == existing_lower:
            dupes.append((issue, 1.0))
            continue
        # Fuzzy match
        ratio = SequenceMatcher(None, title_lower, existing_lower).ratio()
        if ratio >= SIMILARITY_THRESHOLD:
            dupes.append((issue, ratio))
    return sorted(dupes, key=lambda x: -x[1])


def create_issue(title, body, labels=None, token=None):
    """Create a GitHub issue. Returns (number, url)."""
    data = {"title": title, "body": body}
    if labels:
        data["labels"] = labels

    result = _api_request(
        f"https://api.github.com/repos/{REPO}/issues",
        method="POST",
        body=data,
        token=token,
    )
    return result.get("number"), result.get("html_url")


def main():
    parser = argparse.ArgumentParser(description="File GitHub issues with duplicate detection")
    parser.add_argument("--title", help="Issue title (single issue mode)")
    parser.add_argument("--body", help="Issue body (single issue mode)")
    parser.add_argument("--body-file", help="Read body from file")
    parser.add_argument("--labels", nargs="*", help="Issue labels")
    parser.add_argument("--from-json", help="JSON file with issues to create")
    parser.add_argument("--dry-run", action="store_true", help="Check for duplicates only")
    parser.add_argument("--force", action="store_true", help="Create even if duplicate detected")
    args = parser.parse_args()

    # Build issue list
    issues_to_create = []
    if args.from_json:
        with open(args.from_json) as f:
            issues_to_create = json.load(f)
    elif args.title:
        body = args.body or ""
        if args.body_file:
            with open(args.body_file) as f:
                body = f.read()
        issues_to_create = [{"title": args.title, "body": body, "labels": args.labels or []}]
    else:
        parser.error("Provide --title or --from-json")

    token = _load_token()

    # Fetch existing issues for dedup
    print(f"Fetching existing issues from {REPO}...", file=sys.stderr)
    existing = fetch_existing_issues(token)
    print(f"  Found {len(existing)} existing issues", file=sys.stderr)

    created = 0
    skipped = 0

    for issue in issues_to_create:
        title = issue["title"]
        body = issue.get("body", "")
        labels = issue.get("labels", [])

        # Check for duplicates
        dupes = find_duplicates(title, existing)
        if dupes:
            best_match, similarity = dupes[0]
            state_icon = "🟢" if best_match["state"] == "open" else "🔴"
            print(f"\n⚠️  DUPLICATE DETECTED for: {title}", file=sys.stderr)
            print(f"   {state_icon} #{best_match['number']}: {best_match['title']} "
                  f"({similarity:.0%} similar, {best_match['state']})", file=sys.stderr)

            if not args.force:
                print(f"   SKIPPED (use --force to override)", file=sys.stderr)
                skipped += 1
                continue

            print(f"   --force: creating anyway", file=sys.stderr)

        if args.dry_run:
            print(f"  [dry-run] Would create: {title}", file=sys.stderr)
            continue

        num, url = create_issue(title, body, labels, token)
        if num:
            print(f"  ✅ Created #{num}: {title}")
            print(f"     {url}")
            # Add to existing list so subsequent issues in batch are checked against it
            existing.append({"number": num, "title": title, "state": "open"})
            created += 1
        else:
            print(f"  ❌ Failed: {title}", file=sys.stderr)

        # Rate limit courtesy
        time.sleep(1)

    print(f"\nDone: {created} created, {skipped} skipped (duplicates)", file=sys.stderr)

    # Output summary as JSON for programmatic use
    print(json.dumps({"created": created, "skipped": skipped, "total": len(issues_to_create)}))


if __name__ == "__main__":
    main()
