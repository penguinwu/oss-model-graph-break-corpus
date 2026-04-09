#!/usr/bin/env python3
"""Fetch ALL oncall:pt2 issues from GitHub API via web proxy.

Fixed pagination: uses per_page=100 and iterates through all pages.
GitHub currently shows ~9,165 total oncall:pt2 issues (2,076 open + 7,089 closed).
"""

import json
import urllib.request
import sys
import time
import os

PROXY_URL = "http://localhost:7824/fetch"
GH_TOKEN = None
OUTPUT_DIR = "/home/pengwu/projects/oss-model-graph-break-corpus/data/pt2-issues"

# Read token from gh config
try:
    with open("/home/pengwu/.config/gh/hosts.yml") as f:
        for line in f:
            if "oauth_token" in line:
                GH_TOKEN = line.strip().split()[-1]
                break
except Exception:
    pass

if not GH_TOKEN:
    print("ERROR: No GitHub token found", file=sys.stderr)
    sys.exit(1)

all_issues = []

# Resume support: load existing partial results
PARTIAL_PATH = os.path.join(OUTPUT_DIR, "pt2_all_issues.json")
start_page = 1
if os.path.exists(PARTIAL_PATH) and "--resume" in sys.argv:
    with open(PARTIAL_PATH) as f:
        all_issues = json.load(f)
    start_page = len(all_issues) // 100 + 1
    print(f"Resuming from page {start_page} ({len(all_issues)} issues loaded)", file=sys.stderr)

page = start_page
max_pages = 200  # 200 * 100 = 20,000 max, well above 9,165

while page <= max_pages:
    url = (
        f"https://api.github.com/repos/pytorch/pytorch/issues"
        f"?labels=oncall:+pt2&state=all&per_page=100"
        f"&sort=created&direction=asc&page={page}"
    )

    payload = {
        "url": url,
        "max_size": 5000000,  # 5MB to handle 100 issues with long bodies
    }
    if GH_TOKEN:
        payload["headers"] = {"Authorization": f"token {GH_TOKEN}"}

    req = urllib.request.Request(
        PROXY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"Page {page}: request error: {e}", file=sys.stderr)
        # Retry once after a pause
        time.sleep(5)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
        except Exception as e2:
            print(f"Page {page}: retry failed: {e2}", file=sys.stderr)
            break

    if not data.get("ok"):
        error = data.get("error", "unknown")
        print(f"Page {page}: API error: {error}", file=sys.stderr)
        if "rate limit" in str(error).lower():
            print("Rate limited, waiting 60s...", file=sys.stderr)
            time.sleep(60)
            continue
        break

    content = data.get("content", "")
    try:
        issues = json.loads(content)
    except json.JSONDecodeError as e:
        print(
            f"Page {page}: JSON decode error (length={len(content)}): {e}",
            file=sys.stderr,
        )
        break

    if not issues:
        print(f"Page {page}: no more issues (empty response)", file=sys.stderr)
        break

    for issue in issues:
        labels = [l["name"] for l in issue.get("labels", [])]
        all_issues.append({
            "number": issue["number"],
            "title": issue["title"],
            "state": issue["state"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "closed_at": issue.get("closed_at"),
            "comments": issue["comments"],
            "labels": labels,
            "body": (issue.get("body") or "")[:3000],
            "html_url": issue["html_url"],
            "user": issue["user"]["login"],
            "reactions": issue.get("reactions", {}).get("total_count", 0),
        })

    print(
        f"Page {page}: got {len(issues)} issues (total: {len(all_issues)})",
        file=sys.stderr,
    )

    if len(issues) < 100:
        print(f"Page {page}: last page (got {len(issues)} < 100)", file=sys.stderr)
        break

    page += 1
    time.sleep(1)  # rate limit: stay under 30 req/min

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, "pt2_all_issues.json")

# Back up existing file
if os.path.exists(output_path):
    backup = output_path.replace(".json", "_backup.json")
    os.rename(output_path, backup)
    print(f"Backed up existing file to {backup}", file=sys.stderr)

with open(output_path, "w") as f:
    json.dump(all_issues, f, indent=2)

# Summary
open_count = sum(1 for i in all_issues if i["state"] == "open")
closed_count = sum(1 for i in all_issues if i["state"] == "closed")
print(f"\nTotal issues fetched: {len(all_issues)}", file=sys.stderr)
print(f"  Open: {open_count}, Closed: {closed_count}", file=sys.stderr)
print(f"Saved to: {output_path}", file=sys.stderr)
