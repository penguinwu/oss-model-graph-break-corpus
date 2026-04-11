#!/usr/bin/env python3
"""Fetch comments for recent oncall:pt2 issues (last year) via web proxy.

Filters out DISABLED/UNSTABLE test issues. Saves to pt2_issue_comments.json.
"""

import json
import urllib.request
import sys
import time
import os

PROXY_URL = "http://localhost:7824/fetch"
GH_TOKEN = None
OUTPUT_DIR = "/home/pengwu/projects/oss-model-graph-break-corpus/data/pt2-issues"
ISSUES_PATH = os.path.join(OUTPUT_DIR, "pt2_all_issues.json")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pt2_issue_comments.json")
CUTOFF = "2025-04-11"

# Read token
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

# Load issues
with open(ISSUES_PATH) as f:
    all_issues = json.load(f)

# Filter: recent, has comments, not DISABLED/UNSTABLE
issues = [i for i in all_issues
    if i.get('created_at', '') >= CUTOFF
    and i.get('comments', 0) > 0
    and not i.get('title', '').startswith('DISABLED ')
    and not i.get('title', '').startswith('UNSTABLE ')
]

print(f"Fetching comments for {len(issues)} issues...", file=sys.stderr)

# Resume support
results = {}
if os.path.exists(OUTPUT_PATH) and "--resume" in sys.argv:
    with open(OUTPUT_PATH) as f:
        results = json.load(f)
    print(f"Resuming with {len(results)} issues already fetched", file=sys.stderr)

fetched = 0
errors = 0

for idx, issue in enumerate(issues):
    num = str(issue['number'])
    if num in results:
        continue

    url = f"https://api.github.com/repos/pytorch/pytorch/issues/{num}/comments?per_page=100"

    payload = {
        "url": url,
        "max_size": 5000000,
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
        print(f"#{num}: request error: {e}", file=sys.stderr)
        errors += 1
        if errors > 10:
            print("Too many errors, stopping", file=sys.stderr)
            break
        time.sleep(5)
        continue

    if not data.get("ok"):
        error = data.get("error", "unknown")
        if "rate limit" in str(error).lower():
            print("Rate limited, waiting 60s...", file=sys.stderr)
            time.sleep(60)
            continue
        print(f"#{num}: API error: {error}", file=sys.stderr)
        errors += 1
        continue

    try:
        comments = json.loads(data.get("content", "[]"))
    except json.JSONDecodeError:
        print(f"#{num}: JSON decode error", file=sys.stderr)
        continue

    # Store trimmed comments (user, body truncated)
    results[num] = [{
        "user": c.get("user", {}).get("login", "?"),
        "body": (c.get("body") or "")[:2000],
        "created_at": c.get("created_at", ""),
    } for c in comments]

    fetched += 1
    if fetched % 50 == 0:
        print(f"Progress: {fetched}/{len(issues) - len(results) + fetched} issues fetched, {idx+1}/{len(issues)} processed", file=sys.stderr)
        # Save partial results
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f)

    time.sleep(1)  # rate limit

# Save final results
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)

total_comments = sum(len(v) for v in results.values())
print(f"\nDone: {len(results)} issues, {total_comments} comments saved to {OUTPUT_PATH}", file=sys.stderr)
