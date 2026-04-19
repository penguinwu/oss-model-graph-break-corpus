#!/usr/bin/env python3
"""Check whether a PyTorch PR actually landed on main.

GitHub API's `merged` field is unreliable for ghstack PRs — they show
`state: closed, merged: false` even when the code is on main. This tool
checks the ground truth: is the commit on the main branch?

Usage:
    python tools/check_pr_status.py 179611
    python tools/check_pr_status.py 179611 --repo pytorch/pytorch
    python tools/check_pr_status.py 179611 --json
"""
import argparse
import json
import os
import subprocess
import sys


def _github_api(endpoint, token=None):
    """Call GitHub API via sudo + proxy (BPF workaround)."""
    headers = '-H "Accept: application/vnd.github+json"'
    if token:
        headers += f' -H "Authorization: Bearer {token}"'
    cmd = (
        f'curl -s -x http://fwdproxy:8080 '
        f'"https://api.github.com{endpoint}" {headers}'
    )
    result = subprocess.run(
        ["sudo", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _get_token():
    token_file = os.path.expanduser("~/.config/gh/hosts.yml")
    if os.path.exists(token_file):
        with open(token_file) as f:
            for line in f:
                if "oauth_token" in line:
                    return line.split(":")[-1].strip()
    return None


def check_pr_status(pr_number, repo="pytorch/pytorch"):
    """Return a dict with the true landed status of a PR.

    Keys:
        pr_number, repo, title, author, state, merged_via_github,
        is_ghstack, commit_sha, commit_on_main, reverted, verdict, detail
    """
    token = _get_token()
    result = {
        "pr_number": pr_number,
        "repo": repo,
        "verdict": "unknown",
        "detail": "",
    }

    # Step 1: Get PR details
    pr_data = _github_api(f"/repos/{repo}/pulls/{pr_number}", token)
    if not pr_data or "message" in pr_data:
        result["verdict"] = "error"
        result["detail"] = f"Could not fetch PR: {pr_data.get('message', 'unknown error') if pr_data else 'API call failed'}"
        return result

    result["title"] = pr_data.get("title", "")
    result["author"] = pr_data.get("user", {}).get("login", "")
    result["state"] = pr_data.get("state", "")
    result["merged_via_github"] = pr_data.get("merged", False)
    result["merge_commit_sha"] = pr_data.get("merge_commit_sha")

    # Step 2: Detect non-standard merge flows
    body = pr_data.get("body", "") or ""
    head_ref = pr_data.get("head", {}).get("ref", "")
    is_ghstack = "ghstack" in body.lower() or head_ref.startswith("gh/")
    is_phabricator = head_ref.startswith("export-D")
    uses_direct_push = is_ghstack or is_phabricator
    result["is_ghstack"] = is_ghstack
    result["is_phabricator"] = is_phabricator

    # Step 3: If GitHub says merged, trust it
    if pr_data.get("merged"):
        result["verdict"] = "landed"
        result["detail"] = f"Merged via GitHub at {pr_data.get('merged_at', '?')}"
        result["commit_sha"] = pr_data.get("merge_commit_sha")
        result["commit_on_main"] = True
        result["reverted"] = False
        return result

    # Step 4: If still open, it hasn't landed
    if pr_data.get("state") == "open":
        result["verdict"] = "open"
        result["detail"] = "PR is still open"
        result["commit_on_main"] = False
        result["reverted"] = False
        return result

    # Step 5: Closed but not merged — could be ghstack/phabricator or truly abandoned
    if not uses_direct_push:
        result["verdict"] = "closed_not_landed"
        result["detail"] = "Closed without merge (not a ghstack PR)"
        result["commit_on_main"] = False
        result["reverted"] = False
        return result

    # Step 6: ghstack/phabricator PR — check if the commit landed on main
    # Search for commits mentioning this PR number
    search_data = _github_api(
        f"/search/commits?q=repo:{repo}+{pr_number}+committer-date:>2020-01-01"
        f"&sort=committer-date&order=desc&per_page=5",
        token)

    commits = []
    if search_data and "items" in search_data:
        for item in search_data["items"]:
            msg = item.get("commit", {}).get("message", "")
            if f"#{pr_number}" in msg or f"/{pr_number}" in msg:
                commits.append({
                    "sha": item["sha"],
                    "message": msg.split("\n")[0],
                    "date": item.get("commit", {}).get("committer", {}).get("date", ""),
                })

    if not commits:
        result["verdict"] = "closed_not_landed"
        flow = "ghstack" if is_ghstack else "phabricator"
        result["detail"] = f"{flow} PR closed, but no matching commit found on main"
        result["commit_on_main"] = False
        result["reverted"] = False
        return result

    # Step 7: Check for reverts
    land_commits = [c for c in commits if not c["message"].lower().startswith("revert")]
    revert_commits = [c for c in commits if c["message"].lower().startswith("revert")]

    result["commit_sha"] = land_commits[0]["sha"] if land_commits else commits[0]["sha"]
    result["all_commits"] = commits

    if revert_commits and land_commits:
        latest_revert = max(revert_commits, key=lambda c: c["date"])
        latest_land = max(land_commits, key=lambda c: c["date"])
        if latest_land["date"] > latest_revert["date"]:
            result["verdict"] = "landed"
            flow = "ghstack" if is_ghstack else "phabricator"
            result["detail"] = (
                f"{flow} PR landed (commit {latest_land['sha'][:8]} on {latest_land['date'][:10]}). "
                f"Was reverted then re-landed."
            )
            result["commit_on_main"] = True
            result["reverted"] = False
        else:
            result["verdict"] = "reverted"
            result["detail"] = (
                f"Landed then reverted (revert {latest_revert['sha'][:8]} on {latest_revert['date'][:10]}). "
                f"Not currently on main."
            )
            result["commit_on_main"] = False
            result["reverted"] = True
    elif land_commits:
        c = land_commits[0]
        result["verdict"] = "landed"
        flow = "ghstack" if is_ghstack else "phabricator"
        result["detail"] = f"{flow} PR landed (commit {c['sha'][:8]} on {c['date'][:10]})"
        result["commit_on_main"] = True
        result["reverted"] = False
    else:
        result["verdict"] = "reverted"
        result["detail"] = "Only revert commits found — PR was landed then reverted"
        result["commit_on_main"] = False
        result["reverted"] = True

    return result


def main():
    parser = argparse.ArgumentParser(description="Check if a PyTorch PR landed on main")
    parser.add_argument("pr_number", type=int, help="PR number")
    parser.add_argument("--repo", default="pytorch/pytorch", help="GitHub repo (default: pytorch/pytorch)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = check_pr_status(args.pr_number, args.repo)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        v = result["verdict"]
        icon = {"landed": "YES", "reverted": "REVERTED", "open": "OPEN",
                "closed_not_landed": "NO", "error": "ERROR", "unknown": "??"}.get(v, "??")
        print(f"PR #{args.pr_number} — {icon}")
        print(f"  Title:   {result.get('title', '?')}")
        print(f"  Author:  {result.get('author', '?')}")
        flow = "ghstack" if result.get("is_ghstack") else "phabricator" if result.get("is_phabricator") else "standard"
        print(f"  Flow:    {flow}")
        print(f"  Verdict: {result['detail']}")

    sys.exit(0 if v in ("landed", "open") else 1)


if __name__ == "__main__":
    main()
