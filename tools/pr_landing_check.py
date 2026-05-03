#!/usr/bin/env python3
"""Authoritative pytorch/pytorch PR landing check (handles ghstack).

GitHub's `merged: false` field is unreliable for pytorch — many PRs land via
ghstack/MergeBot and show `merged: false` even when the commit is on main.
This script checks the commit log to give the real verdict.

Verdicts: LANDED_GH | LANDED_GHSTACK | NOT_LANDED | OPEN
With --branch, also checks ancestor of release/2.X.

Usage: pr_landing_check.py PR_NUM [PR_NUM...] [--branch release/2.12] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote


PROXY = "http://localhost:7824/fetch"
PYTORCH_REPO = "pytorch/pytorch"
STOPWORDS = {"add", "fix", "support", "make", "use", "the", "a", "an", "and",
             "for", "in", "to", "of", "on", "with", "by", "or", "via", "as",
             "be", "is", "are", "from", "during", "all", "it", "this", "that",
             "compile", "compiling", "compilation", "dynamo"}


def _gh_token() -> str | None:
    p = Path.home() / ".config" / "gh" / "hosts.yml"
    if not p.exists():
        return None
    for line in p.read_text().splitlines():
        if line.strip().startswith("oauth_token:"):
            return line.split(":", 1)[1].strip()
    return None


def _gh(url: str, token: str, *, accept: str = "application/vnd.github+json",
        max_size: int = 600_000) -> dict:
    """GET via local web proxy. Returns parsed JSON dict."""
    payload = json.dumps({
        "url": url, "method": "GET", "max_size": max_size,
        "headers": {"Accept": accept, "Authorization": f"Bearer {token}"},
    }).encode()
    req = urllib.request.Request(PROXY, data=payload,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        wrapper = json.loads(resp.read())
    if not wrapper.get("ok"):
        raise RuntimeError(f"proxy error for {url}: {wrapper}")
    content = wrapper.get("content", "")
    return json.loads(content) if content else {}


def find_landed_commit(pr_number: int, token: str) -> dict | None:
    """Return commit dict whose subject contains '(#PR_NUMBER)', else None."""
    url = (f"https://api.github.com/search/commits?"
           f"q=repo:{PYTORCH_REPO}+%23{pr_number}&sort=committer-date&order=desc")
    try:
        result = _gh(url, token, accept="application/vnd.github.cloak-preview+json")
    except Exception as e:
        print(f"WARN: commit search failed: {e}", file=sys.stderr)
        return None
    needle = f"(#{pr_number})"
    matches = [c for c in result.get("items", [])
               if needle in c["commit"]["message"]]
    return matches[0] if matches else None


def is_ancestor(commit_sha: str, branch: str, token: str) -> dict | None:
    """Check if commit is ancestor of branch via compare API.
    behind_by==0 → commit IS ancestor (= it's in the branch).

    Compare API includes file diffs even with per_page=1 (can be 100KB+ for
    branches that diverge widely). We only need 3 top-level fields, so regex
    them out instead of forcing a full JSON parse on a large payload."""
    url = (f"https://api.github.com/repos/{PYTORCH_REPO}/compare/"
           f"{commit_sha}...{quote(branch, safe='')}?per_page=1")
    try:
        payload = json.dumps({
            "url": url, "method": "GET", "max_size": 600_000,
            "headers": {"Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {token}"},
        }).encode()
        req = urllib.request.Request(PROXY, data=payload,
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = json.loads(resp.read()).get("content", "")
        # Just the 3 fields we need (avoids full-JSON parse on truncated content)
        s = re.search(r'"status":\s*"([^"]+)"', content)
        a = re.search(r'"ahead_by":\s*(\d+)', content)
        b = re.search(r'"behind_by":\s*(\d+)', content)
        if s and a and b:
            return {"status": s.group(1),
                    "ahead_by": int(a.group(1)),
                    "behind_by": int(b.group(1))}
    except Exception as e:
        print(f"WARN: compare failed for {commit_sha}...{branch}: {e}",
              file=sys.stderr)
    return None


def find_possible_successors(pr_data: dict, token: str) -> list[dict]:
    """For NOT_LANDED PRs, find candidate successor PRs by same author with
    overlapping title keywords (>=2 distinctive words shared). Returns [] if
    no candidates found.

    Heuristic exists because 'NOT_LANDED' != 'abandoned' — author may have
    pivoted to a broader/cleaner approach (e.g. PR #179630 → #181552)."""
    author = pr_data.get("user", {}).get("login", "")
    closed_at = pr_data.get("closed_at", "")
    title = pr_data.get("title", "")
    if not (author and closed_at and title):
        return []

    title_words = set(re.findall(r"[a-z_][a-z0-9_]{2,}",
                                  re.sub(r"\[.*?\]", "", title).lower()))
    keywords = title_words - STOPWORDS
    if not keywords:
        return []

    try:
        closed = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
    except Exception:
        return []
    win_start = (closed - timedelta(days=14)).date().isoformat()
    win_end = (closed + timedelta(days=21)).date().isoformat()

    # Search same-author PRs in window; extract (number, title) only — search
    # results have nested objects that confuse multi-field regex.
    q = (f"repo:{PYTORCH_REPO}+author:{author}+is:pr"
         f"+created:{win_start}..{win_end}")
    url = (f"https://api.github.com/search/issues?q={q}"
           f"&sort=created&order=desc&per_page=30")
    try:
        content_dict = _gh(url, token)
    except Exception:
        return []
    # Parse from raw content for robustness against truncation
    content = json.dumps(content_dict)
    candidates = []
    seen = {pr_data["number"]}
    for m in re.finditer(
        r'"number":\s*(\d+)[^{}]*?"title":\s*"((?:[^"\\]|\\.)*)"', content,
    ):
        n = int(m.group(1))
        if n in seen:
            continue
        seen.add(n)
        cand_title = m.group(2).encode().decode("unicode_escape")
        cand_words = set(re.findall(r"[a-z_][a-z0-9_]{2,}",
                                     re.sub(r"\[.*?\]", "", cand_title).lower()))
        shared = (keywords & cand_words) - STOPWORDS
        if len(shared) >= 2:
            candidates.append({
                "pr_number": n, "title": cand_title,
                "shared_keywords": sorted(shared),
            })
    return candidates


def check_pr(pr_number: int, branch: str | None, token: str) -> dict:
    """Return dict with verdict + landing commit + (optional) branch ancestry."""
    pr = _gh(f"https://api.github.com/repos/{PYTORCH_REPO}/pulls/{pr_number}",
             token)
    base_ref = pr.get("base", {}).get("ref", "")

    result = {
        "pr_number": pr_number,
        "title": pr.get("title", ""),
        "user": pr.get("user", {}).get("login", ""),
        "github_state": pr.get("state"),
        "github_merged": pr.get("merged"),
        "github_closed_at": pr.get("closed_at"),
        "is_ghstack_branch": base_ref.startswith("gh/") and "/base" in base_ref,
        "verdict": None,
        "landed_commit": None,
        "landed_at": None,
    }

    if pr.get("state") == "open":
        result["verdict"] = "OPEN"
        return result

    if pr.get("merged"):
        result["verdict"] = "LANDED_GH"
        result["landed_commit"] = pr.get("merge_commit_sha")
        result["landed_at"] = pr.get("merged_at")
    else:
        landed = find_landed_commit(pr_number, token)
        if landed:
            result["verdict"] = "LANDED_GHSTACK"
            result["landed_commit"] = landed["sha"]
            result["landed_at"] = landed["commit"]["committer"]["date"]
            result["landed_committer"] = landed["commit"]["committer"]["name"]
            result["landed_message_subject"] = (
                landed["commit"]["message"].split("\n")[0])
        else:
            result["verdict"] = "NOT_LANDED"
            try:
                successors = find_possible_successors(pr, token)
                if successors:
                    result["possible_successors"] = successors
            except Exception as e:
                result["successor_search_error"] = str(e)

    if branch and result["landed_commit"]:
        cmp_result = is_ancestor(result["landed_commit"], branch, token)
        if cmp_result:
            behind = cmp_result.get("behind_by", -1)
            key = f"in_{branch.replace('/', '_')}"
            result[key] = (behind == 0)
            result[f"compare_to_{branch.replace('/', '_')}"] = {
                "status": cmp_result.get("status"),
                "ahead_by": cmp_result.get("ahead_by"),
                "behind_by": behind,
            }
    return result


def format_text(r: dict) -> str:
    out = [f"PR #{r['pr_number']}: {r['title']}"]
    out.append(f"  Author: {r['user']}"
               f"{' (ghstack PR — base ref starts with gh/)' if r.get('is_ghstack_branch') else ''}")
    out.append(f"  GitHub: state={r['github_state']}, merged={r['github_merged']}, "
               f"closed_at={r['github_closed_at']}")

    v = r["verdict"]
    if v == "OPEN":
        out.append(f"  → VERDICT: OPEN (still being reviewed)")
    elif v == "LANDED_GH":
        out.append(f"  → VERDICT: ✅ LANDED via GitHub Merge button")
        out.append(f"     commit {r['landed_commit'][:12]} at {r['landed_at']}")
    elif v == "LANDED_GHSTACK":
        out.append(f"  → VERDICT: ✅ LANDED via ghstack (PyTorch MergeBot)")
        out.append(f"     commit {r['landed_commit'][:12]} at {r['landed_at']}")
        out.append(f"     committer: {r.get('landed_committer','?')}")
        out.append(f"     subject: {r.get('landed_message_subject','')[:120]}")
        out.append(f"     ⚠️  GitHub shows merged=false because ghstack uses "
                   f"internal merge flow.")
        out.append(f"        DO NOT trust GitHub's merged field for ghstack PRs.")
    elif v == "NOT_LANDED":
        out.append(f"  → VERDICT: ❌ NOT LANDED — closed without any commit "
                   f"reaching main")
        out.append(f"     (no '(#{r['pr_number']})' commit found in {PYTORCH_REPO})")
        succ = r.get("possible_successors", [])
        if succ:
            out.append(f"     ⚠️  POSSIBLE SUCCESSOR PRs found "
                       f"({len(succ)} candidate{'s' if len(succ)>1 else ''}):")
            out.append(f"        Author may have pivoted to different approach. "
                       f"INSPECT MANUALLY:")
            for s in succ[:5]:
                out.append(f"          • #{s['pr_number']}: {s['title'][:80]}")
                out.append(f"            shared keywords: {s['shared_keywords']}")
            if len(succ) > 5:
                out.append(f"          + {len(succ)-5} more (use --json for full list)")
        else:
            out.append(f"     (no successor candidates found by "
                       f"same-author + similar-title heuristic)")

    for k, in_branch in r.items():
        if k.startswith("in_release_"):
            branch = k.replace("in_", "").replace("_", "/")
            cmp_data = r.get(f"compare_to_{k.replace('in_','')}", {})
            if in_branch:
                out.append(f"  → IN {branch}: ✅ YES "
                           f"(ahead_by={cmp_data.get('ahead_by')}, behind_by=0)")
            else:
                out.append(f"  → IN {branch}: ❌ NO "
                           f"(status={cmp_data.get('status')}, "
                           f"behind_by={cmp_data.get('behind_by')})")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pr_numbers", nargs="+", type=int)
    p.add_argument("--branch", default=None,
                   help="Check release branch ancestry (e.g. release/2.12)")
    p.add_argument("--json", action="store_true",
                   help="Output JSON instead of human-readable text")
    args = p.parse_args()

    token = _gh_token()
    if not token:
        print("ERROR: no GitHub token at ~/.config/gh/hosts.yml", file=sys.stderr)
        return 1

    results = []
    for n in args.pr_numbers:
        try:
            results.append(check_pr(n, args.branch, token))
        except Exception as e:
            print(f"ERROR for PR #{n}: {e}", file=sys.stderr)
            results.append({"pr_number": n, "error": str(e)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"PR #{r['pr_number']}: ERROR: {r['error']}\n")
            else:
                print(format_text(r))
                print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
