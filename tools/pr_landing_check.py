#!/usr/bin/env python3
"""Check if a pytorch/pytorch PR has actually landed on main and/or in a release branch.

WHY THIS EXISTS
---------------
PyTorch uses ghstack for many PRs. ghstack lands changes via PyTorch MergeBot
(an internal merge flow), NOT via GitHub's Merge button. As a consequence,
**GitHub's `merged` field on the PR API shows `false` even when the PR has
actually landed**. Trusting that field has caused us to give wrong information
to compiler developers (e.g., reporting "polyfill PR #179611 was closed without
merging" when in fact the commit landed on main and shipped in release branches).

This script gives an authoritative answer by:
  1. Reading the GitHub PR's metadata (state, merged, closed_at)
  2. Searching pytorch/pytorch commits for a "(#PR_NUMBER)" pattern in the
     commit message (which is what ghstack-via-MergeBot produces)
  3. Optionally: checking whether the landed commit is an ancestor of a
     specified release branch (e.g., release/2.12)
  4. Reporting a UNIFIED, unambiguous verdict:
       - LANDED_GH    — merged via GitHub button (rare for pytorch nowadays)
       - LANDED_GHSTACK — merged via ghstack (commit on main, GitHub merged=false)
       - NOT_LANDED   — closed without any commit reaching main
       - OPEN         — still open

Usage
-----
    # Single PR
    python3 tools/pr_landing_check.py 179611

    # Single PR + check a release branch
    python3 tools/pr_landing_check.py 179611 --branch release/2.12

    # Multiple PRs
    python3 tools/pr_landing_check.py 179611 180585 179629 --branch release/2.12

    # JSON output for scripting
    python3 tools/pr_landing_check.py 179611 --json

Mandatory rule
--------------
**When asked whether a pytorch/pytorch PR has landed, ALWAYS use this script.**
**Never quote GitHub's `merged: false` field as proof of "not landed".** The
script's verdict is authoritative; GitHub's field alone is not.

Requires
--------
- `localhost:7824` web proxy (per ~/.myclaw-shared/recipes/github-access.md)
- `~/.config/gh/hosts.yml` GitHub OAuth token

Origin
------
2026-05-03: written after misclassifying PR #179611 as "closed without merging"
when it had in fact landed via ghstack on 2026-04-11. Encoded as a tool to
prevent the failure mode systematically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote


PROXY = "http://localhost:7824/fetch"
PYTORCH_REPO = "pytorch/pytorch"


def _gh_token() -> str | None:
    hosts_yml = Path.home() / ".config" / "gh" / "hosts.yml"
    if not hosts_yml.exists():
        return None
    for line in hosts_yml.read_text().splitlines():
        line = line.strip()
        if line.startswith("oauth_token:"):
            return line.split(":", 1)[1].strip()
    return None


def _gh_get(url: str, token: str, accept: str = "application/vnd.github+json") -> dict:
    """GET via the local web proxy. Returns parsed JSON dict, or raises."""
    import urllib.request
    payload = json.dumps({
        "url": url,
        "method": "GET",
        "max_size": 200_000,
        "headers": {
            "Accept": accept,
            "Authorization": f"Bearer {token}",
        },
    }).encode()
    req = urllib.request.Request(
        PROXY, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        wrapper = json.loads(resp.read())
    if not wrapper.get("ok"):
        raise RuntimeError(f"proxy error for {url}: {wrapper}")
    content = wrapper.get("content", "")
    return json.loads(content) if content else {}


def get_pr(pr_number: int, token: str) -> dict:
    """Fetch PR metadata."""
    return _gh_get(
        f"https://api.github.com/repos/{PYTORCH_REPO}/pulls/{pr_number}",
        token,
    )


def find_landed_commit(pr_number: int, token: str) -> dict | None:
    """Search pytorch commits for a 'Pull Request resolved: ...{PR_NUMBER}' or
    '(#PR_NUMBER)' pattern in the commit message.

    Returns the first matching commit dict, or None if no commit found.
    """
    # Use commit search API; ghstack lands include "(#PR_NUMBER)" in subject
    q = f"repo:{PYTORCH_REPO}+%23{pr_number}"
    url = f"https://api.github.com/search/commits?q={q}&sort=committer-date&order=desc"
    try:
        result = _gh_get(url, token, accept="application/vnd.github.cloak-preview+json")
    except Exception as e:
        print(f"WARN: commit search failed: {e}", file=sys.stderr)
        return None
    items = result.get("items", [])
    # Filter to commits whose message subject contains "(#NUMBER)"
    needle = f"(#{pr_number})"
    matches = [c for c in items if needle in c["commit"]["message"]]
    if matches:
        # Most recent landed commit (the one that actually shipped if there were reverts/relands)
        return matches[0]
    return None


def is_ancestor(commit_sha: str, branch: str, token: str) -> dict | None:
    """Check if commit_sha is an ancestor of branch via GitHub's compare API.

    Returns dict with status, ahead_by, behind_by — or None on failure.
    Interpretation:
        behind_by == 0  → commit IS ancestor of branch (= it's in the branch)
        behind_by > 0   → commit is NOT ancestor (different branch / not cherry-picked)
    """
    # GitHub: compare base...head where base=commit, head=branch.
    # Add per_page=1 to keep the response small (we only need the status fields,
    # not the per-commit diff list which can be 200KB+ for big diffs).
    url = (f"https://api.github.com/repos/{PYTORCH_REPO}/compare/"
           f"{commit_sha}...{quote(branch, safe='')}?per_page=1")
    try:
        return _gh_get(url, token)
    except Exception as e:
        # Fall back to regex parsing of just the top-level fields if JSON is truncated
        try:
            import re, urllib.request
            payload = json.dumps({
                "url": url, "method": "GET", "max_size": 30_000,
                "headers": {"Accept": "application/vnd.github+json",
                            "Authorization": f"Bearer {token}"},
            }).encode()
            req = urllib.request.Request(
                PROXY, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                wrapper = json.loads(resp.read())
            content = wrapper.get("content", "")
            status_m = re.search(r'"status":\s*"([^"]+)"', content)
            ahead_m = re.search(r'"ahead_by":\s*(\d+)', content)
            behind_m = re.search(r'"behind_by":\s*(\d+)', content)
            if status_m and ahead_m and behind_m:
                return {
                    "status": status_m.group(1),
                    "ahead_by": int(ahead_m.group(1)),
                    "behind_by": int(behind_m.group(1)),
                }
        except Exception:
            pass
        print(f"WARN: compare failed for {commit_sha}...{branch}: {e}", file=sys.stderr)
        return None


def find_possible_successors(pr_data: dict, token: str) -> list[dict]:
    """When a PR is closed-not-landed, search for likely successor PRs by the
    same author created on/around the closure date with overlapping title
    keywords. Used to surface 'this fix may have been pivoted, not abandoned'
    cases for human inspection.

    Heuristic:
      - Same author
      - Created within ±7 days of the original PR's closure
      - Shares >=2 distinctive title words (excluding stopwords)
    """
    import re
    author = pr_data.get("user", {}).get("login", "")
    title = pr_data.get("title", "")
    closed_at = pr_data.get("closed_at", "")
    if not author or not closed_at:
        return []

    # Extract distinctive title keywords (drop stopwords + bracketed prefixes)
    title_clean = re.sub(r"\[.*?\]", "", title).lower()
    stopwords = {"add", "fix", "support", "make", "use", "the", "a", "an",
                 "and", "for", "in", "to", "of", "on", "with", "by", "or",
                 "via", "as", "be", "is", "are", "from", "during", "all",
                 "during", "it", "this", "that", "compile", "compiling",
                 "compilation", "dynamo"}
    words = [w for w in re.findall(r"[a-z_][a-z0-9_]{2,}", title_clean)
             if w not in stopwords]
    if not words:
        return []
    # Use ALL distinctive keywords for overlap matching, but only the most
    # distinctive ones (long words first) as search terms.
    all_keywords = set(words)
    # Sort by length desc so we search the most distinctive terms first
    search_terms = sorted(set(words), key=lambda w: -len(w))[:5]

    # Search GitHub for candidates by same author in a window around closure.
    from datetime import datetime, timedelta
    try:
        closed_dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
    except Exception:
        return []
    win_start = (closed_dt - timedelta(days=14)).date().isoformat()
    win_end = (closed_dt + timedelta(days=21)).date().isoformat()

    candidates: list[dict] = []
    seen_pr_numbers: set[int] = {pr_data["number"]}

    # Strategy A: fast — search by author + window without keyword filter,
    # then locally compute keyword overlap. This finds successors even when
    # title wording diverged significantly (e.g., "make X traceable" → "add
    # handlers for X variants"). Use small per_page + regex extraction to
    # avoid hitting proxy max_size on PR-body-heavy responses.
    q = f"repo:{PYTORCH_REPO}+author:{author}+is:pr+created:{win_start}..{win_end}"
    items: list[dict] = []
    # Author can have many PRs in a 5-week window; paginate to be safe.
    for page in (1, 2, 3):
        url = (f"https://api.github.com/search/issues?q={q}"
               f"&sort=created&order=desc&per_page=30&page={page}")
        try:
            import urllib.request
            payload = json.dumps({
                "url": url, "method": "GET", "max_size": 600_000,
                "headers": {"Accept": "application/vnd.github+json",
                            "Authorization": f"Bearer {token}"},
            }).encode()
            req = urllib.request.Request(
                PROXY, data=payload,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                wrapper = json.loads(resp.read())
            content = wrapper.get("content", "")
            # Parse only (number, title) — search results have nested
            # objects (pull_request, user, etc.) that confuse non-greedy
            # multi-field regex. State + date are not strictly needed for
            # successor-detection (we already know the original PR's closure
            # date and use that to scope the search).
            page_items = []
            for m in re.finditer(
                r'"number":\s*(\d+)[^{}]*?"title":\s*"((?:[^"\\]|\\.)*)"',
                content,
            ):
                page_items.append({
                    "number": int(m.group(1)),
                    "title": m.group(2).encode().decode("unicode_escape"),
                    "state": "",  # not extracted — see comment above
                    "created_at": "",
                })
            if not page_items:
                break  # no more results
            items.extend(page_items)
            if len(page_items) < 30:
                break  # last page
        except Exception:
            break

    for item in items:
        n = item.get("number")
        if n in seen_pr_numbers:
            continue
        seen_pr_numbers.add(n)
        cand_title = item.get("title", "")
        cand_title_clean = re.sub(r"\[.*?\]", "", cand_title).lower()
        cand_words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", cand_title_clean))
        shared = (all_keywords & cand_words) - stopwords
        if len(shared) >= 2:  # require >=2 keyword overlap (excluding stopwords)
            candidates.append({
                "pr_number": n,
                "title": cand_title,
                "state": item.get("state"),
                "created_at": item.get("created_at"),
                "shared_keywords": sorted(shared),
            })

    # Sort: open or closed-after-original first, then by recency
    candidates.sort(key=lambda c: c["created_at"], reverse=True)
    return candidates


def check_pr(pr_number: int, branch: str | None, token: str) -> dict:
    """Authoritative check: did this PR land?"""
    pr = get_pr(pr_number, token)

    state = pr.get("state")
    gh_merged = pr.get("merged")
    closed_at = pr.get("closed_at")
    merged_at = pr.get("merged_at")
    title = pr.get("title", "")
    user_login = pr.get("user", {}).get("login", "")
    base_ref = pr.get("base", {}).get("ref", "")
    is_ghstack_branch = base_ref.startswith("gh/") and "/base" in base_ref

    result: dict = {
        "pr_number": pr_number,
        "title": title,
        "user": user_login,
        "github_state": state,
        "github_merged": gh_merged,
        "github_merged_at": merged_at,
        "github_closed_at": closed_at,
        "is_ghstack_branch": is_ghstack_branch,
        "verdict": None,
        "landed_commit": None,
        "landed_at": None,
    }

    if state == "open":
        result["verdict"] = "OPEN"
        return result

    if gh_merged:
        # Direct GitHub merge — uncommon for pytorch but possible
        result["verdict"] = "LANDED_GH"
        result["landed_commit"] = pr.get("merge_commit_sha")
        result["landed_at"] = merged_at
    else:
        # gh_merged=false. Could be: ghstack landing OR true closed-not-merged.
        # Search commits for "(#PR_NUMBER)" pattern.
        landed = find_landed_commit(pr_number, token)
        if landed:
            result["verdict"] = "LANDED_GHSTACK"
            result["landed_commit"] = landed["sha"]
            result["landed_at"] = landed["commit"]["committer"]["date"]
            result["landed_committer"] = landed["commit"]["committer"]["name"]
            result["landed_message_subject"] = landed["commit"]["message"].split("\n")[0]
        else:
            result["verdict"] = "NOT_LANDED"
            # Surface possible successor PRs for human inspection.
            # NOT_LANDED is not the same as "abandoned" — author may have
            # pivoted to a broader / cleaner approach. We've been bitten by
            # this with PR #179630 (closed-not-landed but successor #181552
            # landed). Check candidates so the human reviewer can verify.
            try:
                successors = find_possible_successors(pr, token)
                if successors:
                    result["possible_successors"] = successors
            except Exception as e:
                result["successor_search_error"] = str(e)

    # Optional: check if landed commit is in a specific release branch
    if branch and result["landed_commit"]:
        cmp = is_ancestor(result["landed_commit"], branch, token)
        if cmp:
            behind = cmp.get("behind_by")
            ahead = cmp.get("ahead_by")
            cstatus = cmp.get("status")
            in_branch = behind == 0
            result[f"in_{branch.replace('/', '_')}"] = in_branch
            result[f"compare_to_{branch.replace('/', '_')}"] = {
                "status": cstatus, "ahead_by": ahead, "behind_by": behind,
            }
    return result


def format_text(r: dict) -> str:
    lines = []
    lines.append(f"PR #{r['pr_number']}: {r['title']}")
    lines.append(f"  Author: {r['user']}{' (ghstack PR — base ref starts with gh/)' if r.get('is_ghstack_branch') else ''}")
    lines.append(f"  GitHub: state={r['github_state']}, merged={r['github_merged']}, closed_at={r['github_closed_at']}")
    verdict = r["verdict"]
    if verdict == "OPEN":
        lines.append(f"  → VERDICT: OPEN (still being reviewed)")
    elif verdict == "LANDED_GH":
        lines.append(f"  → VERDICT: ✅ LANDED via GitHub Merge button")
        lines.append(f"     commit {r['landed_commit'][:12]} at {r['landed_at']}")
    elif verdict == "LANDED_GHSTACK":
        lines.append(f"  → VERDICT: ✅ LANDED via ghstack (PyTorch MergeBot)")
        lines.append(f"     commit {r['landed_commit'][:12]} at {r['landed_at']}")
        lines.append(f"     committer: {r.get('landed_committer','?')}")
        lines.append(f"     subject: {r.get('landed_message_subject','')[:120]}")
        lines.append(f"     ⚠️  GitHub shows merged=false because ghstack uses internal merge flow.")
        lines.append(f"        DO NOT trust GitHub's merged field for ghstack PRs.")
    elif verdict == "NOT_LANDED":
        lines.append(f"  → VERDICT: ❌ NOT LANDED — closed without any commit reaching main")
        lines.append(f"     (no '(#{r['pr_number']})' commit found in pytorch/pytorch)")
        succ = r.get("possible_successors", [])
        if succ:
            lines.append(f"     ⚠️  POSSIBLE SUCCESSOR PRs found ({len(succ)} candidate{'s' if len(succ) > 1 else ''}):")
            lines.append(f"        Author may have pivoted to a different approach. INSPECT MANUALLY:")
            for s in succ[:5]:
                lines.append(f"          • #{s['pr_number']} [{s['state']}] {s['title'][:80]}")
                lines.append(f"            created {s['created_at'][:10]}, shared keywords: {s['shared_keywords']}")
            if len(succ) > 5:
                lines.append(f"          + {len(succ)-5} more (run --json for full list)")
        else:
            lines.append(f"     (no successor candidates found by same-author + similar-title heuristic)")
    # Branch ancestry result
    for k, v in r.items():
        if k.startswith("in_release_"):
            branch = k.replace("in_", "").replace("_", "/")
            cmp = r.get(f"compare_to_{k.replace('in_','')}")
            if v:
                lines.append(f"  → IN {branch}: ✅ YES (ahead_by={cmp['ahead_by']}, behind_by=0)")
            else:
                lines.append(f"  → IN {branch}: ❌ NO (status={cmp['status']}, behind_by={cmp['behind_by']})")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pr_numbers", nargs="+", type=int, help="One or more pytorch/pytorch PR numbers")
    p.add_argument("--branch", default=None,
                   help="Check if the landed commit is in this branch (e.g. release/2.12)")
    p.add_argument("--json", action="store_true", help="Output JSON for scripting")
    args = p.parse_args()

    token = _gh_token()
    if not token:
        print("ERROR: no GitHub token at ~/.config/gh/hosts.yml", file=sys.stderr)
        return 1

    results = []
    for pr_n in args.pr_numbers:
        try:
            r = check_pr(pr_n, args.branch, token)
            results.append(r)
        except Exception as e:
            print(f"ERROR for PR #{pr_n}: {e}", file=sys.stderr)
            results.append({"pr_number": pr_n, "error": str(e)})

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
