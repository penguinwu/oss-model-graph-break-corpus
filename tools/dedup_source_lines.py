#!/usr/bin/env python3
"""Per-source-line dedup search against existing issue bodies.

Phase 2 V2 of the file-issue dedup gate (per Peng directive 2026-05-11 11:36 ET):

> "be very careful whenever we open umbrella issues" + "we never group
>  different GB reasons into one issue"

The existing `dedup_search.py` checks title + label keyword matches — necessary
but NOT sufficient for umbrella-style filings. An umbrella's break_reason is
generic ("Detected data-dependent branching"), so its title-keyword query
returns many results, but a human (or current Mode A) doesn't enforce reading
those results' BODIES for source-line overlap.

This tool fixes that gap. Given a draft body or an explicit list of source
lines, it fetches the bodies of all open `[dynamo]`-titled issues and greps
each source line. Any match → overlap candidate.

Run examples:
    # Auto-extract source lines from a draft body file
    python3 tools/dedup_source_lines.py --draft /tmp/file-issue-<case>-draft.md

    # Use an explicit list of source lines (one per line)
    python3 tools/dedup_source_lines.py --source-lines /tmp/sites.txt

    # Dry-run / pretty output
    python3 tools/dedup_source_lines.py --draft <path> --pretty

Cautionary tale (2026-05-11): umbrella #122 was filed 2026-05-05 with 9 named
source sites. Six days earlier, #77 (LayerDrop pattern) and #78 (mask==1
pattern) were filed and covered 9 of those 9 sites between them. Title-keyword
dedup missed it because the umbrella's title contains the word "umbrella," not
the per-site source paths. THIS tool catches that class.

Requires Python 3.9+.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("ERROR: dedup_source_lines.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from file_issues import fetch_open_issues, REPO_SLUG  # noqa: E402

# Match transformers / diffusers source-line citations of the form
#   transformers/models/<name>/modeling_<name>.py:<NNN>
# Be conservative: require both .py and a line number, accept any path under
# transformers/ or diffusers/ or torch/ or sweep/.
SOURCE_LINE_RE = re.compile(
    r"\b((?:transformers|diffusers|torch|sweep|tools)/[A-Za-z0-9_/.]+\.py:\d+)\b"
)


def extract_source_lines(text: str) -> list[str]:
    """Pull source-line citations (file:NNN) from arbitrary markdown text."""
    seen: set[str] = set()
    out: list[str] = []
    for m in SOURCE_LINE_RE.finditer(text):
        s = m.group(1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def grep_in_body(body: str, source_line: str) -> bool:
    """Does the issue body contain this source-line citation?

    Match the literal `path:NN` string AND also try matching just the
    `path` portion (loose) — both are signals.
    """
    if not body:
        return False
    if source_line in body:
        return True
    # Loose match: filename + ":<digits>" with possible different line number
    # (catches case where umbrella cites :1219 but tracked issue cites :1218)
    path_part = source_line.rsplit(":", 1)[0]
    if path_part and path_part in body:
        return True
    return False


def find_overlapping_issues(
    source_lines: list[str], issues: list[dict], dynamo_only: bool = True
) -> list[dict]:
    """For each source line, find existing issues whose body cites it.

    Returns a list of {issue_num, title, source_line, match_type} dicts,
    one per (source_line, issue) overlap. Same issue can match multiple
    source lines.
    """
    out: list[dict] = []
    for issue in issues:
        title = issue.get("title", "") or ""
        body = issue.get("body", "") or ""
        if dynamo_only and "[dynamo]" not in title.lower():
            continue
        for sl in source_lines:
            exact = sl in body
            loose = (not exact) and grep_in_body(body, sl)
            if exact or loose:
                out.append({
                    "issue_num": issue.get("number"),
                    "title": title,
                    "source_line": sl,
                    "match_type": "exact" if exact else "loose-path",
                    "labels": [
                        (lbl.get("name") if isinstance(lbl, dict) else str(lbl))
                        for lbl in (issue.get("labels") or [])
                    ],
                })
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--draft", type=Path,
                     help="Path to draft body markdown — auto-extract source lines")
    grp.add_argument("--source-lines", type=Path,
                     help="Path to text file with source-line citations, one per line")
    p.add_argument("--all-issues", action="store_true",
                   help="Search ALL open issues, not just [dynamo]-titled (default: dynamo-only)")
    p.add_argument("--pretty", action="store_true",
                   help="Human-readable output (default: JSON)")
    p.add_argument("--exclude-issue", type=int, action="append", default=[],
                   help="Exclude this issue # from the search (e.g., the issue itself).")
    args = p.parse_args()

    # Load source lines
    if args.draft:
        text = args.draft.read_text()
        source_lines = extract_source_lines(text)
        if not source_lines:
            sys.exit(f"ERROR: no source-line citations found in {args.draft}. "
                     f"Pattern looked for: <repo>/<path>.py:<NNN> "
                     f"(transformers/diffusers/torch/sweep/tools roots).")
    else:
        source_lines = [
            ln.strip() for ln in args.source_lines.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if not source_lines:
            sys.exit(f"ERROR: no source lines in {args.source_lines}")

    print(f"Source lines to check: {len(source_lines)}", file=sys.stderr)
    for sl in source_lines:
        print(f"  - {sl}", file=sys.stderr)

    print(f"\nFetching open issues from {REPO_SLUG}...", file=sys.stderr)
    issues = fetch_open_issues()
    if args.exclude_issue:
        issues = [i for i in issues if i.get("number") not in args.exclude_issue]
    print(f"  {len(issues)} open issues fetched"
          f" ({'all' if args.all_issues else '[dynamo]-only filter applied below'})",
          file=sys.stderr)

    overlaps = find_overlapping_issues(
        source_lines, issues, dynamo_only=not args.all_issues
    )

    if args.pretty:
        if not overlaps:
            print("\n✓ No overlap found.")
            return 0
        print(f"\n⚠️  Found {len(overlaps)} (source_line × issue) overlap(s):\n")
        # Group by issue
        by_issue: dict[int, list[dict]] = {}
        for o in overlaps:
            by_issue.setdefault(o["issue_num"], []).append(o)
        for num, hits in sorted(by_issue.items()):
            title = hits[0]["title"][:90]
            print(f"  #{num}: {title}")
            for h in hits:
                tag = "✓" if h["match_type"] == "exact" else "~"
                print(f"    {tag} {h['source_line']} ({h['match_type']})")
        print(f"\nFor umbrella filings, this means action lives at the existing"
              f" issue(s) above for the matched sites.")
        print(f"Per Peng directive 2026-05-11: be very careful with umbrella"
              f" issues; default against opening them. For GB issues, NEVER"
              f" group different GB reasons.")
    else:
        out = {
            "source_lines_checked": source_lines,
            "issues_searched": len(issues),
            "scope": "dynamo-only" if not args.all_issues else "all-open",
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
        }
        print(json.dumps(out, indent=2, default=str))

    return 0 if not overlaps else 2


if __name__ == "__main__":
    sys.exit(main())
