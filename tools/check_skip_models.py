#!/usr/bin/env python3
"""check_skip_models.py — schema + freshness gate on sweep/skip_models.json.

Per Peng directive (from audit_new_models adversary case adv-2026-05-10-150000
gap #7): skip-list adds are TEMPORARY and MUST come with a follow-up fix task.
The legacy flat-string-array format can't represent that. This tool enforces
the upgraded dict-of-objects schema:

    {
      "ModelA": {
        "reason": "<≥30 chars documenting why model is skipped>",
        "follow_up_task": "<task description, OR null with reason explaining permanence>",
        "added": "YYYY-MM-DD"
      },
      ...
    }

Plus: warns on entries older than 90 days (skip-list rot detector).

Usage:
  python3 tools/check_skip_models.py [path/to/skip_models.json]
    Validate. Exit 0 = OK, 1 = policy violation, 2 = legacy format detected.

  python3 tools/check_skip_models.py --diff <git-ref>
    Validate only entries added/modified vs the given git ref. Pre-commit-friendly.

  python3 tools/check_skip_models.py --no-warn-stale
    Skip the 90-day staleness warning (CI-friendly).

Schema upgrade designed: dict-of-objects + per-entry metadata. Legacy flat-array
format is REJECTED by default (forces migration before tool runs).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "sweep" / "skip_models.json"
MIN_REASON_LEN = 30
STALENESS_DAYS = 90

REQUIRED_FIELDS = ("reason", "follow_up_task", "added")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_entry(name: str, entry: dict, *, today: date) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one entry."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(entry, dict):
        errors.append(f"{name}: entry must be an object, got {type(entry).__name__}")
        return errors, warnings
    for field in REQUIRED_FIELDS:
        if field not in entry:
            errors.append(f"{name}: missing required field '{field}'")
    if "reason" in entry:
        reason = entry["reason"] or ""
        if not isinstance(reason, str) or len(reason) < MIN_REASON_LEN:
            errors.append(
                f"{name}: 'reason' must be a string ≥{MIN_REASON_LEN} chars; "
                f"got len={len(reason) if isinstance(reason, str) else '?'}: {str(reason)[:60]!r}"
            )
    if "follow_up_task" in entry:
        ft = entry["follow_up_task"]
        # Allow null IFF reason explains why no follow-up (≥30 chars handles this)
        if ft is not None and (not isinstance(ft, str) or len(ft) < 5):
            errors.append(
                f"{name}: 'follow_up_task' must be a string (or null with reason "
                f"explaining permanence); got {ft!r}"
            )
    if "added" in entry:
        added = entry["added"]
        if not isinstance(added, str) or not DATE_RE.match(added):
            errors.append(f"{name}: 'added' must be 'YYYY-MM-DD'; got {added!r}")
        else:
            try:
                added_date = datetime.strptime(added, "%Y-%m-%d").date()
                age = (today - added_date).days
                if age > STALENESS_DAYS:
                    warnings.append(
                        f"{name}: added {age} days ago ({added}); STALE — skip-list "
                        f"adds are TEMPORARY. Investigate root cause + remove if fixed."
                    )
            except ValueError as e:
                errors.append(f"{name}: 'added' parse failed: {e}")
    return errors, warnings


def changed_entry_names(path: Path, git_ref: str) -> set[str] | None:
    """Return set of names added/modified vs git_ref. None on error."""
    try:
        prev = subprocess.run(
            ["git", "show", f"{git_ref}:{path.relative_to(REPO_ROOT)}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        return None
    if prev.returncode != 0:
        return None
    try:
        prev_data = json.loads(prev.stdout)
    except json.JSONDecodeError:
        return None
    cur_data = json.loads(path.read_text())
    if not isinstance(cur_data, dict):
        return None
    if isinstance(prev_data, dict):
        return {n for n, e in cur_data.items() if prev_data.get(n) != e}
    # prev was legacy flat list, cur is dict → all entries are "modified"
    return set(cur_data.keys())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(DEFAULT_PATH),
                    help=f"Path (default: {DEFAULT_PATH.relative_to(REPO_ROOT)})")
    ap.add_argument("--diff", default=None, metavar="GIT_REF",
                    help="Validate only entries added/modified vs git ref (pre-commit-friendly)")
    ap.add_argument("--no-warn-stale", action="store_true",
                    help="Suppress 90-day staleness warnings (CI-friendly)")
    ap.add_argument("--allow-legacy-format", action="store_true",
                    help="Allow legacy flat-string-array schema (used during migration only)")
    args = ap.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"OK: {path} does not exist; nothing to validate")
        return 0
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {rel}: {e}", file=sys.stderr)
        return 1

    if isinstance(data, list):
        if args.allow_legacy_format:
            print(f"OK: {rel} uses legacy flat-string-array format ({len(data)} entries); "
                  f"--allow-legacy-format set — skipping per-entry validation")
            return 0
        print(
            f"FAIL: {rel} uses legacy flat-string-array format. Per Peng directive "
            f"(audit_new_models adv-2026-05-10-150000 gap #7), skip-list adds are "
            f"TEMPORARY and require per-entry metadata (reason / follow_up_task / "
            f"added).\n\n"
            f"Migrate to dict-of-objects:\n"
            f'  {{\n    "ModelName": {{\n      "reason": "<≥30 chars>",\n'
            f'      "follow_up_task": "<task or null>",\n      "added": "YYYY-MM-DD"\n'
            f'    }},\n    ...\n  }}\n\n'
            f"Pass --allow-legacy-format to bypass during migration window only.",
            file=sys.stderr,
        )
        return 2

    if not isinstance(data, dict):
        print(f"ERROR: {rel} root must be a dict (or legacy list); got {type(data).__name__}",
              file=sys.stderr)
        return 1

    if not data:
        print(f"OK: {rel} has 0 entries; nothing to validate")
        return 0

    if args.diff:
        validate_names = changed_entry_names(path, args.diff)
        if validate_names is None:
            print(f"--diff {args.diff}: prior file unparseable or absent; "
                  f"validating ALL {len(data)} entries", file=sys.stderr)
            validate_names = set(data.keys())
    else:
        validate_names = set(data.keys())

    today = date.today()
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for name, entry in data.items():
        if name not in validate_names:
            continue
        errors, warnings = validate_entry(name, entry, today=today)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if all_warnings and not args.no_warn_stale:
        print(f"WARN: {len(all_warnings)} stale entries in {rel}:", file=sys.stderr)
        for w in all_warnings:
            print(f"  - {w}", file=sys.stderr)

    if all_errors:
        print(f"FAIL: {len(all_errors)} schema violations in {rel}:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"OK: {len(validate_names)} of {len(data)} entries validated; "
          f"no errors, {len(all_warnings)} stale warnings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
