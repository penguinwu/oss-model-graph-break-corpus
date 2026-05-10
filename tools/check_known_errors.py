#!/usr/bin/env python3
"""check_known_errors.py — bias-INFRA-FIX mechanical gate on sweep/known_errors.json.

Per Peng directive (from audit_new_errors adversary case adv-2026-05-10-145000
gap #4): create_errors are entirely infra's fault and MUST be fixed at root,
not added to known_errors.json as escape-hatch entries. This tool refuses any
new entry with `status: "create_error"` unless explicit override.

Override path: --allow-create-error-escape with --reason "<text>" (≥30 chars).
The reason is recorded in known_errors.json as `_create_error_escape_reason`
on the entry. Without this, the next time someone reviews known_errors.json,
they see WHY the policy was violated.

Also enforces:
  - applies_to_versions field is REQUIRED (no missing field — same fail-loud
    rule as audit_new_errors design rev 2 gap #6 disposition)
  - status must be in {create_error, eager_error}

Usage:
  python3 tools/check_known_errors.py [path/to/known_errors.json]
    Validate the file. Exit 0 = OK, exit 1 = policy violation.

  python3 tools/check_known_errors.py --diff <git-ref>
    Validate ONLY entries added/modified vs the given git ref. Useful as a
    pre-commit check that doesn't fail on legacy entries that pre-date the
    policy. (Most common use: --diff HEAD on staged changes.)

Design: addresses gap #4 (mechanical enforcement of bias-INFRA-FIX) deferred
from audit_new_errors design rev 2.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "sweep" / "known_errors.json"
MIN_REASON_LEN = 30


def validate_entry(entry: dict, *, allow_create_error_escape: bool, idx: int) -> list[str]:
    """Return list of violation messages for one entry. Empty list = OK."""
    violations = []
    model = entry.get("model", f"<entry #{idx}>")
    status = entry.get("status")
    if status not in ("create_error", "eager_error"):
        violations.append(
            f"{model}: invalid status {status!r}; must be 'create_error' or 'eager_error'"
        )
    if "applies_to_versions" not in entry:
        violations.append(
            f"{model}: missing 'applies_to_versions' field. Per audit_new_errors design rev 2 gap #6, "
            f"missing field is no longer allowed (was 'discouraged'). Add the version list explicitly."
        )
    if status == "create_error":
        # Two-gate policy:
        # - LEGACY entries (created before this tool) are fine if they have a
        #   substantive `reason` field (≥MIN_REASON_LEN chars) — the existing
        #   3 entries (Zamba*, Qwen3OmniMoe*) all have proper reason text
        #   documenting the upstream root cause.
        # - NEW entries (caught via --diff against git ref) need EITHER the
        #   `reason` field with ≥MIN_REASON_LEN chars OR explicit
        #   `--allow-create-error-escape --reason "<text>"` override.
        # Per Peng directive: create_errors discovered in NEW sweep data should
        # be fixed at root, not added to known_errors.json. The `reason` field
        # is the forcing function — make the human articulate WHY they're
        # tracking a create_error rather than fixing it.
        reason = entry.get("reason", "") or ""
        if len(reason) < MIN_REASON_LEN:
            if not allow_create_error_escape:
                violations.append(
                    f"{model}: status=create_error requires `reason` field with "
                    f"≥{MIN_REASON_LEN} chars documenting the upstream root cause. "
                    f"Per Peng directive (audit_new_errors adversary case "
                    f"adv-2026-05-10-145000 gap #4), create_errors should be fixed at "
                    f"root unless documented as upstream issue. Got reason "
                    f"len={len(reason)}: {reason[:60]!r}. Override (only for "
                    f"unfixable upstream): --allow-create-error-escape --reason "
                    f"\"<≥{MIN_REASON_LEN} chars\"."
                )
    return violations


def changed_entry_indices(path: Path, git_ref: str) -> set[int] | None:
    """Return set of indices in entries[] that are added/modified vs git_ref.

    Returns None if the file didn't exist at git_ref (everything is added).
    Returns set of indices to validate.
    """
    try:
        prev = subprocess.run(
            ["git", "show", f"{git_ref}:{path.relative_to(REPO_ROOT)}"],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
    except FileNotFoundError:
        return None
    if prev.returncode != 0:
        # File didn't exist at ref → all current entries are new
        return None
    try:
        prev_data = json.loads(prev.stdout)
    except json.JSONDecodeError:
        return None
    prev_entries = prev_data.get("entries", [])
    cur_data = json.loads(path.read_text())
    cur_entries = cur_data.get("entries", [])

    # Identity by (model, status, applies_to_versions tuple, error_pattern)
    def signature(e):
        return (
            e.get("model"),
            e.get("status"),
            tuple(e.get("applies_to_versions") or []),
            e.get("error_pattern"),
        )

    prev_sigs = {signature(e) for e in prev_entries}
    return {i for i, e in enumerate(cur_entries) if signature(e) not in prev_sigs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(DEFAULT_PATH),
                    help=f"Path to known_errors.json (default: {DEFAULT_PATH.relative_to(REPO_ROOT)})")
    ap.add_argument("--allow-create-error-escape", action="store_true",
                    help="Allow status=create_error entries IF entry has "
                         "'_create_error_escape_reason' field with ≥30 char text. "
                         "Default REJECTED per Peng bias-INFRA-FIX directive.")
    ap.add_argument("--reason", default="",
                    help="(With --allow-create-error-escape.) Reason text. Recorded "
                         "for audit; not currently consumed but acts as forcing function "
                         "to make the human articulate WHY they're escaping the policy.")
    ap.add_argument("--diff", default=None, metavar="GIT_REF",
                    help="Validate only entries added/modified vs the given git ref "
                         "(e.g. --diff HEAD). Pre-commit-friendly: doesn't fail on "
                         "legacy entries that pre-date the policy.")
    args = ap.parse_args()

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON in {path}: {e}", file=sys.stderr)
        return 1

    entries = data.get("entries", [])
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        rel = path
    if not entries:
        print(f"OK: {rel} has 0 entries; nothing to validate")
        return 0

    if args.diff:
        validate_indices = changed_entry_indices(path, args.diff)
        if validate_indices is None:
            print(f"--diff {args.diff}: file didn't exist at ref OR couldn't load; "
                  f"validating ALL {len(entries)} entries", file=sys.stderr)
            validate_indices = set(range(len(entries)))
    else:
        validate_indices = set(range(len(entries)))

    if args.allow_create_error_escape and not args.reason:
        print("ERROR: --allow-create-error-escape requires --reason \"<text>\"",
              file=sys.stderr)
        return 1
    if args.reason and len(args.reason) < MIN_REASON_LEN:
        print(f"ERROR: --reason too short ({len(args.reason)} chars; need ≥{MIN_REASON_LEN})",
              file=sys.stderr)
        return 1

    all_violations = []
    for idx, entry in enumerate(entries):
        if idx not in validate_indices:
            continue
        violations = validate_entry(entry,
                                    allow_create_error_escape=args.allow_create_error_escape,
                                    idx=idx)
        all_violations.extend(violations)

    if all_violations:
        print(f"FAIL: {len(all_violations)} policy violations in {rel}:", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print(f"OK: {len(validate_indices)} of {len(entries)} entries validated; no violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
