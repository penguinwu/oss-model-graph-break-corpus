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
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "sweep" / "known_errors.json"
MIN_REASON_LEN = 30

# Format: <org>/<repo>#<num> — e.g. "pytorch/pytorch#182339"
TRACKING_ISSUE_RE = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+#\d+$")


def validate_entry(entry: dict, *, allow_create_error_escape: bool, idx: int,
                   require_tracking_issue: bool = False) -> list[str]:
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
    # tracking_issue field: required by --require-tracking-issue (default False);
    # warned if missing on legacy entries. Per Peng directive 2026-05-10 21:04 ET
    # surfaced from MimiModel discovery: each known_errors entry should record
    # the upstream issue tracking the bug, so weekly sweep can detect when the
    # tracked issue closes (then the entry can be re-verified + removed).
    if "tracking_issue" not in entry:
        if require_tracking_issue:
            violations.append(
                f"{model}: missing 'tracking_issue' field. Per Peng directive "
                f"2026-05-10 21:04 ET, every entry must record the upstream "
                f"issue tracking the bug. Format: '<org>/<repo>#<num>' "
                f"(e.g. 'pytorch/pytorch#182339') OR null with reason explaining "
                f"why no issue exists yet."
            )
    else:
        ti = entry["tracking_issue"]
        if ti is not None:
            if not isinstance(ti, str) or not TRACKING_ISSUE_RE.match(ti):
                violations.append(
                    f"{model}: invalid 'tracking_issue' format {ti!r}; "
                    f"must be '<org>/<repo>#<num>' (e.g. 'pytorch/pytorch#182339') OR null"
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
    ap.add_argument("--require-tracking-issue", action="store_true",
                    help="Require every entry to have a 'tracking_issue' field "
                         "(format '<org>/<repo>#<num>' OR null). Default: warn but pass.")
    ap.add_argument("--check-tracking-status", action="store_true",
                    help="For each entry with tracking_issue set, fetch the issue's "
                         "current state from GitHub. Reports closed/updated tracking "
                         "issues so the corresponding known_errors entry can be "
                         "re-verified + removed. Per Peng directive 2026-05-10 21:04 ET. "
                         "Requires github-access (sudo + fwdproxy + auth token).")
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
        violations = validate_entry(
            entry,
            allow_create_error_escape=args.allow_create_error_escape,
            idx=idx,
            require_tracking_issue=args.require_tracking_issue,
        )
        all_violations.extend(violations)

    if all_violations:
        print(f"FAIL: {len(all_violations)} policy violations in {rel}:", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        return 1
    print(f"OK: {len(validate_indices)} of {len(entries)} entries validated; no violations")

    # Tracking-status check (separate from validation; informational + actionable)
    if args.check_tracking_status:
        return check_tracking_status(entries) or 0
    return 0


def check_tracking_status(entries: list) -> int:
    """For each entry with tracking_issue set, fetch GitHub issue state.

    Reports closed/updated issues so corresponding known_errors entries can
    be re-verified + removed. Returns non-zero if any tracked issue is closed
    (operator action required). Returns 0 if all tracked issues still open.

    Per Peng directive 2026-05-10 21:04 ET — surfaced from MimiModel discovery
    where pytorch/pytorch#182339 closed but our known_errors entry was unchanged.
    """
    import os as _os
    fetch_pairs = []  # (entry, tracking_issue)
    null_count = 0
    legacy_count = 0
    for e in entries:
        ti = e.get("tracking_issue")
        if ti is None:
            null_count += 1
        elif "tracking_issue" not in e:
            legacy_count += 1
        else:
            fetch_pairs.append((e, ti))

    print(f"\n=== tracking_issue status check ===")
    print(f"Entries with tracking_issue: {len(fetch_pairs)} to fetch, "
          f"{null_count} null (no issue), {legacy_count} legacy (no field)")
    if not fetch_pairs:
        return 0

    # Lazy import — needs sudo + fwdproxy via the gh-access pattern
    closed_count = 0
    open_count = 0
    error_count = 0
    closed_entries = []
    for entry, ti in fetch_pairs:
        org_repo, num = ti.split("#")
        url = f"https://api.github.com/repos/{org_repo}/issues/{num}"
        cmd = [
            "sudo", "bash", "-c",
            f"HTTPS_PROXY=http://fwdproxy:8080 curl -s -u penguinwu:$(cat ~/.github_token 2>/dev/null) "
            f"-H 'Accept: application/vnd.github+json' '{url}'"
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            data = json.loads(r.stdout) if r.returncode == 0 else None
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            data = None
        if data is None or "state" not in data:
            print(f"  ? {entry['model']:<45} {ti:<35} (fetch failed)")
            error_count += 1
            continue
        state = data["state"]
        title = data.get("title", "")[:60]
        if state == "closed":
            closed_count += 1
            closed_entries.append((entry["model"], ti, data.get("closed_at"), title))
            print(f"  ✗ {entry['model']:<45} {ti:<35} CLOSED ({data.get('closed_at')}): {title}")
        else:
            open_count += 1
            print(f"  · {entry['model']:<45} {ti:<35} open: {title}")

    print(f"\nSummary: {open_count} still open, {closed_count} CLOSED (action needed), "
          f"{error_count} fetch failed")
    if closed_count:
        print(f"\nACTION REQUIRED: {closed_count} tracking issues are closed. "
              f"For each, re-run the affected models against current torch nightly:")
        for model, ti, closed_at, title in closed_entries:
            print(f"  - {model} (tracking_issue={ti}, closed {closed_at}). "
                  f"If it now compiles cleanly, remove the known_errors entry.")
        return 1  # actionable signal
    return 0


if __name__ == "__main__":
    sys.exit(main())
