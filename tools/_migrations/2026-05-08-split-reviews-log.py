#!/usr/bin/env python3
"""ONE-SHOT migration (2026-05-08): split skills/adversary-review/reviews_log.md
into per-case files at subagents/adversary-review/invocations/adv-<case_id>.md.

Used during the subagents/ migration. Adds `adv-` prefix to each case_id.
Preserves persona_sha verbatim with a note that it references the pre-migration
file path.

LESSON LEARNED (preserved for future similar migrations):
The original regex `[a-z_]+` for parsing YAML keys MISSED `output_sha256`
because the key contains digits. Always use `[a-z0-9_]+` for YAML key matching.
This one-shot script was hand-patched mid-implementation to add `output_sha256`
to frontmatter after running. Future migrations should use the corrected regex
from the start.

Per adversary impl-review case adv-2026-05-08-161753-file-issue-impl gap #9.

Run: python3 tools/_migrations/2026-05-08-split-reviews-log.py
(But this script is one-shot; the original reviews_log.md no longer exists.)
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOG = REPO_ROOT / "subagents/adversary-review/reviews_log.md"  # historical
OUT_DIR = REPO_ROOT / "subagents/adversary-review/invocations"

if not LOG.exists():
    sys.exit(f"This is a one-shot historical script. The source file "
             f"{LOG.relative_to(REPO_ROOT)} no longer exists "
             f"(deleted post-migration). Restore from git history if needed: "
             f"`git show <pre-migration-sha>:skills/adversary-review/reviews_log.md > /tmp/orig.md`")

OUT_DIR.mkdir(parents=True, exist_ok=True)
content = LOG.read_text()

# Find every "### <case_id>" header
header_rx = re.compile(r"^### (\d{4}-\d{2}-\d{2}-\d{6}-[a-z0-9-]+)\s*$", re.MULTILINE)
matches = [(m.group(1), m.start()) for m in header_rx.finditer(content)]
if not matches:
    sys.exit("ERROR: no case_id headers found")

print(f"Found {len(matches)} case entries: {[m[0] for m in matches]}")

for i, (case_id, start) in enumerate(matches):
    end = matches[i + 1][1] if i + 1 < len(matches) else len(content)
    block = content[start:end].rstrip()

    # Remove the leading "### <case_id>" line; embed in frontmatter instead
    block_lines = block.split("\n")
    body = "\n".join(block_lines[1:]).strip()

    # Parse the simple table at top to extract field values.
    # CRITICAL: regex must allow digits in keys (output_sha256, body_sha256, etc.).
    table_rx = re.compile(r"^\|\s*([a-z0-9_]+)\s*\|\s*(.+?)\s*\|\s*$", re.MULTILINE)
    fields = {}
    for fm in table_rx.finditer(body):
        key = fm.group(1).strip()
        val = fm.group(2).strip()
        if key in {"field", "value"}:
            continue
        fields[key] = val

    new_case_id = f"adv-{case_id}"
    fm_lines = [
        "---",
        f"case_id: {new_case_id}",
        f"original_case_id: {case_id}",
        "subagent: adversary-review",
        "migrated_from: skills/adversary-review/reviews_log.md",
        "migration_date: 2026-05-08",
    ]
    for k in ("date_utc", "trigger", "files", "persona_sha", "verdict", "output_sha256"):
        if k in fields:
            v = fields[k]
            if ":" in v or "|" in v:
                v = '"' + v.replace('"', '\\"') + '"'
            fm_lines.append(f"{k}: {v}")
    fm_lines.append("---")
    fm_lines.append("")
    fm_lines.append(
        "> **Pre-migration entry.** `persona_sha` references the file at its "
        "pre-migration path `skills/adversary-review/persona.md@<sha>`. "
        "Use `git show <sha>:skills/adversary-review/persona.md` to retrieve."
    )
    fm_lines.append("")

    out = "\n".join(fm_lines) + "\n" + body + "\n"
    out_path = OUT_DIR / f"{new_case_id}.md"
    out_path.write_text(out)
    print(f"  wrote {out_path.relative_to(REPO_ROOT)} ({len(out)} bytes)")

print(f"\nDone. {len(matches)} per-case files written to {OUT_DIR.relative_to(REPO_ROOT)}/")
print(f"Now safe to delete: {LOG.relative_to(REPO_ROOT)}")
