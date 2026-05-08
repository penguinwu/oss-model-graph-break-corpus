# Subagents Migration Plan

**Phase 1 commit, 2026-05-08.** Migrates the existing `skills/adversary-review/` to `subagents/adversary-review/` and creates `subagents/file-issue/`. Addresses adversary-review case 2026-05-08-153427-file-issue-design gaps #1, #6, #9, #12.

## Decisions baked in

| Question | Decision | Rationale |
|---|---|---|
| Top-level dir name | `subagents/` | Explicit about what these are (Claude Code sub-agents); distinct from peer agents (Rocky/Beaver) |
| File name for the activation contract | KEEP `SKILL.md` (not `AGENT.md`) | Harness convention — `~/.claude/plugins/cache/**/SKILL.md` is what discovery walks. Renaming breaks `myclaw-skills` discovery. (Gap #12.) |
| Log file format | Per-file-per-case + generated aggregator | Eliminates concurrent-append corruption; makes per-case migrations easier (Gap #9.) |
| Case ID namespace | Sub-agent prefix (`adv-` / `file-`) | Avoids collision when both sub-agents log on the same minute; greppable in footer markers + cross-references (Gap #6.) |

## What moves where

| Before | After |
|---|---|
| `skills/adversary-review/SKILL.md` | `subagents/adversary-review/SKILL.md` |
| `skills/adversary-review/persona.md` | `subagents/adversary-review/persona.md` |
| `skills/adversary-review/escalation_template.md` | `subagents/adversary-review/escalation_template.md` |
| `skills/adversary-review/V2_PROMOTION.md` | `subagents/adversary-review/V2_PROMOTION.md` |
| `skills/adversary-review/reviews_log.md` | **Deleted** — content split into per-case files (see below) + new `invocations_log.md` (generated aggregate) |

## Existing 3 entries in reviews_log.md — what happens

The pre-migration `reviews_log.md` has 3 logged entries (smoke, cohort-regen-fix, doc-vs-impl) plus one being added now (this design review = 4 total). All 4 migrate to per-file-per-case form with prefixed case_ids:

| Pre-migration case_id | New case_id | New path |
|---|---|---|
| `2026-05-07-093400-smoke` | `adv-2026-05-07-093400-smoke` | `subagents/adversary-review/invocations/adv-2026-05-07-093400-smoke.md` |
| `2026-05-07-124100-cohort-regen-fix` | `adv-2026-05-07-124100-cohort-regen-fix` | `subagents/adversary-review/invocations/adv-2026-05-07-124100-cohort-regen-fix.md` |
| `2026-05-07-190947-doc-vs-impl` | `adv-2026-05-07-190947-doc-vs-impl` | `subagents/adversary-review/invocations/adv-2026-05-07-190947-doc-vs-impl.md` |
| `2026-05-08-153427-file-issue-design` | `adv-2026-05-08-153427-file-issue-design` | `subagents/adversary-review/invocations/adv-2026-05-08-153427-file-issue-design.md` |

Migration script splits the markdown body by `### <case_id>` headings, prepends a YAML frontmatter from the existing per-row table, writes each as a new file. **`persona_sha` values are PRESERVED verbatim** — they reference git revs of `skills/adversary-review/persona.md` at invocation time, which was the canonical path at that point. Adding a top-line note to each migrated file: "Pre-migration entry. `persona_sha` references the file at its pre-migration path `skills/adversary-review/persona.md@<sha>`. Use `git show <sha>:skills/adversary-review/persona.md` to retrieve."

This preserves audit-chain provenance without rewriting history.

## What `invocations_log.md` becomes

A GENERATED aggregate index. One row per case file, with the YAML header fields. Built by `tools/build_invocations_log.py subagents/<sub-agent>/`. The aggregator runs:

- Nightly via cron (added in Phase 1 commit)
- On demand: any time `tools/file_issues.py --via-skill` runs, it triggers a rebuild first to ensure freshness

`invocations_log.md` is **regeneratable from `invocations/*.md`** — it is NOT the source of truth. Hand-editing is forbidden; the per-case files are canonical.

The aggregate's purpose: scanning the audit chain at a glance (`cat subagents/adversary-review/invocations_log.md`). Deep-dive uses the per-case files.

## Reference updates required (the grep gate)

Pre-commit grep for stale references — must be 0 hits before commit:

```bash
# These paths must NOT appear anywhere except in MIGRATION.md and per-case files' historical notes:
grep -r "skills/adversary-review" --include="*.md" --include="*.py" .  # except MIGRATION.md, invocations/*.md
grep -r "skills/file-issue" --include="*.md" --include="*.py" .          # never — that path never existed in committed form
grep -r "reviews_log.md" --include="*.md" --include="*.py" .             # except MIGRATION.md
```

Files known to need updating in the same commit:

| File | Change |
|---|---|
| `~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md` | Update adversary-review trigger path: `skills/adversary-review/SKILL.md` → `subagents/adversary-review/SKILL.md`. Add new file-issue trigger: "before filing ANY issue → invoke `subagents/file-issue/SKILL.md`. Skipping requires Peng's written approval." |
| `tools/file_issues.py` | Modified to require `--via-skill` (argparse `required=True`) and read from `subagents/<sub-agent>/invocations/<case_id>.md` (not the aggregate log) |
| `tools/check_doc_consistency.py` | Add 2 new rules: `subagent_required_fields` (every `subagents/*/SKILL.md` has YAML frontmatter with required fields), `subagent_paths_migrated` (no stale `skills/(adversary-review\|file-issue)` refs anywhere) |

## Backward-compat (none — clean break)

No symlinks. The grep gate ensures all internal references are updated atomically. Any external references (e.g., commits in git history that mention `skills/adversary-review/...`) remain accurate as historical truth. Per-case files include their original-path provenance for audit reconstruction.

## Phase 1 commit checklist

- [ ] `git mv skills/adversary-review subagents/adversary-review`
- [ ] Run migration script: split `reviews_log.md` → `invocations/adv-*.md` per-case files, delete the aggregate
- [ ] Build first aggregate: `tools/build_invocations_log.py subagents/adversary-review/`
- [ ] Add `subagents/adversary-review/RETROSPECTIVE.md` with empty header (cadence per Phase 1 SKILL.md Step 7)
- [ ] Create `subagents/file-issue/` with `SKILL.md` + `persona.md` + `PREREQ-CHECK.sh` + empty `invocations/` + empty `invocations_log.md`
- [ ] Create `subagents/README.md` with the convention + audit chain explainer
- [ ] Create `subagents/MIGRATION.md` (this file)
- [ ] Modify `tools/file_issues.py`: add `--via-skill` as `required=True` to `corpus-issue` and `pytorch-upstream` posting subcommands; add per-case-file lookup; deprecate `correctness-apply` (umbrella-issue path)
- [ ] Add `tools/build_invocations_log.py` aggregator
- [ ] Add `tools/test_file_issues.py` with: `test_via_skill_required` (CLI rejects naked posts) + `test_body_sha256_round_trip`
- [ ] Extend `tools/check_doc_consistency.py` with `subagent_required_fields` + `subagent_paths_migrated` rules
- [ ] Update `~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md` triggers
- [ ] Run grep gate — must be 0 hits for stale paths
- [ ] Run all tests — `tools/test_*.py` + `tools/check_doc_consistency.py` must pass
- [ ] Commit: "subagents/: migrate adversary-review + scaffold file-issue (Phase 1)"
