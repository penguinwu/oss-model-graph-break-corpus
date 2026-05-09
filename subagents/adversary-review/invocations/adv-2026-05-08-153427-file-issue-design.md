---
case_id: adv-2026-05-08-153427-file-issue-design
original_case_id: 2026-05-08-153427-file-issue-design
subagent: adversary-review
migrated_from: skills/adversary-review/reviews_log.md
migration_date: 2026-05-08
date_utc: "2026-05-08T22:34:27Z"
trigger: "other (design review of NEW file-issue subagent + subagents/ dir restructure + adversary-review migration; pre-implementation per Peng directive 2026-05-08 18:34 ET — \"send NOW before implementation\")"
files: /tmp/file-issue-design/SKILL.md (the proposed AGENT.md), /tmp/file-issue-design/persona.md (Mode A + Mode B), /tmp/adversary-review-2026-05-08-153427-file-issue-design.md (request context including dir-structure / migration plan), skills/adversary-review/SKILL.md + persona.md + reviews_log.md (existing precedent), tools/issue_filing_plan.md (existing 5-must-haves the new persona supersedes)
persona_sha: b9a495616d731234299ad9bb6638418e328d8c2e
verdict: approve-with-gaps
output_sha256: 2dcf34bfe79cb033ffcce0fc14e61bc16c47a3be9f1862900197ac5639de3a0e
---

> **Pre-migration entry.** `persona_sha` references the file at its pre-migration path `skills/adversary-review/persona.md@<sha>`. Use `git show <sha>:skills/adversary-review/persona.md` to retrieve.

| field | value |
|-------|-------|
| date_utc | 2026-05-08T22:34:27Z |
| trigger | other (design review of NEW file-issue subagent + subagents/ dir restructure + adversary-review migration; pre-implementation per Peng directive 2026-05-08 18:34 ET — "send NOW before implementation") |
| files | /tmp/file-issue-design/SKILL.md (the proposed AGENT.md), /tmp/file-issue-design/persona.md (Mode A + Mode B), /tmp/adversary-review-2026-05-08-153427-file-issue-design.md (request context including dir-structure / migration plan), skills/adversary-review/SKILL.md + persona.md + reviews_log.md (existing precedent), tools/issue_filing_plan.md (existing 5-must-haves the new persona supersedes) |
| persona_sha | b9a495616d731234299ad9bb6638418e328d8c2e |
| verdict | approve-with-gaps |
| output_sha256 | 2dcf34bfe79cb033ffcce0fc14e61bc16c47a3be9f1862900197ac5639de3a0e |

**Reviewer raw output:** see `/tmp/adv-output-2026-05-08-153427.txt` (full verbatim copy preserved at invocation time; hash matches above).

**GAPS_FOUND summary:** 12 gaps total — 4 high (1-4), 6 medium (5-10), 2 low (11-12).
- HIGH 1: SKILL.md still uses pre-migration `skills/file-issue/...` paths (12+ refs); breaks the activation contract on day one
- HIGH 2: Mode A 4-verdict label set has no `proceed-with-fixes` mode; dominant case (1 small gap, fix-and-go) has no clean home → predicted criterion erosion within 5 invocations
- HIGH 3: Mode B failure markers (OVERSCOPE/MRE_TOO_LARGE/VALIDATION_FAILED) have no documented disposition in SKILL.md Step 4; SKILL/persona contract drift on self-revision protocol
- HIGH 4: `--via-skill` enforcement contradicts itself between Step 5 (corpus uses `--body` only, no flag) and "What enforces this" (all paths require flag) — corpus path (dominant) has NO mechanical refusal
- MED 5: invocations_log.md schema missing `body_sha256` (the load-bearing audit field) + `validation_file_path` + `footer_marker`; mode_b_sha256 hashes wrong scope (includes TITLE/LABELS/SELF_CHECK)
- MED 6: Migration doesn't address existing 3 reviews_log.md entries + persona_sha values pointing at to-be-deleted file path + case_id collision risk between sub-agents
- MED 7: PII scrub only on validation file (Mode A item 7), not on draft/body for corpus target; collides with shared CLAUDE.md "Visibility Control - External Repos" rule
- MED 8: Implicit dependencies undeclared (Agent tool, CLAUDE.md trigger, --via-skill flag, gh auth, web proxy, validation venvs)
- MED 9: Concurrent log-append corruption not addressed (heartbeats + interactive); recommend per-file-per-case + aggregator
- MED 10: `--via-skill` enforcement is wishful (no implementation); no test that tool actually rejects naked posts
- LOW 11: Mode A duplicate-check accepts Otter's claim without verifying search-query quality
- LOW 12: SKILL.md → AGENT.md rename may break `myclaw-skills` discovery (walks `**/SKILL.md`)

**SUGGESTED_ADDITIONAL_TESTS:** 7 mechanical contract tests with full SETUP/ACTION/EXPECTED/DETECTS structure. Highest-leverage: 1 (subagent_required_fields rule), 2 (path_audit no stale refs), 4 (via_skill_enforcement at CLI argparse level — pre-merge gate).

**Reviewer META observations (3):**
1. **Ship in TWO phases.** Phase 1 minimal: AGENT.md + persona + log + migration + trigger + doc-consistency rules. Phase 2 (after 1-2 real invocations): templates + recipes + Mode B calibration. Matches Peng req 4 (iterate via use) — don't ship 10 files speculatively.
2. **Strong happy-path, weak failure-path.** All 4 high-severity gaps are facets of "happy-path crispness, failure-path muddiness." Walk every blind-spot "what if" before commit; pin OR explicitly defer.
3. **Missed adjacency:** no `iteration_cadence` for file-issue (existing adversary-review has "every 3 reviews, retrospective" but the corpus's reviews_log has 3 entries + no formal retrospective row — discipline-only forcing function failed).

**My disposition:** PENDING — surface to Peng for direction. All 4 high-severity gaps are real and need fixes before commit. Pre-disposition lean: accept staged-shipping recommendation; address gaps 1, 3, 4 in SKILL.md revision; address 2 in persona.md (add proceed-with-fixes); address 5, 6 in log schema + add MIGRATION.md; add Pre-requisites + concurrency notes; defer 11 (lowering expectations is fine for v1); resolve 12 by checking myclaw-skills discovery first.

**Commit:** pending — design revisions land before any new files are created. Will be backfilled.
