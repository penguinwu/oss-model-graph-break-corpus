# file-issue RETROSPECTIVE

Per the iteration cadence in `SKILL.md` § Step 7: after every 3 issues filed, spend 5 minutes on retrospective. Forcing function: every 4th invocation must include "retrospective check: <date of last entry>" in the disposition notes.

## 2026-05-08 — Phase 1 ship baseline

**Reviews so far (0 — Phase 1 just shipped 2026-05-08).**

Phase 1 ship surface (commit 1f36118 + the impl-fix commit 1.1):
- SKILL.md (273 lines): activation contract + 7-step procedure + authority gate + Phase 1 file inventory
- persona.md (256 lines): Mode A (adversary, 5 verdicts) + Mode B (assembler, target-aware checklists) in one file
- PREREQ-CHECK.sh: smoke-test all dependencies (8 checks)
- invocations/ + invocations_log.md (empty — first invocation pending)

Phase 1 fixes from adversary impl-review (case adv-2026-05-08-161753-file-issue-impl, 11 gaps):
1. PREREQ-CHECK.sh broken `grep -qv` logic → fixed (positive match)
2. pytorch-upstream missing case_id marker → fixed (injected as first body line)
3. Footer marker not enforced at post → fixed (`_validate_via_skill` now checks)
4. posted_url manual update → fixed (`_update_case_posted_url` auto-updates)
5. Aggregator silent on malformed input → fixed (warnings + `--strict` flag + 8 new tests)
6. `rule_subagent_required_fields` missing checks → extended (invocations_log.md + RETROSPECTIVE.md required)
7. proceed-with-fixes cap not enforced → DEFERRED to Phase 2 (deferred reason: needs Mode A output parser; better to wait until first real invocation surfaces the parsing shape)
8. mode_a_fixes_applied schema field → DOCUMENTED in SKILL.md but enforcement DEFERRED (same reason as #7)
9. Migration script not committed → fixed (added to tools/_migrations/)
10. Exclude list too broad → narrowed (specific anchored exclusions only)
11. TOCTOU race in body read → fixed (`_validate_via_skill` returns body bytes)

**Action items for first 3 invocations:**
- Apply file-issue to ONE of the 14 NGB D1 divergences (suggest: Wav2Vec2Model, highest max_diff in audio family)
- Capture: Mode A's typical verdict on first-pass NGB-correctness draft
- Capture: which Mode B calibration items fail most often on this class of issue
- Capture: how Otter's draft framing differs from the persona's expectations
- After 3 invocations, retrospective + propose persona / template / checklist edits

**Open questions (to resolve via use, not in advance):**
- Does proceed-with-fixes appear naturally on first-pass drafts, or does Mode A skip straight to reframe? (Predicts whether the cap is needed.)
- Does the corpus-bug template need separate variants for "numeric divergence" vs "graph break" (correctness vs identification)?
- Does the validation recipe for numeric divergence need a standardized re-measurement script (not just "run reproduce.py manually")?

---

(Append next retrospective after `file-` invocations #1 + #2 + #3. Date this entry: 2026-05-08.)
