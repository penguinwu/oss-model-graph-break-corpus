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

## 2026-05-08T20:23 ET — Major persona revision after first invocation

**First invocation (`file-2026-05-08-170223-issue77-review`)** revealed a structural defect in the persona's notion of "actionable." Mode A reviewed corpus issue #77 (layerdrop pattern) and returned `reframe` with 7 gaps. The TOP gap was "no `single_fix_claim` — body lists 3 alternative directions." Mode A's rewrite proposal told Otter to "Pick ONE direction as the primary ask."

**Peng's correction (verbatim, 2026-05-08 20:19 ET):**

> I do not trust Otter's ability to suggest potential fixes for an issue. So reject the proposal of creating 3 issues for each direction.
>
> I see there is ambiguity in "actionable". Right now, actionable means whether the compiler can reproduce the problem in their own environment and take it from there.
>
> The propose Fix pattern is a recurring anti-pattern. For the pytorch/pytorch issue that we opened and tagged Alban, Otter also suggested a fix but refuted by Alban. We should stay away from suggesting fixes in any issues we create unless we have clear evidence that a recent PR had caused a regression (needs to be proven), then we can point out he offending PR.

### What changed in the persona/skill (5 places)

1. **persona.md criterion #4 redefined** — "Actionable (= reproducible)": maintainer can reproduce the symptom in their own environment from the body alone. The body does NOT propose a fix. Single carve-out: `regression_evidence` field (commit sha + bisect proof) for the only case naming a fix is allowed.
2. **persona.md Mode A check #1** — `single_fix_claim` field replaced with `repro_strategy` (the concrete command/script the maintainer runs).
3. **persona.md Mode A check #8 (NEW)** — fix-suggestion anti-pattern detection. Forbidden section headers + inline phrases enumerated. Verdict `reframe`.
4. **persona.md Mode B templates** — "Proposed fix" / "What this issue closes" / "Possible directions" sections REMOVED. Templates end at Environment + Source. Note added: "the maintainer reads the symptom + repro + environment + source and decides the fix."
5. **SKILL.md** — new "What this skill does NOT do" section explicitly forbids fix-suggestion content. Step 1 triage field requirements updated (`repro_strategy` replaces `single_fix_claim`; `regression_evidence` added as optional).

### Mechanical pinning (so the rule can't silently drift)

- `tools/check_doc_consistency.py::rule_no_fix_suggestions_in_templates` — scans persona.md's inline template blocks for forbidden section headers. Pre-push gate.
- `tools/test_check_doc_consistency.py` — 4 new tests: positive (current persona passes), negative (synthetic persona with "Proposed fix" header in template fails), negative (synthetic persona with "Possible directions" fails), allow-prose (anti-pattern phrases in PROSE outside templates are OK).
- `tools/test_file_issues.py::test_persona_documents_no_fix_suggestion_rule` — pins that persona.md contains the load-bearing language. Deletion would fail the test.
- `tools/test_file_issues.py::test_skill_md_documents_no_fix_suggestion_rule` — same pin for SKILL.md.

### The recursion lesson (worth preserving)

Mode A's first invocation REINFORCED the anti-pattern it was supposed to prevent. The persona was trained on the wrong notion of "actionable." This is the most important Phase 2 candidate Otter has surfaced so far: **Mode A's persona definitions can have load-bearing defects that propagate into every invocation, and only show up when a real review surfaces a result that exposes them.** The remedy: every retrospective MUST sample a recent invocation and ask "did the persona enforce something Peng would object to?" Don't just count gap-find rates.

### Remaining work for the re-invocation

After this commit lands, re-invoke Mode A on issue #77 with the amended persona. Expected: Mode A drops gaps #1 (no single_fix_claim) and #5 (3-direction enumeration dilutes); keeps gaps #2 (no MRE), #3 (stale counts), #4 (no env), #6 (PII forward-looking), #7 (title polish). Verdict probably stays `reframe` (5 substantive gaps still > 3 cap), but the rewrite proposal should NOT include "Pick ONE direction." If it still does, the persona amendment didn't take.

---

(Append next retrospective after `file-` invocations #2 + #3 + #4. Original entry date: 2026-05-08; major-revision update: 2026-05-08T20:23 ET.)
