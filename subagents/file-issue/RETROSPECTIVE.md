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
8. mode_a_fixes_applied schema field → DOCUMENTED in SKILL.md but enforcement DEFERRED (same reason as 7)
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

**First invocation (`file-2026-05-08-170223-issue77-review`)** revealed a structural defect in the persona's notion of "actionable." Mode A reviewed corpus issue 77 (layerdrop pattern) and returned `reframe` with 7 gaps. The TOP gap was "no `single_fix_claim` — body lists 3 alternative directions." Mode A's rewrite proposal told Otter to "Pick ONE direction as the primary ask."

**Peng's correction (verbatim, 2026-05-08 20:19 ET):**

> I do not trust Otter's ability to suggest potential fixes for an issue. So reject the proposal of creating 3 issues for each direction.
>
> I see there is ambiguity in "actionable". Right now, actionable means whether the compiler can reproduce the problem in their own environment and take it from there.
>
> The propose Fix pattern is a recurring anti-pattern. For the pytorch/pytorch issue that we opened and tagged Alban, Otter also suggested a fix but refuted by Alban. We should stay away from suggesting fixes in any issues we create unless we have clear evidence that a recent PR had caused a regression (needs to be proven), then we can point out he offending PR.

### What changed in the persona/skill (5 places)

1. **persona.md criterion 4 redefined** — "Actionable (= reproducible)": maintainer can reproduce the symptom in their own environment from the body alone. The body does NOT propose a fix. Single carve-out: `regression_evidence` field (commit sha + bisect proof) for the only case naming a fix is allowed.
2. **persona.md Mode A check 1** — `single_fix_claim` field replaced with `repro_strategy` (the concrete command/script the maintainer runs).
3. **persona.md Mode A check 8 (NEW)** — fix-suggestion anti-pattern detection. Forbidden section headers + inline phrases enumerated. Verdict `reframe`.
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

After this commit lands, re-invoke Mode A on issue 77 with the amended persona. Expected: Mode A drops gaps 1 (no single_fix_claim) and 5 (3-direction enumeration dilutes); keeps gaps 2 (no MRE), 3 (stale counts), 4 (no env), 6 (PII forward-looking), 7 (title polish). Verdict probably stays `reframe` (5 substantive gaps still > 3 cap), but the rewrite proposal should NOT include "Pick ONE direction." If it still does, the persona amendment didn't take.

---

(Append next retrospective after `file-` invocations 2 + 3 + 4. Original entry date: 2026-05-08; major-revision update: 2026-05-08T20:23 ET.)

---

## 2026-05-09 — V1 cluster+dedup gate ship (per Peng directive 2026-05-08T22:01 ET)

**What shipped:**
1. `tools/cluster_failures.py` — `single-manual <case_id>` mode + `from-sweep` mode (cluster types: `numeric`, `graph-break`, `fallback`). Audit invariant pinned in 5 tests.
2. `tools/dedup_search.py` — surfaces ANY title/label match with `decision: needs_peng_review`. NO auto-thresholds (per Peng directive). 8 tests.
3. `tools/file_issues.py` — `--cluster-plan-approved <sha256>` is `argparse required=True` on `corpus-issue`. Validator (3 conditions: token == file sha, peng_approval.approved_at non-null, case_id in plan's affected_cases). 5 new tests + 16 updated existing tests.
4. `subagents/file-issue/SKILL.md` — Step 0 documented (cluster→dedup→approve→per-cluster pipeline). Authority gate table updated. "What enforces this" section now lists 4 layers (added the per-batch gate).
5. `subagents/file-issue/persona.md` — Mode A check 10 (cluster cohesion). Activates when draft has `cluster_id` field; verifies the representative_case's MRE applies to the cluster's claim using sweep evidence already in the plan.
6. `subagents/file-issue/cluster-plans/` — first plan written from real NGB D1 sweep data (worked example).

**Smoke + tests:**
- 21 file_issues tests pass (16 existing updated + 5 new V1)
- 5 cluster_failures tests pass (audit invariant; magnitude-bucket fix; D1-row counting fix)
- 8 dedup_search tests pass (keyword extraction; needs_peng_review marking)
- 26 doc-consistency tests pass; 9 live rules clean
- End-to-end smoke: 3 negative-path attempts (tampered body / bad token / case_id-not-in-plan) all refused at the gate as expected

**Two design decisions worth preserving:**

1. **Token semantics simplified mid-build.** First draft of `_validate_cluster_plan` required `peng_approval.token == sha256(plan)`. That's a self-referential constraint (writing the token field changes the file's sha). Fixed by changing the rule to: `cli_token == sha256(plan)` AND `peng_approval.approved_at is non-null`. The cryptographic content-binding is in the sha; the approval marker is just a non-null timestamp recording the event. Cleaner, no convergence dance.

2. **Magnitude bucket dropped from numeric clusterer.** Smoke test on real NGB D1 data revealed (family + mode + magnitude) split the audio-encoder family into 7+2 by magnitude — but the 9 audio models share the SAME root cause (NGB feature × audio encoder forward). Magnitude is now metadata in `root_signal.magnitude_range` (informational); only `(family, mode)` are clustering keys. Pinned in `test_numeric_clusterer_does_not_split_audio_family_by_magnitude`.

**Honest gap (not a defect, just an audit-chain note):** the V1 design went through 2 adversary-review iterations (verbal, in-session) BEFORE landing. The verbatim adversary outputs were not captured to `subagents/adversary-review/invocations/` at write-time, so the cluster+dedup design lacks the standard adversary case files. Future practice: when running adversary review on a multi-round design, write the case file BEFORE iterating, even if the early case files are short — the chain is what matters.

**Open Phase 2 candidates** (not blocking V1 ship):
- `tools/file_issues.py corpus-issue --comment <issue_num>` (post-comment mode for clusters with `action: comment-on-existing`)
- Mode A `comment-review` + Mode B `comment-assembler` persona variants (different shape from full-issue review)
- `tools/check_back_references.py` daily-brief integration (validates auto-link prevention rule is holding on issue 77)
- `tools/audit_repo_side_orphans.py` (replaces dropped GitHub-side marker scan)
- Auto-decision thresholds for dedup_search candidates — only design after enough real surfaces give us example data (per Peng "we will figure out a better system by examples")

## 2026-05-13 — 3-invocation cadence trigger (#25 EDIT, #96 EDIT, #27 EDIT REJECTED)

**Invocations since last retrospective:** 3
- `file-2026-05-13-204500-issue25-mre-fullgraph` — EDIT shipped
- `file-2026-05-13-220000-issue96-scope-refresh` — EDIT shipped
- `file-2026-05-13-223000-issue27-uninit-nnmod-note` — EDIT REJECTED at Mode A

**Recurring gap classes Mode A surfaced:**

1. **Unit-conflation in scope numbers** (twice — surfaced on #96 EDIT Mode A FIX 1 and indirectly on #27 number discrepancy). The corpus aggregates break_reasons into "scope numbers" but distinguishing (a) distinct model classes / (b) (model, mode) pair-rows / (c) total break_reasons (including duplicate-suppressed) is critical. Maintainer-meaningful unit is model classes. Encoded as **methodology R12** in `skills/weekly-sweep-brief/methodology.md` (commit `f71232c`); persona check 2 (Symptom validity) implicitly enforces via validation file truth-source.

2. **Soft-graph-break ≠ no-bug misread** (twice in one day — #14 and #25 first-pass). When MRE runs with `backend="eager"` only (no `fullgraph=True`), graph break is logged but compile completes — looks like success. I misread BOTH cases as "bug not reproducing." Encoded as persona Mode A check 2 subsection (commit `ee9033d`) requiring validation files for graph-break-class issues to capture BOTH soft-mode + fullgraph=True.

3. **Step 2.5 verify_repro architectural mismatch** (surfaced on the queued `torch._check closure` NEW filing). Step 2.5 greps stderr for the expected_signal fragment, but the corpus sweep emits per-model graph_break_reasons to JSON files (`explain_results.json`), NOT stderr. Aggregate sweep-driven NEW filings cannot satisfy the gate without a code-fix. NO NEW corpus filing has gone through Step 2.5 end-to-end. Three resolution paths (a/b/c) surfaced to Peng; awaiting decision. New WS2 task in PLAN.md.

4. **Check 13 HARD BLOCK on multiple GB reasons in one body — tension** (surfaced on #27 EDIT REJECTED today). Rule was written to prevent grouping distinct breaks that force maintainer triage. The #27 EDIT was the *inverse* — observational cross-reference helping a maintainer searching "Uninitialized nn.Module" land at the same-root-cause primary issue. Mode A correctly REJECTED per the strict text-presence rule, and surfaced the tension to Peng for path decision (strict + separate cross-ref issue, OR amend check 13 with downstream-in-same-trace carve-out).

**What's working well:**

- **Mode A as adversary-with-teeth pays off.** #27 EDIT would have shipped subtly-wrong content (number discrepancy + check-13 violation + dedup-not-run) without the Mode A pass. The Agent-spawn cost (one cycle) is tiny vs the cost of a contested or wrong issue body.
- **Inlining Mode B for mechanical EDITs is fine.** #96 was a 45-word footer addendum + count refresh on a 700-word body; spawning a fresh Mode B Agent for that would have been disproportionate. Mode A's FIX list was clear enough to apply directly. Decision recorded in case file `mode_b_sha256: inline-otter-direct`.
- **Surfacing rule tensions per persona NOTES guidance.** Mode A explicitly surfaced the check-13 tension with two paths instead of silently relaxing. This is the iteration loop the persona was designed for.

**Persona amendments shipped this cycle:**

- Check 15: dynamo↔dyn-shape exclusivity (commit `c9f40b7`) — 4-bucket label classifier with explicit signal vocabularies
- Check 16: capture-scalar-output auto-suggest for `.item()` patterns (commit `d762e81`) — 2-MRE comparison required
- Check 2 subsection: soft-graph-break ≠ no-bug discipline (commit `ee9033d`)
- methodology R12: distinguish (model, mode) pair-rows from model classes (commit `f71232c`)

**Action items surfaced for future invocations:**

- Reconcile validation-script vs body de-duplication semantics for break_reason counting (the 16-vs-8 discrepancy from #27)
- Resolve Step 2.5 architectural gap (Peng's path choice pending)
- Resolve check-13 carve-out question (Peng's path choice pending)
- Consider Mode A check 17 (numbering TBD): when a draft enumerates >1 distinct break_reason text but the second is ATTRIBUTED to the first as a downstream observation in the same trace, allow proceed-with-fixes (with strict shape requirements) rather than HARD REJECT. Defer encoding until Peng approves the carve-out.
