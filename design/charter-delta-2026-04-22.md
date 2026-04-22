# Charter Delta — OSS Model Graph Break Corpus

**Date:** 2026-04-22
**Author:** Otter
**Status:** Approved framing; per-artifact edits in flight
**Trigger:** Morning riffing session expanded project scope; need to align artifacts before building

---

## 1. What Expanded Today

Five things shifted the project envelope. Each is externally-driven, not internal scope creep.

1. **Use case #3 (skill eval) — promote to active, but reframe as niche.**
   We discovered Arsh Zahed has built `debug-graph-breaks` (D99943226, Unpublished draft) and a generator script `generate_oss_corpus_evals.py` that consumes our `corpus.json` directly. SkillWatch eval suite is auto-derived from our schema. Arsh has 3 open TODOs that intersect with our work (pegged env, hardcoded local paths, mid-execution agent stops).

   *Framing:* the corpus is **not** the primary skill-eval source — the doc-eval project has the richer Q&A surface for general skill capability. The corpus serves a **specific niche**: clean, isolated, single-model reproducers for testing diagnostic accuracy on individual graph breaks in a controlled environment. Position #3 as a complementary fixture set, naming the niche, not generalizing beyond strengths.

2. **Use case #6 — non-strict tracer soundness validation. Real and active.**
   Trace on input A; run on inputs B, C, D; compare outputs. Validates whether traced graphs generalize across input distributions. Animesh wants this result — there is an active debate between non-strict tracer and Dynamo, and our corpus is well-positioned to produce comparative data. Real consumer = promote per the scope-posture rule.

3. **Compile-quality companion to LLM-Perf — scoping out.**
   `optimum-benchmark` is HF's canonical multi-backend benchmarking framework and powers the LLM-Perf leaderboard. Adopting its input/results schema would make our corpus interoperable with the HF benchmarking ecosystem. Effort and integration surface still being researched. No commitment until cost is known.

4. **Multi-input architecture — required by #6 and useful for #3, #4.**
   Today: one input per model. Needed: multiple input variations per model (for tracer soundness, for skill-eval realism, for benchmarking parity with `optimum-benchmark`).

5. **Skill testing as deliverable; failure modes as outputs.**
   "Anything you discover about using these skills, especially when it doesn't work, is important input for skill-eval design." Failure-mode catalog from piloting Arsh's skill is a deliverable, not a side experiment.

---

## 2. Per-Artifact Deltas

For each artifact: current state → proposed update → rationale → blast radius.

### 2.1 Design Doc (Rev 35 → proposed Rev 36)

**Current state.**
Section 1 lists 5 use cases. #2 is "Fix validation — validate graph break fix skills/tools against known breaks (e.g., Arsh's GraphBreak skill)." This is a one-line mention; no further sections elaborate.

**Proposed update.**
- **Section 1 (Goal).** Reframe skill-eval bullet with the niche framing (clean isolated single-model fixture set; not the primary skill-eval source). Add use case #6 (tracer soundness) with Animesh as named consumer. Mention `optimum-benchmark` schema alignment as a scoping item, not a commitment.
- **Section 3.2 (Input Generation).** Expand from "single input per modality" to "input set per modality." Each model has a default input plus optional variations.
- **New Section 3.7 (Tracer Soundness Mode).** Define the trace-on-A, run-on-B/C/D protocol. Defines a new measurement type and result schema.
- **Section 7 (Future Work).** Add: multi-input infrastructure; `optimum-benchmark` scoping; skill-eval failure-mode catalog.

**Rationale.**
The design doc is the canonical statement of what the corpus IS. Rev 35 doesn't yet reflect what we built (Arsh integration), what we discovered we need (multi-input), or where we're heading. Without this update, contributors and consumers work from a stale charter.

**Blast radius.**
Rev 36. New section is additive. Section 3.2 changes are forward-compatible (existing single-input case = a 1-element input set).

---

### 2.2 USE_CASES.md

**Current state.**
5 use cases. #3 (skill eval) and #4 (skill-as-benchmark) added 2026-04-21, both PROPOSED. No "active integrations" subsection.

**Proposed update.**
- **Promote #3 from PROPOSED to ACTIVE — with the niche framing.**
  - Audience: maintainers of compiler-diagnostic skills who need clean per-pattern fixtures
  - Clear scope statement: "The corpus is not a general skill-eval source. It serves a specific niche — clean, isolated, single-model reproducers tied to known root causes. For broader skill capability evaluation (multi-step reasoning, ambiguous diagnosis), use the doc-eval project's Q&A corpus instead."
  - Add a "Known consumers" bullet listing Arsh's `debug-graph-breaks` pipeline + his open TODOs.
  - Failure-mode-as-deliverable note: skill testing produces signal whether the skill succeeds or fails; failure catalog is the output.
- **Add #6 — Non-strict tracer soundness validation** (ACTIVE, named consumer).
  - Audience: PyTorch export / non-strict tracer team; Animesh as named stakeholder
  - Signal consumed: traced graph + multi-input set + per-input outputs
  - Output format: per-model soundness report (which inputs pass, which diverge, divergence pattern)
  - Code path: TBD — `tools/check_tracer_soundness.py`
- **Stable Signals section.** Add: "Input set" as a forthcoming signal (versioned per-model JSON of input variations).

**Rationale.**
USE_CASES.md is the consumer-facing catalog. Arsh is a real consumer; #3 should reflect that — with honest framing of the niche. #6 has a real audience (Animesh) and is gated on multi-input infra; explicit cataloging forces clarity on what we're building toward.

**Blast radius.**
Single file edit. No code changes implied yet.

---

### 2.3 README.md

**Current state.**
Lists 4 workflows for "compiler developers working on torch.compile." Doesn't mention skill eval, multi-input, or external benchmarking interop.

**Proposed update.**
- **Audience expansion (narrow).** Add: "Skill maintainers needing clean per-pattern fixtures for diagnostic-skill evaluation." Name the niche; don't oversell.
- **Workflow #5: Per-pattern fixtures for skill evaluation** — short paragraph pointing to the eval pipeline (when ready), with a sentence redirecting general skill-eval needs to doc-eval.
- **Defer until built:** multi-input and `optimum-benchmark` mentions. Don't promise what isn't there.

**Rationale.**
README is the front door. New audiences should see themselves in it — but only the niche we actually serve. README should describe what the corpus does today, not aspirations.

**Blast radius.**
Single file edit. Audience addition is the substantive change.

---

### 2.4 generalized-compiler-testing.md

**Current state.**
Designs two orthogonal axes: compile configuration × measurements. Compositional. Today's measurements are `errors`, `compile_timing`, `correctness`, `runtime_perf`.

**Proposed update.**
- **Add a third axis: input variation.** Today implicit (one input per model); proposed explicit (input-set per model, indexed).
- **Add new measurement: `tracer_soundness`.** Specify the trace-once / run-many semantics, the result schema, and the cost.
- **Update test-profile examples** to show a soundness profile.

**Rationale.**
This doc is the natural home for compositional testing primitives. Multi-input is a third axis; tracer soundness is a new measurement that requires it. Adding both here keeps the design coherent.

**Blast radius.**
Doc-only. Implementation deferred to phase plan.

---

### 2.5 AI-Native Maintenance Doc (now Otter-owned)

**Current state.**
Designs the feedback intake → triage → fix loop. Four message classifications: Bug / Feature / Data correction / Question. Authored by Rocky 2026-04-02; Rocky is no longer on the corpus project as of 2026-04-22.

**Proposed update.**
- **Ownership transfer.** Otter takes over the doc. First pass: read carefully and refresh anything stale (the project state has moved since 2026-04-02).
- **Add a fifth classification: "Skill integration request"** (or similar). When a skill maintainer (e.g., Arsh) reports a need that affects the eval pipeline (schema instability, missing field, broken reproducer), it routes differently than a generic feature request — it's a *consumer-driven contract change*.
- **Add a section on consumer SLAs.** If we promise schema stability for skill-eval consumers, what's the change-management process?

**Rationale.**
Active integrations create new failure modes. The maintenance design assumes feedback comes from end users; consumer integrations introduce a different class of feedback (contract violations) that deserves explicit handling. Plus: orphaned doc needs an owner.

**Blast radius.**
Doc edit + ownership change. No external review gate now that Rocky is off-project.

---

### 2.6 Repo Structure

**Current state.**
- `tools/` — single-script consumers
- `sweep/` — sweep infra
- `corpus/` — data
- `correctness/` — Phase 3 outputs
- `skills/` — model-setup, sweep skills
- `analysis/`, `experiments/`, `data/`, `docs/`, `design/`, `results/`

**Proposed update.**
- **Defer creating new top-level directories until needed.** USE_CASES.md says new use cases start as `tools/` scripts; promote to subdirectory only when they outgrow that. Use cases #3, #4, #6 should follow this rule.
- **New scripts to anticipate** (not create now):
  - `tools/export_skill_eval.py` (use case #3)
  - `tools/apply_skill.py` (use case #4)
  - `tools/check_tracer_soundness.py` (use case #6)
  - `tools/export_optimum_benchmark.py` (interop, scoping)

**Rationale.**
Premature directory churn breaks cron jobs and integrations (cited explicitly in USE_CASES.md). Build first, then organize.

**Blast radius.**
None today. Documenting intent only.

---

### 2.7 Artifact Location (new — meta)

**Decision (2026-04-22):** GitHub is canonical for project artifacts going forward. This doc lives at `design/charter-delta-2026-04-22.md` in the repo. Drive becomes a mirror for share-with-others convenience, not the source of truth.

Already encoded in repo CLAUDE.md.

---

## 3. Open Items (post-decision)

1. **Sequencing.** Build first, but capture-what's-built-now in docs. Start with USE_CASES.md edits and design doc Rev 36 capturing the new framing for #3 and #6. Defer aspirational sections until built.

2. **Use case #6 priority.** Real near-term consumer (Animesh). Multi-input MVP comes after we get clarity on what specific soundness signal Animesh wants — don't pre-build the wrong thing.

3. **`optimum-benchmark` schema adoption — scoping.** Otter to research the schema, integration surface, and produce effort estimate (a/b/c with hours). No commitment until cost is known.

4. **Arsh outreach approach.** Run pilot of `debug-graph-breaks` on 3-5 corpus models first (capture what works, what fails, where schema friction shows up). Outreach with findings in hand. Seed in daily standup so Arsh can self-initiate if there's overlap.

5. **AI-native maintenance doc.** Otter takes ownership; reads + refreshes; posts revised version for Peng review. No external review gate.

---

## 4. Sequencing

1. **Today.** USE_CASES.md edits (#3 → ACTIVE with niche framing, #6 → ACTIVE with named consumer, failure-mode note). Move charter delta into repo (this file).
2. **This week.** README audience expansion. Design doc Rev 36 (Section 1 + new Section 3.7).
3. **This week.** AI-native maintenance doc takeover and refresh.
4. **This week.** Arsh skill pilot on 3-5 corpus models. Capture findings.
5. **Next week.** Reach out to Arsh with pilot findings in hand.
6. **Build phase (when greenlit, gated on Animesh signal).** generalized-compiler-testing.md gets the third axis spec. Multi-input MVP. `tools/check_tracer_soundness.py`. `tools/export_skill_eval.py`.

---

## 5. Recommended Decision

Approved 2026-04-22. Land USE_CASES.md and design doc Rev 36 changes this week. Defer multi-input build until #6 has a concrete soundness signal from Animesh. This respects the "infra over depth" phase priority while still capturing the expanded scope in the canonical artifacts.

---

## Appendix A: Mapping of Today's Expansions to Artifacts

| Expansion | Design Doc | USE_CASES.md | README | gen-compiler | AI-native | Repo |
|---|---|---|---|---|---|---|
| Use case #3 → active (niche) | §1 reframe | promote, niche framing, add consumer | new audience (narrow) | — | classification | future tools/ |
| Use case #6 (soundness) | new §3.7 | new entry (named consumer) | defer | new measurement | — | future tools/ |
| Multi-input arch | §3.2 expansion | new signal | defer | new axis | — | sweep changes |
| optimum-benchmark schema | §7 mention | new signal | defer | — | — | future tools/ |
| Failure-mode-as-deliverable | — | #3 note | — | — | — | — |
| Artifact location (GitHub-first) | — | — | — | — | — | repo CLAUDE.md (done) |
