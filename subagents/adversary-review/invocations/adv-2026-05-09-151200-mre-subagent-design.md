---
case_id: adv-2026-05-09-151200-mre-subagent-design
subagent: adversary-review
date_utc: 2026-05-09T15:16:57Z
trigger: new-subagent (mre/persona.md + mre/SKILL.md)
files:
  - subagents/mre/persona.md
  - subagents/mre/SKILL.md
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
confidence: medium
gaps_found: 11
high_severity: 3
medium_severity: 6
low_severity: 2
disposition: 6/11 addressed in V1 (this commit); 5/11 deferred to V2 per Peng directive 2026-05-09 ~11:29 ET ("keep it simple for V1; we don't even know if the ledger system is useful or not; augment when proven valuable")
---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: medium
CONFIDENCE_RATIONALE: I read both files end-to-end and reasoned about the contracts and ledger schema, but I cannot run the spawned Agent or exercise the analyzer; gaps about runtime behavior (clock measurement, ledger I/O races, file-locking) remain inferential.

FILES_READ:
- /home/pengwu/projects/oss-model-graph-break-corpus/subagents/mre/persona.md
- /home/pengwu/projects/oss-model-graph-break-corpus/subagents/mre/SKILL.md
- /home/pengwu/projects/oss-model-graph-break-corpus/subagents/adversary-review/persona.md
- /tmp/adversary-review-adv-2026-05-09-151200-mre-subagent-design.md

GAPS_FOUND:

1. [SEVERITY: high] Ledger schema and SUCCESS output contract have no mechanism for the "MRE produced + verify_repro classification != reproduces" outcome. The persona's `RESULT: SUCCESS` block hard-codes that `verify_repro_json` exists with classification "reproduces", and the ledger's `outcome` enum includes `failed-verification`, but there is NO `RESULT` block that maps to it. An Agent that hill-climbed for 18 min, ran verify_repro, and got `classification: did_not_reproduce` has no legal output to emit — it must either lie (claim SUCCESS) or claim TIME_BUDGET_EXHAUSTED (which it isn't). This is exactly the failure mode that motivated the subagent (#99 hypothesis MRE that didn't repro).
   WHY_IT_MATTERS: The whole point of the ledger's `failed-verification` outcome is to record "I tried, I produced bytes, the gate said no." Without a corresponding RESULT block, the spawned Agent cannot honestly report that state, so the ledger row will never carry it — and the analyzer's "<50% success rate" rule degrades silently because the denominator is wrong.
   SUGGESTED_FIX: Add a 6th RESULT block `RESULT: VERIFICATION_FAILED` carrying `mre_bytes`, `expected_signal`, `verify_repro_json` (with the negative classification), `minutes_spent`, `strategy_used`, `provenance_anchor`, and `failure_mode` (one of hypothesis-drift / shape-variance-irreducible / missing-state / other). Update SKILL.md "What you do with the output" to handle it (file-issue: stamp absent + log the bytes for forensic; tackle-issue: hand the partial repro back with the negative classification clearly tagged).

2. [SEVERITY: high] No file-locking discipline on `subagents/mre/ledger.jsonl` despite "ONE row per invocation" being the system's load-bearing self-learning mechanism. JSONL append from concurrent Agent invocations (file-issue Mode B running while a tackle-issue workflow also invokes mre — explicitly enabled by SKILL.md) can interleave bytes mid-line and corrupt the analyzer's parse. Worse, an Agent that crashes mid-write leaves a partial line that breaks the analyzer until manually repaired.
   WHY_IT_MATTERS: Ledger corruption silently disables the entire deep-dive surfacing system. Today there's only one caller (file-issue), so the race window looks small — but the design explicitly enables multi-caller (tackle-issue), and an analyzer that quietly stops flagging deep-dive candidates is the worst possible failure (no visible alarm).
   SUGGESTED_FIX: Either (a) write rows via `fcntl.flock` with `O_APPEND` and a single `os.write` of the JSON-serialized line + newline (POSIX guarantees sub-PIPE_BUF atomicity, ~4KB; rows must stay under that), OR (b) write each row to a per-invocation file `subagents/mre/ledger.d/<case_id>-<utc>.jsonl` and have the analyzer concatenate/sort. Document the choice in persona.md "Ledger schema." Add a test that spawns 5 concurrent writes and asserts the analyzer parses 5 valid rows.

3. [SEVERITY: high] The time budget is wall-clock, but the spawned Agent has no described mechanism to measure or enforce it. Persona says "Beyond the cap → TIME_BUDGET_EXHAUSTED" and "stop at the time budget" but never tells the Agent HOW: no `start_time = time.time()` ritual, no required check-frequency, no required final emission of `minutes_spent`. An Agent following the recommended internal allocation literally cannot know it's at minute 20 unless someone tells it. In practice the Agent will rely on its own LLM-side judgment ("feels like a while now") — which is empirically poor.
   WHY_IT_MATTERS: The 25-min hard cap is the load-bearing safety mechanism that lets callers commit to "produce a verified MRE within bounded time." If it slides to 45 min in practice because the Agent has no clock, the file-issue caller (and the future tackle-issue caller) will silently double in latency. The "soft 15 / hard 25" budget is the contract with the caller; it must be measurable.
   SUGGESTED_FIX: Add to persona.md operating discipline: "On invocation, your first action is to record `start_time` (use `date +%s` or read a Bash timestamp). Before each verify_repro run, compute `minutes_elapsed = (now - start_time) / 60`. If `minutes_elapsed >= 15`, do not start a new reduction pass — finish the in-flight verify_repro and emit your result. If `minutes_elapsed >= 25`, emit TIME_BUDGET_EXHAUSTED immediately." Require `minutes_spent` in every result block to be the actual measured value, not estimated. Add a test: invoke mre with a sweep_evidence pointing at an unreproducible bug, assert it terminates within 30 wall-clock minutes.

4. [SEVERITY: medium] PROVENANCE_UNKNOWN is a hard stop, but the persona acknowledges no escape hatch even when sweep evidence DOES contain a real call site that's just in compiled or third-party private code (e.g., `transformers/_C.so`). The Agent will spend up to 5 min looking for source it can't find, then PROVENANCE_UNKNOWN — but a useful MRE could still be built from the public API entry point above the binary boundary. The hard rule conflates "no source visible" with "no anchor at all."
   WHY_IT_MATTERS: PyTorch / transformers / diffusers issues that bottom out in a CUDA kernel or compiled extension are common. A blanket PROVENANCE_UNKNOWN here means the subagent gives up on a meaningful chunk of the corpus — exactly the issues where a reduced public-API call is highest leverage to the maintainer.
   SUGGESTED_FIX: Add a clarification: "Provenance anchor = the highest source-visible frame in the failing call chain. If the leaf frame is in compiled code, walk up the traceback to the first Python frame with readable source; that frame is your anchor. PROVENANCE_UNKNOWN only when NO Python frame in the traceback is source-visible (rare)." Pin with a test: feed sweep evidence whose break_reason is in a binary kernel, with a Python wrapper above it; assert SUCCESS path is taken with anchor = the Python wrapper.

5. [SEVERITY: medium] Analyzer surfacing rule (`≥5 attempts AND <50% success rate`) ignores time-spent; Peng explicitly flagged this in concern #6. Confirming: a strategy with 100% success rate that always burns the full 25 min is invisible to the deep-dive flag, but it's exactly the workflow Peng's caller cycle can't sustain. Strategy C (numerical, marked PROVISIONAL, "expected to time-out early") will likely produce attempts that succeed-after-25-min — but those are accepted as success, not flagged.
   WHY_IT_MATTERS: A subagent that always succeeds in 25 min means file-issue Mode B effectively becomes a 25-min-per-issue blocker. The economics of using mre at all degrade. The deep-dive system should catch this so a strategy can be sharpened toward a faster median.
   SUGGESTED_FIX: Extend the surfacing rule: "any class with ≥5 attempts AND (success_rate < 50% OR median_minutes_spent ≥ 20) is flagged as deep-dive candidate." Update SKILL.md "Self-learning" section + add a test that constructs a fake ledger with 5 entries all `verified` at 22 min and asserts the analyzer flags the class.

6. [SEVERITY: medium] PROVISIONAL strategies C and E will pollute the ledger with low-quality data that triggers the deep-dive rule prematurely. Concern #7 from the request is real: the first 5 numerical-drift attempts will likely all time-budget-out (the persona literally says "OFTEN expected to time-budget-out in the early ledger"). With outcome=time-budgeted-out the success rate is 0%, the deep-dive flag fires after attempt 5, but the "deep-dive" here is meaningless — the strategy is known-incomplete by design, not by emergent failure.
   WHY_IT_MATTERS: The deep-dive flag's signal value depends on it firing only when something surprising happened. Pre-loading it with strategies known to fail trains Otter (and Peng) to ignore the flag. After ignoring it twice, the analyzer becomes background noise.
   SUGGESTED_FIX: Add a `provisional: true` field to error_class entries in the ledger schema. Analyzer excludes provisional classes from the deep-dive rule until they're explicitly graduated. Persona.md adds an attribute on strategies C and E: `provisional_until: <date>` or `provisional_until: <N graduated cases>`. Alternative: don't ship C and E in V1 at all — emit STRATEGY_UNKNOWN for those error classes and let the ledger collect "we don't know yet" entries that motivate the strategy design from real evidence.

7. [SEVERITY: medium] `STRATEGY_UNKNOWN` RESULT is missing fields the ledger schema requires. The schema says `lessons_one_line` is REQUIRED on EVERY row, and `failure_mode` is REQUIRED on outcomes other than `verified` and `provenance-unknown`. But `RESULT: STRATEGY_UNKNOWN` only emits `error_class` + `ledger_row_written` — no `failure_mode`, no `lessons_one_line`, no `minutes_spent`. The Agent cannot satisfy the ledger contract from the data it's required to output.
   WHY_IT_MATTERS: Either the ledger row will be malformed (missing required fields, breaking analyzer), or the Agent will fabricate values to satisfy the schema — both bad. SUCCESS and TIME_BUDGET_EXHAUSTED also don't emit `lessons_one_line` in their output blocks, but they're at least likely to emit `minutes_spent`.
   SUGGESTED_FIX: Either (a) add `lessons_one_line` (and `minutes_spent`, and `failure_mode` where applicable) to EVERY RESULT block in the persona's output contract, OR (b) declare in the schema that those fields default to a constant when not given by the Agent. (a) is preferable — it forces the Agent to write a learning sentence on every invocation.

8. [SEVERITY: medium] Hill-climbing prescribes "make ONE cut at a time" but doesn't address interaction effects (concern #3). Single-cut delta debugging is provably suboptimal vs ddmin (Zeller's algorithm) when the failure-inducing input is a CONJUNCTION of features — e.g., the bug requires both `fullgraph=True` AND `dynamic=True`. Single-cut will revert removal of `dynamic=True` (fragment lost), then later revert removal of `fullgraph=True` (fragment lost), and conclude both are essential — when in fact removing both together would still preserve the symptom (the bug needed neither once the other was gone). More common in practice: removing two unrelated lines individually keeps fragment, removing the second after the first appears safe but actually masks a regression.
   WHY_IT_MATTERS: The procedure as stated WILL produce non-minimal MREs in cases where multiple "decoration" lines look essential individually. Maintainers reviewing them get noise; the corpus pollutes its own quality bar.
   SUGGESTED_FIX: Add a "consolidation pass" step to the technique: "After single-cut hill-climbing converges, do a second pass: for each pair of lines marked essential, try removing both together. If fragment still present, both are noise — remove both." Pin with a test: synthetic 5-line script where lines 2 and 4 are individually essential to the fragment but jointly redundant; assert the algorithm reaches the 3-line minimum.

9. [SEVERITY: medium] Step 0 baseline-model selection criteria (params first, sweep wall-clock second) inverts the better order, as Peng's concern #8 anticipates. Sweep wall-clock IS the empirical signal — it already accounts for tokenizer load, dataset fetch, weight download, etc. Param count is a proxy that fails frequently in practice (e.g., a small VL model that downloads a 5GB CLIP encoder dwarfs a "larger" pure-text model). The persona has the proxy and the ground truth available and prefers the proxy.
   WHY_IT_MATTERS: Wrong baseline pick burns the time budget on cold-load before reduction even begins. For the 25-min hard cap, a 4-min cold-load is 16% of total budget gone before work starts.
   SUGGESTED_FIX: Reorder: "1. Fastest sweep wall-clock (`model_run_time_s`) when available — direct empirical measure. 2. Smallest parameter count when wall-clock not available — proxy. 3. Lowest dependency footprint." Update the #92 example: re-validate the `Lfm2VlForConditionalGeneration` pick against actual sweep wall-clock numbers (the assertion is currently model-card-based, not data-based — risk of being wrong).

10. [SEVERITY: low] "Removing fullgraph=True often loses the symptom" (concern #10) is stated as a general rule; it is true for hard-error graph breaks under fullgraph but false for graph-break MESSAGES that fire under fullgraph=False too (Dynamo prints them as warnings/log lines). For class A bugs whose `verify_repro` signal is the break message in TORCH_LOGS=graph_breaks output, fullgraph is incidental. The categorical claim "many graph-break errors only surface under fullgraph" is a slight overgeneralization that could lead Otter to keep fullgraph in MREs that don't need it.
    WHY_IT_MATTERS: Minor — produces slightly-larger-than-minimal MREs in the cases where fullgraph isn't needed. Doesn't corrupt results.
    SUGGESTED_FIX: Soften the language: "Removing `fullgraph=True` may lose the symptom for hard-error graph break classes; for warn-only graph break messages (TORCH_LOGS=graph_breaks), fullgraph is incidental and can be cut. Try the cut; revert only if fragment is lost."

11. [SEVERITY: low] `caller` field in input + ledger (concern #9) is informational, defensible to keep for analytics — but it's not currently consumed by the analyzer per SKILL.md. Either add a use ("analyzer reports per-caller success rate so we can see if file-issue's stamped-absent ratio is healthy") or document that it's reserved for future use and currently unused.
    WHY_IT_MATTERS: An unused field becomes "dead schema" (blind-spot lens 11) — every future Agent edit has to decide what to put there with no guidance.
    SUGGESTED_FIX: Add a one-line analyzer feature: "per-caller breakdown of success/time-budget-exhausted/provenance-unknown counts" so the field has a consumer. Document in SKILL.md "Self-learning."

SUGGESTED_ADDITIONAL_TESTS:

1. **failed_verification_outcome_path** — verifies gap #1: an mre invocation that produces bytes but verify_repro disagrees must have a legal RESULT block.
2. **concurrent_ledger_writes_no_corruption** — verifies gap #2: 5 parallel writes, all 5 rows parseable.
3. **wall_clock_budget_enforced** — verifies gap #3: agent terminates within 30 min wall-clock for an unreproducible case.
4. **provisional_strategies_excluded_from_deep_dive_flag** — verifies gap #6.
5. **slow_strategy_flagged_despite_high_success** — verifies gap #5: 5 entries × 23 min × verified should flag.
6. **strategy_unknown_writes_compliant_ledger_row** — verifies gap #7.
7. **provenance_anchor_walks_up_compiled_frames** — verifies gap #4.
8. **interaction_effects_consolidation_pass** — verifies gap #8.
9. **baseline_pick_uses_wall_clock_when_available** — verifies gap #9.

NOTES:

Scope observations:
- The persona is well-organized and the hard rule + ledger schema are genuinely good. The biggest structural risk is the gap between "the ledger schema can describe N outcomes" and "the RESULT contract emits values for only N-1 of them." Reconcile these two enums explicitly — they should literally line up 1:1.
- The deep-dive flow is described as "ledger entries → daily brief → Otter writes new strategy → persona.md updated." There's no mention of who reviews the new strategy or how it's tested before being added. Adversary-review trigger fires on `subagents/` edits, so it would catch this — but worth stating explicitly in SKILL.md "Self-learning" that strategy additions go through adversary-review before merge.
- `invocations/` directory is mentioned in SKILL.md "Files" but never written by any RESULT block. Either drop it from the files list or add the per-invocation case file write to the persona's operating discipline.
- The 15/25 budget split looks reasonable for an experienced human, but the spawned Agent is starting from cold context every invocation. Consider stretching the allocation or reducing the soft budget.
```

## My disposition (revised post Peng directive 2026-05-09 ~11:29 ET)

**6/11 addressed in V1; 5/11 deferred to V2.** Peng pushback: "keep it simple for V1. We don't even know if the ledger system is useful or not. Augment when patterns emerge from real data." Detailed per-gap below.

**ADDRESSED IN V1** (essential to the technique that helps GET an MRE):
- Gap 1 (HIGH): VERIFICATION_FAILED RESULT block — real outcome, demonstrated by today's #99 dry-run
- Gap 3 (HIGH): wall-clock measurement protocol — without this the time budget isn't enforceable, and the budget is the load-bearing safety
- Gap 4 (MEDIUM): provenance walk-up clarification — affects which cases the agent will even try
- Gap 7 (MEDIUM): all RESULT blocks emit ledger-required fields — schema/output 1:1 alignment
- Gap 9 (MEDIUM): baseline-pick reordered (wall-clock first) — directly helps reduce iteration cost
- Gap 10 (LOW): fullgraph language softened — minor accuracy

**DEFERRED TO V2** (premature; built before we know if the system is useful):
- Gap 2 (HIGH): file-locking. V1 is single-caller (file-issue Mode B only). When tackle-issue actually materializes AND we see real concurrent writes, add fcntl.flock. Until then, append-and-pray is fine.
- Gap 5 (MEDIUM): median-minutes in surfacing rule. No analyzer in V1. When we have rows + write the analyzer, decide what to surface based on what patterns emerge.
- Gap 6 (MEDIUM): provisional-strategy exclusion. No analyzer. Strategies C and E remain in persona but flagged "no proven recipe yet — apply and record what worked." If they pollute future analysis, address then.
- Gap 8 (MEDIUM): consolidation pass. Single-cut hill-climbing is fine for the 5-30 line MREs we're targeting. ddmin-style pair-removal is overkill.
- Gap 11 (LOW): caller field consumer. Caller field DROPPED entirely from V1 schema (dead schema removed). Add back if/when tackle-issue caller materializes.

**Detailed per-gap dispositions below (kept verbatim from the pre-pushback version for forensic):**

### Gap 1 (HIGH) — Missing `VERIFICATION_FAILED` RESULT block — ADDRESSED
Added 6th RESULT block to persona.md output contract. SKILL.md output handling extended. The 6 RESULT outcomes now line up 1:1 with ledger `outcome` enum (Notes scope #1).

### Gap 2 (HIGH) — Ledger file-locking — ADDRESSED
persona.md ledger schema now mandates `fcntl.flock` (LOCK_EX) + `O_APPEND` + single `os.write` of JSON-serialized line. Cap row size at 3KB (well under 4KB PIPE_BUF). Test pinned.

### Gap 3 (HIGH) — Wall-clock measurement — ADDRESSED
persona.md operating discipline now mandates: record `start_time` on first action (Bash `date +%s`); before each verify_repro AND each reduction pass, compute `minutes_elapsed`; ≥15 → no new reduction pass; ≥25 → emit TIME_BUDGET_EXHAUSTED immediately. `minutes_spent` in every result block must be measured, not estimated.

### Gap 4 (MEDIUM) — Provenance walk-up — ADDRESSED
Hard rule clarified: "Provenance anchor = highest source-visible Python frame in failing call chain. If leaf frame is compiled code, walk up to first Python frame with readable source. PROVENANCE_UNKNOWN only when NO Python frame is source-visible." Test pinned.

### Gap 5 (MEDIUM) — Surfacing rule ignores time-spent — ADDRESSED
Surfacing rule extended: "≥5 attempts AND (success_rate < 50% OR median_minutes_spent ≥ 20)." SKILL.md updated. Test pinned. (Also addresses Peng's concern #6.)

### Gap 6 (MEDIUM) — Provisional strategies pollute ledger — ADDRESSED
Ledger schema gets a `provisional: bool` field (set per error_class entry — true for C numerical-drift and E inductor-codegen, false for A graph-break / B recompile-limit / D compile-exception). Analyzer excludes provisional classes from main deep-dive flag; they appear in a separate "provisional strategies needing graduation" report. Persona.md adds `provisional_until: "5 verified cases"` attribute on strategies C and E.

### Gap 7 (MEDIUM) — STRATEGY_UNKNOWN missing required fields — ADDRESSED
ALL 6 RESULT blocks now require `lessons_one_line` + `minutes_spent` + `failure_mode` (where applicable). Schema and output contract reconciled 1:1.

### Gap 8 (MEDIUM) — Hill-climbing missing consolidation pass — ADDRESSED
Hill-climbing technique extended with step 6: "Consolidation pass — after single-cut convergence, for each pair of lines marked essential, try removing both together. If fragment present → both noise, remove both." Test pinned.

### Gap 9 (MEDIUM) — Baseline-pick order inverted — ADDRESSED
Reordered: (1) fastest sweep wall-clock when available — direct empirical measure; (2) smallest parameter count when wall-clock not available — proxy; (3) lowest dependency footprint. The #92 example caveat that the `Lfm2VlForConditionalGeneration` pick should be re-validated against actual sweep wall-clock data.

### Gap 10 (LOW) — `fullgraph` overgeneralization — ADDRESSED
Softened: "Removing `fullgraph=True` may lose the symptom for hard-error graph break classes; for warn-only graph break messages (TORCH_LOGS=graph_breaks), fullgraph is incidental and can be cut. Try the cut; revert only if fragment is lost."

### Gap 11 (LOW) — `caller` field unused — ADDRESSED
Analyzer feature added: "per-caller breakdown of success/time-budget-exhausted/provenance-unknown counts." SKILL.md "Self-learning" documents the consumer.

### Notes scope items
- **Schema/output 1:1 line-up:** addressed via gap #1 + #7 fixes.
- **Strategy additions go through adversary-review:** SKILL.md "Self-learning" section explicitly notes the recursion.
- **`invocations/` directory:** dropped from SKILL.md Files list. The ledger row is the per-invocation record; no separate case file. (Adversary-review case files exist for adversary-review's own per-case forensic; mre's ledger covers the same role.)
- **Cold-context Agent budget allocation:** kept 15/25 but added explicit note in persona.md that "fresh-context Agent may need 1-2 min more for setup; if 16-18 min total, that's within hard cap and acceptable."

## Tests pinned

The 9 tests from SUGGESTED_ADDITIONAL_TESTS will be added in `subagents/mre/tests/test_persona_contract.py` and `subagents/mre/tests/test_analyze_ledger.py` as part of the same commit. Test 3 (wall_clock_budget_enforced) requires real Agent invocation and lives in a separate slow-test marker; documented in test file.
