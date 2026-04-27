---
name: per-case-analysis
disable-model-invocation: true
description: 7-phase blueprint for analyzing a per-case discovery run. Starts with data-trustworthiness audit (Phase 0), then per-trial fingerprint classification, aggregations, question-by-question walk, surprise inventory, cross-case implications, writeup. Use after any per-case run completes.

---

# per-case-analysis

## What this skill is

A reusable blueprint for analyzing a single completed discovery case (e.g. Mistral3, VitsModel, Aria, PaddleOCRVL, future cases). Produces a per-case findings document at `discovery/experiments/<exp>/reports/<case_id>/findings.md` plus a per-trial fingerprint CSV.

Same structure for every case → cross-case synthesis becomes mechanical.

## When to invoke

After any per-case discovery run completes (24 trials × matrix). The run produces a `run_dir` at `/tmp/discovery-runs/<case_id>/<run_id>/` with one subdir per trial. This skill walks that data and produces the analysis.

## The 7 phases

### Phase 0 — Data trustworthiness audit (BLOCKING gate)

**Run BEFORE any other phase.** Establishes that the data is sound enough to draw conclusions from. If Phase 0 surfaces serious issues, fix them or document the caveat — do NOT proceed to analysis on suspect data.

Checks to run:

1. *Artifact completeness* — every trial dir has `agent_diff.patch`, `result.json`, `prompt.txt`, `validation_*.log`, `perf_*.log`, `stream.jsonl`, `claude_stderr.log`. Zero-length stderr is OK (no warnings); zero-length result/diff/stream is NOT.
2. *.original backup integrity* — verify against pristine source (e.g. `pip download` the package, diff the file). Critical: prior post-hoc tools (revalidate, etc.) MUST NOT have mutated `.original` files.
3. *Trial completion sanity* — runner anomalous flags surveyed. Serious ones: `validate-crashed`, `perf-parse-error`, `watched-file-missing`. Benign ones: `file-mutated:*` (expected when agent edited).
4. *Internal consistency* — result.json fields match summary.json for each trial; gb counts in agent's stream-final-summary match validate's reading (within fix_status-explained divergence); validation_v2 (if present) agrees with legacy validation on `gb_under_canonical_inputs`.
5. *Reasonable value bounds* — speedup ∈ (0, 100); max_diff ∈ [0, 1); elapsed_s ∈ (60s, timeout+5min).
6. *Stream integrity* — every `stream.jsonl` parses cleanly to last line; not truncated.

**Output:** a Phase 0 audit log alongside the findings doc. If any check failed, document the recovery action and any caveat to the analysis.

### Phase A — Per-trial fingerprint classification

For each of N trials, read `agent_diff.patch` + `result.json` + the FINAL `result` event in `stream.jsonl` (only the final summary, the full stream is huge). Classify on:

- *fix-locus:* `model-only` | `setup-only` | `both`
- *fix-shape-family:* `deletion` | `restructure` | `wrap` | `config-flag-flip` | `escape-hatch` | `input-type-tweak` | `mixed` | `other`
- *op-order-preserved:* `yes` | `no` | `unclear`
- *escape-hatch-used:* list (custom_op, disable, cond, allow_in_graph, nonstrict_trace, leaf_function, is_compiling, or `none`)
- *break-shapes-attacked:* semicolon-separated list
- *semantic-equivalence:* `bit-equivalent` | `math-equivalent` | `context-equivalent` | `lossy` | `unclear` (worst equivalence in the diff; values below `math-equivalent` flag escape-hatch candidates — see fingerprint_schema.md for definitions)

Plus per-trial: `files_touched`, `diff_lines`, `turns`, `agent_claim` (one phrase).

**Output:** `<exp>/reports/<case_id>/fingerprints.csv` with one row per trial.

**Tip:** delegate Phase A to parallel subagents (one batch per arm) — speeds up the read-and-classify work without compromising consistency. Give each subagent the same prompt template; merge resulting CSVs.

### Phase B — Aggregations

**Pre-aggregation: declared-fallback classifier (per design.md §4.7).** Before computing any aggregates, tag each trial L/M/S/R based on whether the agent's declarations match the diff:

- For each declared override (in stream-final-summary or `# DECLARED-OVERRIDE: <mech> — <reason>` comments in the diff), check the diff for the named mechanism.
- *L (Legitimate)* — declaration present, diff matches, mechanism is NOT on the measurement-affecting list.
- *M (Measurement-affecting)* — declaration present, diff matches, mechanism IS on the measurement-affecting list. Current measurement-affecting mechanisms: `backend="eager"` (swaps the entire compile path — no codegen, no fused kernels, no Inductor RNG-order divergence — so accuracy + perf axes are not comparable with the rest of the variant matrix; per design.md v0.6 §4.7).
- *S (Shortcut)* — forbidden mechanism in the diff with no matching declaration.
- *R (Refused)* — no fix attempted, agent declared infeasibility cleanly.

Forbidden-mechanism set (default; per-variant overrides apply): non-default backends; capture flags (`capture_scalar_outputs`, `capture_dynamic_output_shape_ops`); escape hatches (`custom_op`, `disable`, `cond`, `allow_in_graph`, `nonstrict_trace`, `leaf_function`, `is_compiling`); `torch._dynamo.config` mutations not covered by capture-flag list.

Add an `lmsr_tag` column to `fingerprints.csv`. M-tagged trials are reported separately in all aggregations below — their numbers do NOT pool with L-tagged trials.

From the fingerprint CSV + result.json data, compute:

- `fix_status × variant × skill` 3D table
- `fix_locus × fix_status` correlation
- Performance distribution per fix_status (median, range, tier-1 + tier-2)
- Per-arm aggregates: median speedup, median elapsed, median turns
- Strategy-cluster identification (group by (fix_locus, escape_hatch))
- Break-shape histogram (across all trials)
- Per-variant fix_status within each skill arm
- `semantic_equivalence` distribution (count of trials per equivalence level). For any trial below `math-equivalent`, identify the specific transformation(s) in the diff that caused the demotion. Aggregate recurring non-equivalent transformations across the case as **escape-hatch candidates** — produce a small table in the findings doc (`Escape hatch candidates` section) listing transformation, trial count, why it's context-equivalent (not stronger), and a one-line escape-hatch idea. Per-case escape-hatch candidates feed cross-case Q7 synthesis: candidates that recur across cases are the highest-value PyTorch RFC signals.

**Conditional V4/V6 trigger check (gate before Phase C if standard matrix only ran V0+V2 per master plan):**

- *V4 trigger:* any V0 or V2 trial has `escape_hatches` containing a canonical hatch (`custom_op`, `disable`, `cond`, `allow_in_graph`, `nonstrict_trace`, `leaf_function`) or `is_compiling`. → Document the trigger in Phase B output, then return to Phase 6 of the per-case execution flow (master plan) and launch `--variants V4`.
- *V6 trigger:* any V0 or V2 trial flipped a `torch._dynamo.config` flag (visible in the diff, e.g. `capture_scalar_outputs=True`). → Same: document and queue `--variants V6`.
- *No triggers:* document "no conditional follow-ups warranted — V0/V2 surfaced no canonical escape hatches and no config flips" in Phase B and continue to Phase C with the 12 trials.

After conditional follow-ups complete, **re-run Phase A on the new trials**, append to `fingerprints.csv`, and re-run Phase B over the full set. Do NOT skip phases for follow-up trials.

**Output:** numerical tables in the findings doc.

### Phase C — Question-by-question walk

The discovery experiment has pre-registered open questions (Q1–QN in the master plan). Walk each systematically with evidence from Phase A/B:

- For each question, state the answer in one paragraph + supporting numbers
- Be honest about ambiguous answers; don't force a verdict
- Distinguish observation (raw data fact) from inference (interpretation)

**Output:** a "Q1–QN walk" section in the findings doc.

### Phase D — Surprise inventory

Document everything in the data that diverged from what you'd have predicted before the run. Each surprise is a discovery signal — the kind of finding that wouldn't show up in a pre-registered hypothesis test.

Examples of valid surprises:
- A cell that produced uniform results when you expected variance
- A cell that produced variance when you expected uniformity
- A constraint that didn't constrain
- A skill that didn't help
- A pattern that crosses cells in an unexpected direction

DO NOT manufacture surprises. If everything was as expected, write "no surprises" — that itself is signal.

### Phase E — Open observations + cross-case implications

Two parts:

1. *Open observations:* data points the analysis surfaces but doesn't explain. Honest "unexplained" beats fabricated story (see `feedback_overfitting_explanations.md` if applicable).
2. *Cross-case implications:* what should the next case in the experiment series look for? What axes should be added to the fingerprint? What constraints proved (un)useful? What methodology gaps did this case expose?

### Phase F — Writeup (commit to main + summary on the per-case issue)

Compose the findings doc at `<exp>/reports/<case_id>/findings.md` with structure:

```
# <case> — Findings (N-trial cross-case skill discovery)
## Setup (1 paragraph + links — see below)
## TL;DR (3-7 bullets, the headline)
## The data (aggregation tables)
## Phase C — Q1–QN walk
## Phase D — Surprises
## Open observations (Phase E.1)
## Cross-case implications (Phase E.2)
## Recommendations
## Methodology notes
## Appendix: reference to fingerprints.csv
```

Include: headline metrics, all aggregation tables, per-question evidence, surprises with the divergence-from-expectation noted, recommendations both for skill curation and harness changes.

**The Setup section — MUST be the first section, before TL;DR.** Lets an outside reader land on this doc and orient without grepping the repo. Required content:

```markdown
## Setup

This is the **<case_id>** case (<Model class>) inside the [Cross-Case Skill Discovery experiment](../../plan.md) (umbrella issue #60). Per-case issue: #<NN>. The experiment asks: *when the `debug-graph-breaks` skill is loaded into the discovery agent, does the agent's reasoning, fix-space, or fix-shape change vs. bare Claude?*

**Matrix at a glance** — N trials = (skill arms) × (variants) × (replicates).

| Axis | Values |
|---|---|
| Skill arm | `none` (bare Claude), `debug-graph-breaks` (Arsh Zahed's fork) |
| Variant | V0 (bare prompt), V2 (bitwise equivalence required), V4 (no escape hatches), V6 (no config flags) |
| Replicates | 3 per cell |

**Where data lives.** Per-trial fingerprints in [`fingerprints.csv`](fingerprints.csv) (one row per trial). Raw artifacts (`agent_diff.patch`, `result.json`, `stream.jsonl`) in `/tmp/discovery-runs/<case_id>/<run_id>/`. Methodology details in the [master plan](../../plan.md). Phase 0 audit log alongside this doc as `phase0_audit.md` (if present).
```

Substitute case_id, Model class, and the per-case issue number. If a variant axis is dropped (e.g., V0+V2 only — see master plan), reflect that in the matrix table.

**Delivery workflow (current as of 2026-04-25):**

1. Commit the findings doc + fingerprints.csv to main directly. Methodology / scaffolding changes can ride in the same commit or a separate one — Peng's preference is "ship as a coherent unit."
2. Push to main.
3. Post a headline summary as a comment on the per-case issue. Body should be enough that Peng can read the comment alone and know what's in the report. Link the file path on main.
4. Per-case issue moves to Done.

**Why no PR-FIRST:** earlier sessions encoded a PR-FIRST workflow for analysis output. Peng discontinued it on 2026-04-25 — PR diffs are hard to read for analysis docs (most of the value is the prose, not the line-level changes), and feature branches accumulated merge conflicts when methodology landed on main during a review cycle. The workflow now is direct-to-main + issue comment for the headline.

## Anti-patterns to avoid (per `feedback_overfitting_explanations.md`)

- **Don't invent mechanisms.** If a pattern is unexplained, write "unexplained" and stop.
- **Don't rush to a headline.** The analysis IS the experiment's purpose. Walk the data first.
- **Don't conflate "every trial succeeded" with "success rate is the headline."** When success rate is high (every trial succeeded), the strategy-cluster and perf axes become the headline.
- **Distinguish observation from inference at every step.** "X happened" is observation; "X happened because Y" is inference.

## Reusability across cases

Same 7 phases apply to every case in the experiment series. The `fingerprints.csv` schema is fixed; the question list (Q1–QN) is the same per master plan. Cross-case synthesis (item Q7 in the master plan) is mechanical: stack the per-case findings docs + fingerprint CSVs and look for patterns.

## Approximate cost in agent time

- Phase 0: 1 cycle (mostly mechanical script + cross-verify with pip)
- Phase A: 1 cycle if delegated to N subagents in parallel; 3-5 cycles if serial
- Phase B: 1 cycle (aggregation script)
- Phase C-E: 2-4 cycles (the actual thinking work)
- Phase F: 1 cycle (writeup)

Total: ~half a session per case if parallelized.
