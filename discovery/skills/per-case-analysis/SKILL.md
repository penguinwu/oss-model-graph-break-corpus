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

Plus per-trial: `files_touched`, `diff_lines`, `turns`, `agent_claim` (one phrase).

**Output:** `<exp>/reports/<case_id>/fingerprints.csv` with one row per trial.

**Tip:** delegate Phase A to parallel subagents (one batch per arm) — speeds up the read-and-classify work without compromising consistency. Give each subagent the same prompt template; merge resulting CSVs.

### Phase B — Aggregations

From the fingerprint CSV + result.json data, compute:

- `fix_status × variant × skill` 3D table
- `fix_locus × fix_status` correlation
- Performance distribution per fix_status (median, range, tier-1 + tier-2)
- Per-arm aggregates: median speedup, median elapsed, median turns
- Strategy-cluster identification (group by (fix_locus, escape_hatch))
- Break-shape histogram (across all trials)
- Per-variant fix_status within each skill arm

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

### Phase F — Writeup (PR-FIRST, NEVER direct-to-main)

Compose the findings doc at `<exp>/reports/<case_id>/findings.md` with structure:

```
# <case> — Findings (N-trial cross-case skill discovery)
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

**Delivery — non-negotiable workflow:**

1. Create branch `review/<case_id>-findings` from current main.
2. Add ONLY the findings doc + fingerprints.csv on the branch — do NOT bundle workflow scaffolding, methodology improvements, or other commits with the analysis.
3. Push branch + open PR. PR description = TL;DR + headlines + workflow notes + link to per-case issue.
4. Comment on the per-case issue linking the PR.
5. **Wait for Peng's review.** Do not merge without sign-off. Do not commit findings to main as a workaround.
6. After review + comments addressed, Peng merges (or closes if no merge needed). The per-case issue then moves to Done.

**Anti-pattern — bundled megacommit:** in the Mistral3 Case 3a session (2026-04-25), findings were bundled with workflow scaffolding in commit ca3092c and pushed direct to main. PR #65 was opened retroactively as a review surface — but the content was already on main, defeating the gate purpose. **Don't do this.** Each case's analysis output is its own PR.

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
