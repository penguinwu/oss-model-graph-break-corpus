---
plan: PT 2.12 graph-break analysis + deep-stack / cascading / accounting investigation
status: active
owner: Otter
created: 2026-05-03
last_check: 2026-05-03
forcing_function: tools/check_plan.py + daily brief
---

# Graph-Break Deep-Dive — PT 2.12 release analysis + accounting validation

> **Origin:** Peng's tomorrow backlog drop on 2026-05-02 22:37 ET (end of session). Three concrete tracks + three investigation questions. Captured here before sleep so tomorrow's session picks up clean.

## Tracks (concrete deliverables)

### Track 1 — PT 2.12 release sweep: detailed analysis with new-model focus

**What:** Produce a detailed analysis of the PT 2.12 release sweep data, with a specific subsection covering graph-break stats for **new models added since PT 2.11**.

**Why:** Standard release analyses report aggregate numbers. Newer models that landed after the prior release are the highest-leverage signal — they're the "fresh ground" for graph-break work and the first place compiler regressions or improvements show up. Treating them as a first-class subsection ensures we don't lose that signal in the aggregate.

**Data:**
- 2.12 baseline: `sweep_results/baseline/pt2.12-2026-04-30/`
- 2.11 baseline: `sweep_results/baseline/pt2.11-fresh-2026-04-29/`
- Diff tool: `tools/compare.py`
- Model identity tracking: need a "new since 2.11" classifier — check `corpus.json` model-list diff between the two baselines, OR use `created_at` if encoded, OR cross-reference HF `since` field if available

**Done means:**
- Analysis doc written, includes section "📊 Aggregate (all models)" + "🆕 New models since PT 2.11"
- For new-models subsection: gb count distribution, per-model breakdown, comparison to fullgraph rate of older corpus models
- Land in PARA + post to PT2 working group / Compile Q&A (Peng-guided framing for community release)

### Track 2 — Train vs eval graph-break asymmetry investigation

**What:** Dig into why training has more graph breaks than inference; produce insight.

**Why:** Standing observation in the corpus that training-mode graph break counts exceed eval-mode counts. Hypothesis directions: (a) backward-graph adds breaks (loss.backward path), (b) train-mode-only ops (dropout, batch-norm running stats) add breaks, (c) some break categories specifically affect autograd, (d) model-specific train-only branches.

**Data:**
- Per-model `eval` vs `train` rows in `sweep_results/baseline/pt2.12-2026-04-30/identify_results.json` + `explain_results.json`
- Compute Δ = train_gb_count - eval_gb_count per model; rank by Δ
- For top-N models by Δ, drill into break_reasons to identify train-only patterns

**Done means:**
- Distribution of Δ characterized (mean / median / tail)
- Top causes of train-only breaks identified by category (filed_issues classifiers + manual inspection of unmatched)
- Insight written up; cross-link to any existing dynamo-team issues (#54 data-dep branching, etc.) where relevant

### Track 3 — "Compiler gives up" / fallback graph breaks (#102 + #103)

**What:** Dig into the two open issues where the graph break is the compiler's fallback rather than a specific user-code pattern:
- **#102** [dynamo] "Failed to handle graph break gracefully" wrapper — 166 models / 718 breaks
- **#103** [dynamo] "Cannot resume from graph break" wrapper — 107 models / 345 breaks
- Related meta-issue **#104** [dynamo] Better error messages + structured fields for wrapper-pattern graph breaks (the systematic ask)

**Why:** These breaks have no informative reason — they're "compiler couldn't handle this gracefully" infrastructure fallbacks. Without specificity we can't bucket them or attribute them to root causes. They're presumably MASKING the real upstream bugs.

**Data + approach:**
- For each issue, cross-reference all affected models from `sweep-report.json`'s `proposed_body`
- For top-N most-affected models, inspect `explain_results.json[model].break_reasons` to see if there's structured info beyond the wrapper text (file:line, surrounding context)
- Identify if these breaks correlate with specific code locations (HF utility files, dynamo internal bailouts, etc.)
- File a sharpening request upstream OR propose a corpus-side rule that buckets these by some derivable secondary signal

**Done means:**
- Each issue (#102, #103) gets an analysis comment with: bucket distribution, hypothesized root causes, repro recipe for the most concentrated cluster
- Decision: fixable in corpus (bucket via secondary signal), or needs upstream dynamo work (file detailed asks against #104), or both

## Investigation questions (research before deciding)

### Q1 — Does nested graph break support change reported counts?

**Hypothesis:** Nested graph break support might reduce the count of breaks that come from "returning from a deep stack." Currently when a graph break happens deep in a call stack, dynamo bails out of the entire frame and reports N breaks instead of 1.

**To answer:**
- Find current state of nested graph break support in dynamo (check torch/_dynamo for `nested_graph_break` / `resume_from_nested` / similar)
- If feature flag exists: run a small subset (5-10 models) with feature on/off, compare gb counts
- Look at how break_reasons differ between flat and nested reporting

### Q2 — Cascading graph breaks from deep-stack origin

**Hypothesis:** Not all graph breaks are equal — a break deep in the call stack may produce many cascading downstream breaks. We should be able to characterize "1 deep break causes 12 cascade breaks" vs "1 isolated leaf break."

**To answer:**
- Inspect `break_reasons` structure — does it include a stack depth or stack trace? (looking at the brief's earlier output, `user_stack_summary` and `user_code_loc` fields exist)
- Build a small analyzer that groups breaks by stack-depth + code-location proximity to identify cascading clusters
- Top-N cascading clusters → file as a single root cause + N derivative breaks rather than N independent

### Q3 — Graph-break count vs subgraph count consistency

**Hypothesis:** `graph_break_count` and `graph_count` should have a derivable relationship — typically `graph_count = graph_break_count + 1` for sequential breaks (each break splits one graph into two). If the relationship doesn't hold, the accounting is wrong somewhere.

**To answer:**
- For each model in `explain_results.json`: compute `graph_count - graph_break_count` (expected: 1)
- Distribution of the delta — anything other than 1 is suspect
- For deltas != 1: drill into break shapes (parallel breaks? bailouts that don't split?)
- Cross-link to Q2 if cascading breaks explain non-1 deltas

## Sequencing for tomorrow

Recommended order (Tier 1 = fresh-mind tasks, Tier 2 = needs cross-checking):

**Morning (Tier 1):**
- Q3 first — pure-data check, fast verification of accounting. If accounting is wrong, everything else builds on shaky ground.
- Track 1 — produce the 2.12 release analysis with new-models subsection (well-scoped, demonstrable output by EOD)

**Afternoon (Tier 2):**
- Q2 — cascading analysis (depends on Q3 verifying accounting)
- Q1 — nested graph break support (research first, then experiment if feature flag exists)
- Track 2 — train vs eval insight (depends on Q2 if cascading explains the asymmetry)
- Track 3 — #102 + #103 deep-dive (most valuable when Q1 + Q2 give us better break taxonomy)

## Out of scope

- Performance measurement (Phase 2 of DSV4 plan; separate workstream)
- Skill discovery work (pt2-skill-discovery scope)
- Anything requiring external publishing approval (drafts may be ready by EOD; landing them is Peng's call)

## Revision log

- *2026-05-02 22:37 ET:* Drafted from Peng's end-of-session backlog drop. Three tracks + three investigation questions captured. Sequencing sketched but not locked.
