# Weekly Sweep Brief — template

Fill in the 11 sections below. Each section is required (omit a section only if the corresponding finding is genuinely zero, e.g. "Compile regressions: 0" still gets a section even if there's nothing to detail).

**Encoding principle (per Peng directive 2026-05-10 21:40 ET):** when a sweep changes (re-run, amend_sweep), the report is RE-GENERATED from this template + augmented sweep data. Do NOT manually edit the previous report. New findings encode here, propagate via re-generation.

**Brevity principle (per Peng directive 2026-05-10 20:56 ET):** if a category is "no change," keep the section terse. Title carries the verdict (e.g. "## 2. Pure Dynamo wins — 0 this week"); body is one sentence. Don't conflate the report with internal infra work — corpus-side workstreams live in PLAN.md, not in this brief.

---

# PT2 Nightly Sweep Brief — {{current_date}} vs {{baseline_date}} (Dynamo scope)

**Window:** {{N}} days.
**Scope:** Dynamo-relevant changes only — apple-to-apple on common (name, mode) HF transformers pairs that ran in BOTH baseline and current. Diffusers + custom suites + timm-dependent models excluded. Eager-side timeouts and harness gaps tracked separately.

## 1. Headline

One paragraph. Required content:
- Net GB delta on common compile-success pairs (apple-to-apple, cat 3 only)
- Number of models that flipped graph_break → full_graph
- Attribution status: VERIFIED on N≥2 models? If not, mark as UNVERIFIED + name the suspected source.
- Real upstream compile regressions count (cat 2 minus known-bug-not-regression). Usually 0.

Forbidden in headline: any number that mixes cat 3 with cat 1/cat 4 (use `--pattern` segmentation).

## 1.5. Setup

If transformers / diffusers versions are NOT RECORDED for either side, surface the ⚠️ caveat at the TOP of this section (cross-week attribution claims become unverifiable). The new fail-loud guardrail prevents this going forward — only legacy sweeps will trip the warning.

| Setup         | Last week ({{baseline_date}})  | This week ({{current_date}}) |
|---|---|---|
| torch nightly | `{{baseline_torch}}`           | `{{current_torch}}`          |
| transformers  | `{{baseline_tx}}`              | `{{current_tx}}`             |
| diffusers     | `{{baseline_df}}`              | `{{current_df}}`             |

## 1.6. Apple-to-apple Topline

({{N_common_pairs}} (model × mode) HF pairs present in BOTH sweeps)

| Metric                                          | Last week     | This week     | Δ                  |
|---|---|---|---|
| Compiles fullgraph                              | {{a_full}}    | {{b_full}}    | {{delta_full}}     |
| Compiles with graph breaks                      | {{a_break}}   | {{b_break}}   | {{delta_break}}    |
| Pairs with errors                               | {{a_err}}     | {{b_err}}     | {{delta_err}}      |
| Total graph breaks ({{N_reliable}} reliable pairs; {{N_excluded}} with explain-coverage gaps excluded) | {{a_gb}} | {{b_gb}} | {{delta_gb}} ({{delta_gb_attribution}}) |

Status counts on the common-pair set are by definition byte-identical UNLESS Dynamo behavior changed. Total-GB delta is the load-bearing apple-to-apple metric.

## 1.7. Cohort Delta

|                                                  | Count           |
|---|---|
| Models added (in this week, NOT in last week)    | {{N_added}}     |
| Models removed (in last week, NOT in this week)  | {{N_removed}}   |

If N_removed > 0, list categories — each with reasons readers can act on:
- **N models** added to `skip_models.json` this week (intentional skip-list growth)
- **N models** added to `known_errors.json` this week (briefly: which bug)
- **N model classes** absent from current modellibs transformers (cohort/version drift; list the names)

## 2. Pure Dynamo wins — N flipped graph_break → full_graph

If 0: title is "## 2. Pure Dynamo wins — 0 this week" and body is one sentence ("No flips on the apple-to-apple HF set."). Skip the attribution test subsection entirely.

If >0: table of clusters / models. Then table of break-reason types eliminated by the flips. Then "### Attribution test (verified on K models)" — name the K models tested, the older-torch venv used, what reproduced, what code-paths verified byte-identical, and the narrowed window for the candidate Dynamo PR. If attribution is not yet verified, mark "### Attribution status: UNVERIFIED — testing pending" and propose the test.

## 2. Pure Dynamo wins — N models flipped graph_break → full_graph

Table of clusters / models. Then table of break-reason types eliminated by the flips (counted across baseline break_reasons of the flipped models).

Then "### Attribution test (verified on K models)" subsection — name the K models tested, the older-torch venv used, what reproduced, what code-paths verified byte-identical, and the narrowed window for the candidate Dynamo PR.

If attribution is not yet verified, mark "### Attribution status: UNVERIFIED — testing pending" and propose the test.

## 3. Compile-success → compile-success with reduced (but non-zero) GBs

Models in cat 3 with `gb_delta < 0`. Per-model table with baseline → current → Δ. Total GB reduction across this group.

## 4. Compile regressions: K real

If 0 real: state explicitly "0 real" with rationale (any cat 2 work items + why they don't count, e.g., known PT bug, deterministic-mode trigger, etc.).

If >0 real: per-regression detail, with attribution + filed pytorch issue link.

## 5. Issues — actions taken

Required subsections (each terse if zero, e.g. "Closed: 0"):
- **Closed** — list with brief framing (closed-as-completed via close-mode evidence; OR closed via known_errors entry retired after upstream fix tracking_issue confirmed closed)
- **New issues filed** — table with #, pattern, breaks, models
- **Comments added to existing issues** — list of issue numbers + brief description (model adds, status updates, etc.)
- **Net effect on tracked issues** — before/after counts

## 6. Newly compile-testable models added this week

Per definition: any flip from error to eager-success/compile-success counts as a new model. Cat 1 + cat 4 successes combined.

Required:
- Total work items + distinct model count
- % full_graph out of newly-compile-testable
- Decomposition: cat 4 (truly new) vs cat 1 (was error in baseline). For cat 1, name the prior-status breakdown (eager_error → success: N, timeout → success: N, worker_error → success: N).

## 7. NEW break-reason types surfaced (not seen in any baseline model)

Table of patterns + counts + top affected files. Highlight any new operator (e.g. `aten.bincount`) that needs an ops-coverage decision.

## 8. Actionable for Animesh / PT2 team

Bulleted, 5±2 items. Each item must:
- Name the issue # (or filable item) directly
- State the leverage (how many breaks / models cleared if the action lands)
- Be specific enough that the PT2 team can act on it without further triage from us

Higher-leverage items first.

If 0 items: state explicitly "Nothing actionable for PT2 team this week." Do NOT mention internal corpus-side workstreams (they live in PLAN.md, not the brief). Per Peng directive 2026-05-10 20:56 ET: corpus-side work is not of Dynamo team's interest.

---

(End of template.)
