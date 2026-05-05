# Weekly Sweep Brief — template

Fill in the 8 sections below. Each section is required (omit a section only if the corresponding finding is genuinely zero, e.g. "Compile regressions: 0" still gets a section even if there's nothing to detail).

---

# PT2 Nightly Sweep Brief — {{current_date}} vs {{baseline_date}} (Dynamo scope)

**Window:** {{N}} days. torch nightly `{{baseline_torch}} → {{current_torch}}`. transformers `{{baseline_tx}} → {{current_tx}}`. diffusers `{{baseline_df}} → {{current_df}}`.
**Scope:** Dynamo-relevant changes only — apple-to-apple on common (name, mode) pairs that compiled in BOTH baseline and current. timm/einops out-of-scope models excluded. Eager-side timeouts and harness gaps tracked separately.

## 1. Headline

One paragraph. Required content:
- Net GB delta on common compile-success pairs (apple-to-apple, cat 3 only)
- Number of models that flipped graph_break → full_graph
- Attribution status: VERIFIED on N≥2 models? If not, mark as UNVERIFIED + name the suspected source.
- Real upstream compile regressions count (cat 2 minus known-bug-not-regression). Usually 0.

Forbidden in headline: any number that mixes cat 3 with cat 1/cat 4 (use `--pattern` segmentation).

## 2. Pure Dynamo wins — N models flipped graph_break → full_graph

Table of clusters / models. Then table of break-reason types eliminated by the flips (counted across baseline break_reasons of the flipped models).

Then "### Attribution test (verified on K models)" subsection — name the K models tested, the older-torch venv used, what reproduced, what code-paths verified byte-identical, and the narrowed window for the candidate Dynamo PR.

If attribution is not yet verified, mark "### Attribution status: UNVERIFIED — testing pending" and propose the test.

## 3. Compile-success → compile-success with reduced (but non-zero) GBs

Models in cat 3 with `gb_delta < 0`. Per-model table with baseline → current → Δ. Total GB reduction across this group.

## 4. Compile regressions: K real

If 0 real: state explicitly "0 real" with rationale (any cat 2 work items + why they don't count, e.g., known PT bug, deterministic-mode trigger, etc.).

If >0 real: per-regression detail, with attribution + filed pytorch issue link.

## 5. Issues — actions taken (umbrella-split policy)

Required subsections:
- **Closed as superseded** — list of umbrella issues split this week, with sub-issue numbers
- **New issues filed** — table with #, pattern, breaks, models
- **Updated existing issues with current data** — list of issues refreshed with this week's numbers + any new findings
- **Net effect on tracked issues** — before/after counts

If no issue cleanup happened this week: state "No issue cleanup this week — existing umbrellas remain (#102, #103, #122, etc.)."

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

---

(End of template.)
