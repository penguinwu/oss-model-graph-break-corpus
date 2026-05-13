# Weekly Sweep Brief — template

Fill in the 8 sections below (plus optional one-off §9+). Each section is required (omit a section only if the corresponding finding is genuinely zero, e.g. "Compile regressions: 0" still gets a section even if there's nothing to detail).

**Encoding principle (per Peng directive 2026-05-10 21:40 ET):** when a sweep changes (re-run, amend_sweep), the report is RE-GENERATED from this template + augmented sweep data. Do NOT manually edit the previous report. New findings encode here, propagate via re-generation.

**Brevity principle (per Peng directives 2026-05-10 20:56 ET + 2026-05-11 15:39 ET):** if a category is "no change," keep the section terse. Title carries the verdict (e.g. "## 6.1 Pure Dynamo wins — 0 this week"); body is one sentence. Don't conflate the report with internal infra work — corpus-side workstreams live in PLAN.md, not in this brief. Sections that have NOTHING new (no additions, no flips, no regressions) get one-sentence bodies — explanation belongs only when there IS something to explain.

**Numbering principle (per Peng directive 2026-05-11 15:39 ET):** section numbers are CONTIGUOUS integers (1, 2, 3, ...). No decimals like 1.5 or 8.5. Subsections use 6.1, 6.2, 6.3 etc. (single-decimal under the parent only).

---

# PT2 Nightly Sweep Brief — {{current_date}} vs {{baseline_date}} (Dynamo scope)

**Window:** {{N}} days.
**Scope:** Dynamo-relevant changes only — apple-to-apple on common (name, mode) HF transformers pairs that ran in BOTH baseline and current. Diffusers + custom suites + timm-dependent models excluded. Eager-side timeouts and harness gaps tracked separately.

## 1. Headline

One paragraph, 2-4 sentences max. Required content:
- Net GB delta on common compile-success pairs (apple-to-apple set — pairs that ran in BOTH baseline and current with compile-success status both times)
- Number of models that flipped graph_break → full_graph
- Real upstream compile regressions count (pairs that flipped from compile-success → error this week, minus known-bug-not-regression). Usually 0.

Forbidden in headline (per Peng directive 2026-05-11 15:39 ET):
- Attribution-status discussion (`UNVERIFIED` / `VERIFIED on N≥2 models` etc.) — that's body content for §6, not headline.
- Multi-sentence rationale on why the delta is suspect, what's covered by which issue, what's not load-bearing — body content for §6 / §7.
- Any number that mixes the apple-to-apple set with newly-compile-testable or truly-new-model exposures (use `--pattern` segmentation).
- Cat-N internal jargon ("cat 3", "cat 1", "cat 4") — Dynamo team readers don't know what those mean. Use plain English ("apple-to-apple set", "newly compile-testable", "truly new this week").

## 2. Setup

If transformers / diffusers versions are NOT RECORDED for either side, surface the ⚠️ caveat at the TOP of this section (cross-week attribution claims become unverifiable). The new fail-loud guardrail prevents this going forward — only legacy sweeps will trip the warning.

| Setup         | Last week ({{baseline_date}})  | This week ({{current_date}}) |
|---|---|---|
| torch nightly | `{{baseline_torch}}`           | `{{current_torch}}`          |
| transformers  | `{{baseline_tx}}`              | `{{current_tx}}`             |

**Diffusers row (conditional, per Peng directive 2026-05-11 17:19 ET):** include the `diffusers` row ONLY when this sweep's scope INCLUDES diffusers models. For HF-only sweeps (the default since 2026-05-10), the diffusers version is not load-bearing signal and should be omitted from the table to avoid noise.

## 3. Apple-to-apple Topline

({{N_common_pairs}} (model × mode) HF pairs present in BOTH sweeps)

| Metric                                          | Last week     | This week     | Δ                  |
|---|---|---|---|
| Compiles fullgraph                              | {{a_full}}    | {{b_full}}    | {{delta_full}}     |
| Compiles with graph breaks                      | {{a_break}}   | {{b_break}}   | {{delta_break}}    |
| Pairs with errors                               | {{a_err}}     | {{b_err}}     | {{delta_err}}      |
| Total graph breaks ({{N_reliable}} reliable pairs; {{N_excluded}} with explain-coverage gaps excluded) | {{a_gb}} | {{b_gb}} | {{delta_gb}} ({{delta_gb_attribution}}) |

Status counts on the common-pair set are by definition byte-identical UNLESS Dynamo behavior changed. Total-GB delta is the load-bearing apple-to-apple metric.

## 4. Per-team focus — top-5 lists + actionable this week (per Peng directives 2026-05-11 15:39 + 15:50 ET; per-team breakout added 2026-05-13 per WS2 dyn-shape↔dynamo split)

This section is the "if you read nothing else, read this" surface for each compiler team. Following the 2026-05-13 label exclusivity rule (`for:dynamo-team` and `dynamic-shape` are mutually exclusive buckets routing to different oncalls), §4 now has per-team subsections.

**§4.0 Per-team summary (one-liner per team):**

| Team label | Open issue count | This week's net delta (closed − new − scope-expanded) | Actionable this week (count from §4.x.3) |
|---|---:|---:|---:|
| `for:dynamo-team` | {{N}} | {{Δ}} | {{N actionable}} |
| `dynamic-shape` | {{N}} | {{Δ}} | {{N actionable}} |
| `other-compile-issue` | {{N}} | {{Δ}} | {{N actionable}} |

(Auxiliary labels like `capture-scalar-output` are SECONDARY signals layered on the team labels — counted in their primary team's row.)

---

### 4.1 — Dynamo team

#### 4.1a Top 5 Dynamo issues by blast radius (model count × break count)

| # | Symptom | Scope |
|---|---|---|
| {{issue}} | {{symptom}} | {{N models, N breaks}} |
| ... | ... | ... |

Pull from the open `[dynamo]` issue list, ranked by `model_count × break_count` (or by other agreed-upon load-bearing metric). Refresh weekly; numbers move as patterns get fixed or new issues land.

#### 4.1b Top 5 Dynamo easy fixes (narrow scope, specific symptom, MRE-anchored)

| # | Symptom | Scope | Why easy |
|---|---|---|---|
| {{issue}} | {{symptom}} | {{N models, N modes}} | {{specific reason — has MRE, narrow code path, etc.}} |
| ... | ... | ... | ... |

Curated list of `for:dynamo-team` issues where the path-to-fix is narrow + has a verified live MRE in the issue body. The "Why easy" column is the load-bearing signal — without it, this list is just a smaller version of §4.1a.

#### 4.1c Dynamo — actionable this week

Bulleted, 5±2 items. Each item must:
- Name the issue # (or filable item) directly
- State the leverage (how many breaks / models cleared if the action lands)
- Be specific enough that the PT2 Dynamo team can act on it without further triage from us

Higher-leverage items first. If 0 items: state explicitly "Nothing actionable for Dynamo team this week." Do NOT mention internal corpus-side workstreams.

(Items in §4.1c may overlap with §4.1a / §4.1b — §4.1a is "biggest patterns we should track", §4.1b is "narrow wins available", §4.1c is "what landed THIS week or what's most ripe for action NOW.")

---

### 4.2 — Dynamic-shape team

#### 4.2a Top 5 Dynamic-shape issues by blast radius

| # | Symptom | Scope |
|---|---|---|
| {{issue}} | {{symptom}} | {{N models, N breaks}} |
| ... | ... | ... |

Pull from open `dynamic-shape`-labeled issues, ranked by `model_count × break_count`. Same methodology as §4.1a; different bucket.

#### 4.2b Top 5 Dynamic-shape easy fixes

| # | Symptom | Scope | Why easy |
|---|---|---|---|
| {{issue}} | {{symptom}} | {{N models, N modes}} | {{specific reason — has MRE, narrow shape-graph, etc.}} |
| ... | ... | ... | ... |

#### 4.2c Dynamic-shape — actionable this week

Same shape as §4.1c but addressed to the dynamic-shape oncall. If 0 items: state explicitly "Nothing actionable for Dynamic-shape team this week."

---

### 4.3 — Other-compile-issue (timeout / perf / infra)

If the open `other-compile-issue` set is non-empty, list the top items with model + measured impact (compile time, recompile count, etc.). If empty, omit this subsection entirely (do NOT emit a "0 items" placeholder — keeps the brief concise).

| # | Symptom | Scope | Measured impact |
|---|---|---|---|
| {{issue}} | {{symptom (timeout / OOM / recompile churn)}} | {{N models, N modes}} | {{compile_time s / mem MB / recompile count}} |

## 5. Cohort changes (per Peng directive 2026-05-11 17:19 ET — folded old §5.2 into this section)

**Cohort definition:** the "compile-testable cohort" — model × mode pairs that are NOT in `known_errors.json` (eager-side bugs we deliberately exclude) and NOT in `skip_models.json` (intentional skips). When a `known_errors.json` entry is removed (e.g., upstream eager-side bug closed), the affected pairs re-enter the cohort and naturally appear as "Models added" below; no separate state-flip subsection is needed.

|                                                  | Count           |
|---|---|
| Models added (in this week, NOT in last week)    | {{N_added}}     |
| Models removed (in last week, NOT in this week)  | {{N_removed}}   |

**Detail rule (per Peng directive 2026-05-11 15:39 ET):** explain WHY only when there's something added or removed. Specifically:
- If `skip_models.json` had **N≥1 entries added** this week → list the names + briefly: which bug.
- If `known_errors.json` had **N≥1 entries added or removed** this week → cite the pytorch/pytorch tracking issue # by URL. **Tracking-issue citation requirement (per Peng directive 2026-05-11 13:05 ET):** Pre-publish gate (SKILL Step 5.5) verifies the tracking issue's CURRENT upstream state (closed/open) before publish.
- If `N model classes` are absent because of `transformers` version differences between sweeps (cohort/version drift) → list the names + name the version delta direction (upgrade or downgrade).
- For models that flipped from error → compile-testable this week (covered by the "Models added" line above), name the prior-status breakdown if useful (eager_error → success: N, timeout → success: N, worker_error → success: N) and cite the underlying fix (e.g., upstream tracking issue closed).
- If nothing was added/removed in any of these buckets → omit the breakdown entirely; the count table above is the full §5.

## 6. Model state changes (merged §2/§3/§4 from old template per Peng directive 2026-05-11 15:39 ET)

This section consolidates the three model-state transitions on the apple-to-apple set: full graph wins, graph-break-count reductions, and compile-success-to-error regressions. Each subsection is terse if zero (title carries verdict, body one sentence).

### 6.1 Pure Dynamo wins — N flipped graph_break → full_graph

If 0: title is "### 6.1 Pure Dynamo wins — 0 this week" and body is one sentence ("No flips on the apple-to-apple HF set."). Skip the attribution test subsection entirely.

If >0: title becomes "### 6.1 Pure Dynamo wins — N models flipped graph_break → full_graph". Table of clusters / models. Then table of break-reason types eliminated by the flips (counted across baseline break_reasons of the flipped models). Then "#### Attribution test (verified on K models)" — name the K models tested, the older-torch venv used, what reproduced, what code-paths verified byte-identical, and the narrowed window for the candidate Dynamo PR. If attribution is not yet verified, mark "#### Attribution status: UNVERIFIED — testing pending" and propose the test.

### 6.2 Compile-success → compile-success with reduced GBs

Models on the apple-to-apple set with `gb_delta < 0`. Per-model table with baseline → current → Δ. Total GB reduction across this group. If 0: title carries the verdict ("### 6.2 Reduced GBs — 0 pairs this week") and body is one sentence.

If there are GB-count REGRESSIONS (`gb_delta > 0`) on still-compile-success pairs, they go in this subsection too as a separate sub-table — they are the inverse of reductions, not compile→error transitions.

### 6.3 Compile regressions: K real (compile-success → error)

Pairs that flipped from compile-success → error. If 0 real: state explicitly "0 real" with rationale (any compile-success → error work items + why they don't count, e.g., known PT bug, deterministic-mode trigger, etc.). If 0 with no rationale needed: title carries the verdict ("### 6.3 Compile regressions — 0 real") and body is one sentence.

If >0 real: per-regression detail, with attribution + filed pytorch issue link.

## 7. Issues — actions taken

Required subsections (each terse if zero, e.g. "Closed: 0"):
- **Closed** — list with brief framing (closed-as-completed via close-mode evidence; OR closed via known_errors entry retired after upstream fix tracking_issue confirmed closed). Cite the pytorch/pytorch tracking issue # for any upstream-fix-driven close.
- **New issues filed** — table with #, pattern, breaks, models
- **Comments added to existing issues** — list of issue numbers + brief description (model adds, status updates, etc.)
- **Edited** — list of issues whose title/body materially changed this week (e.g., umbrella scope reduction, MRE addition). Brief description of what changed.
- **Net effect on tracked issues** — before/after counts

**FORBIDDEN in §7:** internal corpus-side process corrections (e.g., "we accidentally closed these issues via a buggy script and reverted"). Those are noise to the Dynamo team — track in PLAN.md, not in the brief. Per Peng directive 2026-05-11 10:55 ET: "the audience is Dynamo team, no need to talk about our internal glitches — too much information can be confusing to users."

## 8. NEW break-reason types surfaced (not seen in any baseline model)

Table of patterns + counts + top affected files. Highlight any new operator (e.g. `aten.bincount`) that needs an ops-coverage decision.

**Definitive-answer requirement (per Peng directive 2026-05-11 13:05 ET):** for EACH new break-reason type listed, the body MUST give one of three definitive answers — no "pending" or "needs investigation":
- **Covered by existing issue # NNN** (cite + link) — pattern is already tracked; this row is observational.
- **Newly-filed this sweep as #NNN** (cite + link) — new dynamo issue filed during the brief composition workflow.
- **TODO + EXECUTED before publish** — only acceptable if the TODO was completed in the same workflow cycle (e.g., file-issue subagent walked + new issue posted) and the # is now cited.

If §8 lists a break-reason without one of those three answers, the brief FAILS the SKILL Step 5.5 pre-publish gate. Do NOT publish a brief that says "needs follow-up" or "TBD" for any new break-reason.

---

(End of standard template. Optional one-off sections — e.g., "Major rewrite of Dynamo issues this week" — appear AFTER §8 with the next contiguous integer (§9, §10, ...) and are clearly labeled as one-off.)
