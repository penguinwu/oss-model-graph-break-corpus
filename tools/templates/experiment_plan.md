# Experiment Plan: {title}

**Slug:** `{slug}`
**Title:** {title}
**Type:** TBD (Discovery / Validation / Elimination — pick one)
**Owner:** Otter
**Workstream:** TBD (WS1 / WS2 / WS3 / WS4)
**Umbrella issue:** {umbrella_issue}
**Status:** active
**Created:** {created}
**Last updated:** {created}

---

## What this experiment is

(One paragraph. What are we actually doing? Be concrete.)

## Why we're doing it

(One paragraph. What decision/understanding does this unblock? If we get answer X, what changes? If Y, what changes? If "nothing changes regardless," reconsider running it.)

## Models / cases in this experiment

(For multi-case experiments. Otherwise delete this section and use `## Setup`.)

| # | Case ID | Model | gb count | Selection rationale | Per-case issue |
|---|---|---|---|---|---|
| 1 | TBD | TBD | TBD | TBD | TBD |

Order is deliberate (highest priority first); do not reorder without amending this plan.

## Methodology — the experimental matrix

(What axes do we vary? What do we hold constant? Be explicit. List EVERY axis — agents pick this up cold.)

**Axis 1: TBD**

| Setting | What it means |
|---|---|
| TBD | TBD |

**Axis 2: TBD**

| Setting | What it means |
|---|---|
| TBD | TBD |

**Cross product:** TBD cells. **N=? trials per cell.** **Total trials per case:** TBD. **Per-trial wall budget:** TBD.

## What we hold constant

(LLM model, agent persona, random seed, env, tools, etc. Everything that's NOT being varied.)

- *LLM model:* TBD
- *Agent persona / system prompt:* TBD
- *Random seed:* TBD
- *Env:* TBD
- *Tools agent has:* TBD
- *Files agent may edit:* TBD

## What we are NOT testing in this experiment (gaps + future axes)

(Things that COULD matter but we're explicitly not varying this round. Surface so they don't get forgotten as future axes.)

## Open questions to answer by observation

(Questions, NOT hypotheses. We collect data and report what we see. No "expectation: X" baked in unless you have one and want to pre-register.)

- **Q1:** TBD
- **Q2:** TBD

## What we record per trial

(Per-trial data schema. What fields? What raw artifacts? What derived metrics?)

- TBD

## Stop conditions

**Per case:**

- TBD

**Per experiment:**

- TBD

## Per-case execution shape

(Step-by-step shape that applies to every case in this experiment.)

1. Author `corpus/discovery/cases/<case>.{{py,baseline.json}}`
2. Pre-flight: model loads, baseline correctness recorded
3. Pre-register the case as a per-model issue (`tools/new_case_issue.py {slug} <case_id> <model_name>`)
4. Launch the harness
5. Tier-2 enrichment via `enrich_tier2.py`
6. Write per-case findings → `discovery/experiments/{slug}/reports/<case>.md`
7. Submit PR adding the report → links back to per-model issue → merge → issue moves to Done

## Cross-case synthesis (after all cases close)

(For multi-case experiments — what does the cross-case writeup answer? Final deliverables?)

Write `discovery/experiments/{slug}/synthesis.md` answering Q? (cross-case generalization).

## Revision log

- *{created}:* Plan created.
