# Experiment Plan: Cross-Case Skill Discovery (April 2026)

**Slug:** `2026-04-cross-case-skill-discovery`
**Title:** Cross-Case Skill Discovery
**Type:** Discovery (no preconceived expectations on agent behavior)
**Owner:** Otter
**Workstream:** WS1 — Skill discovery via corpus
**Umbrella issue:** #60
**Status:** active
**Created:** 2026-04-24
**Last updated:** 2026-04-25

---

## What this experiment is

Point an agent at real models that hit many graph breaks under `torch.compile(fullgraph=True)`. Tell it to fix them. Watch what it does. Repeat across multiple models and across multiple agent configurations. Write down what we observe.

This is a **path-finding** experiment, not a hypothesis test. We do not predict outcomes — we collect them.

## Why we're doing it

To learn what a code agent reaches for when handed a complex graph-break-heavy model. Phase 1 cases (Dbrx, Jamba) each had a single dominant break shape and produced limited variation. We want richer signal:

- Which break shapes does the agent attack first when the model has multiple kinds layered together?
- Where does it edit (model source vs test script vs both)?
- Which fix patterns does it reach for under different prompt constraints?
- Does loading our `debug-graph-breaks` skill change its strategy, or does the agent ignore the skill content?
- Does the agent fix all the breaks or stop at the easy ones?

The answers will inform the skill catalog, the constraint variant set, and the future shape of the graph-break-resolver agent.

## Models in this experiment

Selected for graph-break shape diversity. Not for ease of fix — agents should encounter the messy real cases.

| # | Case ID | Model | gb count (corpus) | Distinct user-code shapes | Selection rationale | Per-case issue |
|---|---|---|---|---|---|---|
| 3a | `mistral3_data_dep` | `Mistral3ForConditionalGeneration` | 16 | sdpa is_causal + unfold decomp + scalar_dense/nonzero | Multimodal, multi-shape | #59 |
| 3b | `vits_model_train` | `VitsModel` (train mode) | 29 | `as_proxy`, `find_spec`, novel categories | Train mode + new break categories | TBD |
| 3c | `aria_data_dep` | `AriaForConditionalGeneration` | 27/28 | (multimodal, similar shape space to Mistral3) | Second probe of Mistral3-shaped space | TBD |
| 3d | `paddle_ocr_vl` | `PaddleOCRVL` | 19 | (data-dep ops across new operator surface) | Operator-generalization probe | TBD |

Order is deliberate (highest diversity first); do not reorder without amending the umbrella + this plan.

## Methodology — the experimental matrix

For each model, we run the discovery harness across two crossed axes:

**Axis 1: Skill loaded.** Whether the agent has the `debug-graph-breaks` skill loaded into its system prompt before the trial begins.

| Skill setting | What it means |
|---|---|
| `none` | Bare Claude. Agent figures out how to diagnose and fix from scratch. |
| `debug-graph-breaks` | Arsh's skill is loaded — gives the agent a step-by-step Detect → Diagnose → Fix → Benchmark workflow. Pushes toward escape hatches (`@leaf_function`, `@nonstrict_trace`, `torch.compiler.disable()`) and code restructuring. |

**Skill source:**

- Canonical: [`azahed98/pytorch:claude/debug_graph_breaks/.claude/skills/debug-graph-breaks/SKILL.md`](https://github.com/azahed98/pytorch/blob/claude/debug_graph_breaks/.claude/skills/debug-graph-breaks/SKILL.md) — Arsh's fork, not landed in main pytorch yet
- Vendored copy in this repo: `corpus/discovery/skills/debug-graph-breaks/SKILL.md` — re-fetch from canonical URL if you suspect drift
- Design doc (what the skill does, examples, comparison vs no-skill): Google Doc `1vs8pIShvxM4Z9env51-x0BTTtSE5S1Le4RNVgpgCjbw`

**Skill ↔ constraint interactions to watch in results:**

- The skill pushes toward escape hatches. V4 (no escape hatches) forces the skill to find non-escape-hatch fixes — direct test of whether the skill has alternate strategies.
- The skill references external docs at meta-pytorch.org/compile-graph-break-site/. Trial agents may not have internet — skill text says "If the user provides a local path to a clone of the graph break website repository, read the documentation files directly from that directory instead." Future runner change: support `--add-dir <docs-clone>` if internet access becomes the limiter.

**Axis 2: Prompt constraint.** A single-sentence constraint appended to the case prompt. Steers the agent toward / away from particular strategy families.

| Variant | Constraint added | Why include this variant |
|---|---|---|
| V0 (bare) | (none) | Baseline. What does the agent reach for first when given no extra direction? |
| V2 (bitwise) | "Compiled output must be bitwise equal to eager. Strategies that reorder floats are not acceptable." | Pushes toward escape-hatch family (custom_op / disable / cond) — only those preserve op order. |
| V4 (no escape hatches) | "Do not use custom_op / dynamo.disable / allow_in_graph / torch.cond." | Forces in-graph fix (rewrite, not bypass). |
| V6 (no config flags) | "Do not modify torch._dynamo.config." | Forces source-code fix, not a runtime flag flip. |

**Skipped:** V1 (sparsity_preserved) — Dbrx-MoE-specific language, doesn't generalize.

**Cross product:** 2 skill settings × 4 constraints = 8 cells. **N=3 trials per cell** → 24 trials per model.

**Total experiment scope:** 4 models × 24 trials = 96 trials across the experiment. Sequential per-model (no parallel — Pilot 3 race-condition lesson).

**Per-trial wall budget:** 1800s (30 min). Per-model wall: ~12 hrs. Total experiment wall: ~48 hrs spread across multiple sessions.

## What we hold constant

- *LLM model:* Claude Opus (whatever the harness's `claude` CLI binds to).
- *Agent persona / system prompt:* default. No Otter persona injected.
- *Random seed:* 0 (model init + input generation).
- *Env:* `/home/pengwu/envs/torch211/` — PyTorch 2.11.0+cu128, transformers 5.5.3, Python 3.12.
- *Tools agent has:* Read, Write, Edit, Bash. No Plan, no MCP, no internet.
- *Files agent may edit:* model source files (e.g., `modeling_mistral3.py`, `modeling_pixtral.py`) + the per-case test script. Shared infra (sdpa_attention.py, decomp tables) is off-limits — explicit in the prompt.

## What we are NOT testing in this experiment (gaps + future axes)

- *Different LLM models* (only Claude Opus).
- *Different model configs* (only the tiny config defined per-case).
- *Different timeout budgets* (only 1800s).
- *Different agent personas / system prompts.*
- *Different tool sets.*
- *Bitwise-equality field in the validator* (a separate WS4 task — `bitwise_equal: bool` alongside `max_abs_diff: float`. Will be useful here once it ships).

If any of these end up mattering, they get their own follow-up experiment.

## Open questions to answer by observation

These are **questions, not hypotheses.** We collect data and report what we see. No "expectation: X" baked in.

- **Q1: Where does the agent edit?** Model source? Test script? Both? Different across trials?
- **Q2: Which break shapes get fixed?** Does the agent address all the breaks, just the easy ones (e.g., 1-line guard fix), or stop partway?
- **Q3: What fix patterns does the agent reach for?** Restructure / delete branch / `torch._check_*` / try-except / custom_op / disable region / something we haven't catalogued?
- **Q4: Does the fix preserve performance?** Compiled-vs-eager speedup baseline (per case) — does a successful fix improve it, hold it, or regress it?
- **Q5: Do prompt constraints (V2 / V4 / V6) actually steer strategy?** Or does the agent converge on the same fix regardless?
- **Q6: Does loading the `debug-graph-breaks` skill change anything?** Different fix patterns? Faster convergence? More breaks fixed? Or no observable difference (which would itself be data)?
- **Q7: Do answers to Q1–Q6 generalize across models, or is each case its own attractor set?** This is the cross-case synthesis question, answered when all 4 cases close.

## What we record per trial

- *`validate.py` output:* `import_ok`, `eager_ok`, `compile_ok`, `graph_count`, `graph_break_count`, `max_diff_compiled_vs_eager_now`, `max_diff_vs_eager_baseline`.
- *`measure_perf` tier-1 (small inputs) + tier-2 (realistic inputs):* `eager_ms`, `compiled_ms`, `speedup`, `peak_mem_mb`, `compile_s`.
- *`agent_diff.patch`:* which files edited, what changed.
- *Stream metadata:* turns, wall time, $ cost, agent's final summary.
- *Strategy fingerprint (5-axis lock from Phase 1):* `{fix-locus, fix-shape-family, op-order-preserved, sparsity-preserved, escape-hatch-used}`. Per-case reports may add a 6th axis if the case demands it.

## Stop conditions

**Per case:**

- All trials in any cell fail (gb stays at baseline) → prompt is too vague for this case, stop and revise the case file
- First 4 trials in a cell produce identical fixes → no diversity in that cell, halve N for the rest of that cell
- Per-case wall time exceeds 24 hrs → pause, reassess

**Per experiment:**

- If no case in {3a, 3b} produces diverse strategies → harness frame is wrong; pause Phase 3, revisit `discovery/design.md`
- If only V0 trials succeed across cases → variant catalog is broken; rebuild before continuing

## Per-case execution shape

For each case in order:

1. Author `corpus/discovery/cases/<case>.{py,baseline.json}` per `discovery/design.md` §6 schema
2. Pre-flight: model loads, baseline correctness recorded, baseline perf seeded
3. Pre-register the case as a per-model issue (use existing #59 as template)
4. Launch the harness (24 trials sequential, ~12 hrs wall)
5. Tier-2 enrichment via `enrich_tier2.py`
6. Write per-case findings → `discovery/experiments/2026-04-cross-case-skill-discovery/reports/<case>.md`
7. Submit PR adding the report → links back to per-model issue → merge → issue moves to Done

## Cross-case synthesis (after 3a–3d done)

Write `discovery/experiments/2026-04-cross-case-skill-discovery/synthesis.md` answering Q7 (cross-case generalization). Then:

- Mental-model doc in PARA — *The Mental Model of Building a Graph-Break Skill (or Agent)* (per Peng 2026-04-23 forcing function)
- Workplace post to PT2 working group + Compile Q&A

## Runner changes required before relaunch

The current `discovery/runner.py` cannot vary the skill axis or reliably capture diffs under timeout. These must ship before any case in this experiment relaunches:

1. **Add `--skill <id>` argument to `discovery/run_case.py`.** Plumb through `runner.py` and `_run_agent` so the trial agent runs with the named skill loaded. Need to verify whether the Claude CLI accepts skill loading via `--add-dir <skill-dir>`, `--system-prompt-include <file>`, or some other flag. If no clean flag exists, prepend the skill markdown to the prompt.

2. **Fix diff capture under SIGTERM.** When timeout fires after the agent has edited files, the captured `agent_diff.patch` is empty (only the post-trial mutation flag survives). Re-snapshot the diff post-SIGTERM but before declaring done.

3. **Default `--timeout` to 1800s.** Current default is 1200s, recent runs used 900s. Both too short for multimodal cases.

These are scope for a separate runner-changes PR (linked from the umbrella when filed).

## Revision log

- *2026-04-24:* Plan created. Greenlit by Peng. Replaces the inline-per-issue methodology that risked drifting between cases. Skill axis added back per Peng 2026-04-24 23:00 ET (was wrongly dropped in the first draft of the per-model issue #59).
