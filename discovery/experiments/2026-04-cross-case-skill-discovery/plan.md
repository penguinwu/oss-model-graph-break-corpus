# Experiment Plan: Cross-Case Skill Discovery (April 2026)

**Slug:** `2026-04-cross-case-skill-discovery`
**Title:** Cross-Case Skill Discovery
**Type:** Discovery (no preconceived expectations on agent behavior)
**Owner:** Otter
**Workstream:** WS1 — Skill discovery via corpus
**Umbrella issue:** #60
**Status:** active
**Created:** 2026-04-24
**Last updated:** 2026-04-25 (V8 added — "model-layer fix only" — shipped + launched on case 3b; methodology framing "shut the door to shortcuts" elevated to organizing principle)

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

**Organizing principle — *shut the door to shortcut solutions.*** Each non-baseline variant closes one shortcut the agent might otherwise default to. Closing a door doesn't dictate the answer — it forces the agent to explore deeper, more novel graph-break strategies that only become visible when the easy path is removed. The variant catalog is therefore not a flat list but a *progression of door-closings*, each peeling back another layer of agent default behavior. The V0 baseline is the "all doors open" reference; everything else asks "what's the agent capable of when it can't take *this* shortcut?"

| Variant | Tier | Door it closes | Constraint added | What it surfaces |
|---|---|---|---|---|
| V0 (bare) | **standard** | (none — all doors open) | (none) | The agent's default reach. What does it pick first when nothing is forbidden? |
| V2 (bitwise) | **standard** | The float-reorder shortcut | "Compiled output must be bitwise equal to eager. Strategies that reorder floats are not acceptable." | Pushes toward escape-hatch family (custom_op / disable / cond) — only those preserve op order. |
| V4 (no escape hatches) | conditional | The bypass shortcut | "Do not use custom_op / dynamo.disable / allow_in_graph / torch.cond." | Forces in-graph fix (rewrite, not bypass). **Trigger:** any V0 or V2 trial used a canonical escape hatch (`custom_op`, `disable`, `cond`, `allow_in_graph`, `nonstrict_trace`, `leaf_function`) OR used `torch.compiler.is_compiling()` as a runtime guard. |
| V6 (no config flags) | conditional | The runtime-flag-flip shortcut | "Do not modify torch._dynamo.config." | Forces source-code fix, not a config setting. **Trigger:** any V0 or V2 trial flipped a `torch._dynamo.config` flag (e.g. `capture_scalar_outputs`, `capture_dynamic_output_shape_ops`). |
| V8 (model-layer only) | conditional | The setup-edit shortcut | "Do not edit the test/baseline script. The fix must live entirely in the model source files. Setup-layer edits — input shapes, np→torch swaps, compile call site, random-seed patterns — are NOT acceptable." | Forces model-layer fix; agent can't sidestep breaks by reshaping the harness. **Trigger:** standard matrix shows ≥50% of trials with `fix_status = setup-required` on any cell — i.e. the case has a setup-edit attractor. |

**Skipped:** V1 (sparsity_preserved) — Dbrx-MoE-specific language, doesn't generalize.

**Why "door-closing" is the methodology, not just a list of constraints.** Closing a door is the only way to discover what's *behind* it. If V0 always produces a setup-edit fix, we never see whether the agent can reach a model-layer fix — V0 alone tells us nothing about the depth of the agent's strategy space. V8 is the most aggressive door-closing because it removes the largest shortcut family (entire harness rewrite), forcing the agent to either step up (real model-layer fix) or crater (no agent-reachable model-layer fix exists for this break shape). Both outcomes are signal. The catalog should grow this way: *whenever the standard matrix reveals a dominant shortcut attractor, add a door-closing variant for it.*

**Standard matrix (always run):** 2 skill settings × 2 variants (V0, V2) × N=3 trials → **12 trials per case**.

**Conditional follow-ups:** if a trigger fires, run the matching variant on the same skill arm × 3 trials. Each conditional variant adds up to 6 trials (2 skill arms × 3 trials). If none of V4/V6/V8 trigger, the case stops at 12 trials.

**Why conditional, not always-on:** Mistral3 case 3a ran the full 24-trial matrix; the V4/V6 cells produced essentially the same pattern as V0/V2 (master finding: the perf delta lived in `fix_locus`, not `variant`). Spending 12 trials per case on V4/V6/V8 by default wastes budget. Run them when the standard matrix surfaces the thing they're designed to probe — i.e. when there's an actual shortcut attractor worth closing the door on.

**Total experiment scope (lower bound):** 4 models × 12 trials = 48 trials. **Upper bound (V4, V6, and V8 all trigger on every case):** 4 × 30 = 120 trials. Sequential per-model (no parallel — Pilot 3 race-condition lesson).

**Per-trial wall budget:** 1800s (30 min). Per-case wall (standard): ~6 hrs. Per-case wall (full with all conditionals): ~15 hrs. Total experiment wall: 24-60 hrs spread across multiple sessions.

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

- *`validate.py` (legacy field, canonical-input check):* `import_ok`, `eager_ok`, `compile_ok`, `graph_count`, `graph_break_count`, `max_diff_compiled_vs_eager_now`, `max_diff_vs_eager_baseline`.
- *`validation_v2` (preferred — added by `discovery/revalidate.py`):*
  - `integrity`: `{import_ok, eager_ok, compile_ok}`
  - `fix_status`: verdict — `general` (model-layer alone fixes; gb=0 under canonical inputs) | `setup-required` (agent's full fix incl. setup-layer edits achieves gb=0, but model-layer alone doesn't) | `none` (gb>0 even in agent's own run) | `unknown`
  - `details`: `{gb_in_agent_run, gb_under_canonical_inputs, max_diff_compiled_vs_eager, max_diff_vs_baseline}`
- *`measure_perf` tier-1 (small inputs) + tier-2 (realistic inputs):* `eager_ms`, `compiled_ms`, `speedup`, `peak_mem_mb`, `compile_s`.
- *`agent_diff.patch`:* which files edited, what changed.
- *Stream metadata:* turns, wall time, $ cost, agent's final summary.
- *Strategy fingerprint (5-axis lock from Phase 1):* `{fix-locus, fix-shape-family, op-order-preserved, sparsity-preserved, escape-hatch-used}`. Per-case reports may add a 6th axis if the case demands it.

**Vocabulary — "model-layer" vs "setup-layer" changes:**

- *Model-layer changes:* edits to `modeling_*.py` files. Affect the model itself. Generalize across callers (production HF processor outputs, standalone scripts, any input shape).
- *Setup-layer changes:* edits to the run script (`baseline_<model>.py`) — input formatting, flag setting, wrapper code. Local to that run; don't affect the model. Legitimate fixes, just narrower in scope.

The `fix_status` verdict surfaces this distinction: `general` = model-layer fix; `setup-required` = setup-layer or model+setup combination that doesn't generalize past the agent's setup.

## Stop conditions

**Per case:**

- All trials in any cell fail (gb stays at baseline) → prompt is too vague for this case, stop and revise the case file
- First 4 trials in a cell produce identical fixes → no diversity in that cell, halve N for the rest of that cell
- Per-case wall time exceeds 24 hrs → pause, reassess

**Per experiment:**

- If no case in {3a, 3b} produces diverse strategies → harness frame is wrong; pause Phase 3, revisit `discovery/design.md`
- If only V0 trials succeed across cases → variant catalog is broken; rebuild before continuing

## Per-case execution shape (7-phase flow)

For each case in order:

1. *Author the case files.* `corpus/discovery/cases/<case>.{py,baseline.json}` per `discovery/design.md` §6 schema. Set up `/tmp/discovery-runs/<case_id>/` with `baseline_*.py`, `validate.py`, source backups, baseline_eager_output.pt.
2. *Pre-register the case as a per-model issue.* Use `tools/new_case_issue.py <experiment-slug> <case_id> "<Model name>"` (do NOT hand-roll — see `corpus/CLAUDE.md` §"Discovery Experiments"). The scaffold injects the canonical Pre-launch checklist (below) into the issue body.
3. *Walk the Pre-launch checklist* (canonical below + any case-specific additions in the per-model issue). Tick each item before launching. If a case has quirks, add case-specific items to the issue's "Case-specific additions" section before launch.
4. *Launch the standard matrix:* `python -m discovery.run_case --case <case_id> --variants V0,V2 --skills none,/home/pengwu/projects/oss-model-graph-break-corpus/discovery/skills/debug-graph-breaks/SKILL.md --n 3 --timeout 1800` (12 trials sequential, ~6 hrs wall).
5. *Run Phase 0 audit + Phase A-F analysis* per `discovery/skills/per-case-analysis/SKILL.md`. Produces `reports/<case_id>/findings.md` + `fingerprints.csv`. Phase 0 (data trustworthiness) GATES Phase A — don't analyze on suspect data. **Phase B includes the conditional-trigger check:** if any V0/V2 trial used a canonical escape hatch or `is_compiling`, queue a V4 follow-up; if any flipped a `torch._dynamo.config` flag, queue a V6 follow-up; if ≥50% of trials in any cell ended `fix_status = setup-required`, queue a V8 follow-up. If none trigger, document "no conditional follow-ups warranted" in Phase B and stop at 12 trials.
6. *Conditional follow-up runs (if Phase B triggered):* relaunch with `--variants V4`, `--variants V6`, and/or `--variants V8` (same `--skills` and `--n` arguments). Re-run Phase A-F over the union of trials.
7. *Commit the report to main* (`findings.md` + `fingerprints.csv` under `discovery/experiments/<exp>/reports/<case_id>/`). Push. Post a headline summary as a comment on the per-case issue (TL;DR + headlines + note on whether V4/V6 follow-ups ran). Per-case issue moves to Done. (Workflow change 2026-04-25: PR-FIRST discontinued — direct-to-main is the convention now. See `per-case-analysis/SKILL.md` Phase F for rationale.)

## Pre-launch checklist (canonical — snapshot into per-model issues)

Tick each item before launching the harness. Paste command output for any check that's not obviously a one-liner.

- [ ] *Case file imports cleanly:* `python -c "from discovery.cases.<case_id> import get_case_spec; print(get_case_spec().case_id)"` returns the case_id without exception.
- [ ] *Baseline.json present:* `ls discovery/cases/<case_id>.baseline.json` exists with reasonable speedup numbers.
- [ ] *Pristine source backups:* `ls /tmp/discovery-runs/<case_id>/*.original` returns expected files (one `.original` per watched source file).
- [ ] *Baseline eager output saved:* `ls /tmp/discovery-runs/<case_id>/baseline_eager_output.pt` exists.
- [ ] *Validate runs cleanly:* `python /tmp/discovery-runs/<case_id>/validate.py` reports `import_ok`, `eager_ok`, `compile_ok` all true; `graph_break_count` matches the case's documented baseline (in baseline.json or case docstring).
- [ ] *Backend confirmed:* `baseline_*.py` and `validate.py` use `torch.compile(model)` with default backend (inductor) — NOT `backend="eager"`. Discovery uses inductor; sweep uses eager (per design.md §2.1).
- [ ] *Watched files set:* case spec lists exactly the files agent may edit; no shared infra (`sdpa_attention.py`, decomposition tables) included; allowed list matches what the prompt declares.
- [ ] *Skill file accessible:* `ls discovery/skills/debug-graph-breaks/SKILL.md` exists (used in the with-skill arm).
- [ ] *No conflicting trials in flight:* `ps -ef | grep "discovery.run_case --case <case_id>"` returns nothing — sequential trials would conflict on shared model files.

If any item is flagged, fix or document the deviation BEFORE launching. Don't launch with unchecked items.

## Cross-case synthesis (after 3a–3d done)

Write `discovery/experiments/2026-04-cross-case-skill-discovery/synthesis.md` answering Q7 (cross-case generalization). Then:

- Mental-model doc in PARA — *The Mental Model of Building a Graph-Break Skill (or Agent)* (per Peng 2026-04-23 forcing function)
- Workplace post to PT2 working group + Compile Q&A

## Runner changes — DONE (commit 3deefd0, 2026-04-24)

All three blockers landed before launch:

1. ✅ `--skills` argument on `run_case.py`. Comma-separated; `none` = bare baseline arm, anything else = path to a markdown file injected via `--append-system-prompt-file`. Plumbed through `runner.py` → `_run_agent`.
2. ✅ Diff capture fix. Post-validation re-snapshot (`agent_diff.patch` is taken twice; larger one wins; flagged `diff-promoted-post-validate` if the second was used).
3. ✅ Default `--timeout` bumped 1200 → 1800.

These are scope for a separate runner-changes PR (linked from the umbrella when filed).

## Revision log

- *2026-04-24:* Plan created. Greenlit by Peng. Replaces the inline-per-issue methodology that risked drifting between cases. Skill axis added back per Peng 2026-04-24 23:00 ET (was wrongly dropped in the first draft of the per-model issue #59).
- *2026-04-25:* Switched standard matrix from 4 variants (V0,V2,V4,V6 = 24 trials/case) to 2 variants (V0,V2 = 12 trials/case) with V4/V6 as conditional follow-ups gated on Phase B trigger checks. Reason: Mistral3 case 3a ran the full 24-trial matrix and the master finding (perf delta lives in `fix_locus`, not `variant`) showed the V4/V6 cells produced essentially the same pattern as V0/V2 — half the budget would have surfaced the same headline. The conditional triggers (`is_compiling` or canonical escape hatch → V4; config flag flip → V6) ensure V4/V6 still run when they'd be informative. Mistral3's existing 24-trial run is grandfathered.
- *2026-04-25 (later):* PR-FIRST workflow for analysis output discontinued. Per-case findings now commit to main directly + headline summary on the per-case issue. PR diffs are hard to read for analysis docs (most of the value is the prose, not line-level changes), and feature branches accumulated merge conflicts when methodology landed during a review cycle. Encoding removed from CLAUDE.md, per-case-analysis SKILL Phase F, and step 7 of Per-case execution shape above.
- *2026-04-25 (evening):* Added V8 ("model-layer fix only") as a conditional variant. Trigger: ≥50% of trials in any cell end `fix_status = setup-required` (i.e. setup-edit attractor present). Designed in response to VitsModel case 3b's 12/12 setup-required result on V0/V2/V4. Promoted the underlying methodology — *shut the door to shortcut solutions to steer agents toward novel/deeper graph-break strategies* — to organizing principle of the variant catalog. Variant table now lists each variant's "door it closes" alongside its constraint. Catalog growth rule documented: whenever the standard matrix reveals a dominant shortcut attractor, add a door-closing variant for it.
