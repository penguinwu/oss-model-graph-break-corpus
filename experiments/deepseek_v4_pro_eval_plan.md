---
plan: DeepSeek V4 Pro — model evaluation
status: active
owner: Otter
created: 2026-04-25
last_check: 2026-04-25
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# DeepSeek V4 Pro — Evaluation Plan

> **Status: APPROVED for execution** (Peng review 2026-04-25 13:22 + 13:32 ET — all 5 open questions resolved including the AutoDev Kanban board pointer).

## Goal

Evaluate the latest DeepSeek-V4-Pro release ([huggingface.co/deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)) on **four dimensions**, with results landing on a AutoDev Kanban board dashboard for ongoing visibility:

| Dimension | What we measure | Pass criterion |
|---|---|---|
| **Graph breaks** | `torch.compile(fullgraph=True)` graph break count + categorization | gb=0 ideal; >0 → file as corpus issue per category |
| **Correctness** | `torch.compile(default backend)` output vs eager output | `max_abs_diff` ≤ **1e-4** (default per upstream `pytorch/benchmarks/dynamo/common.py:1069`); flag bitwise non-equivalence separately |
| **Performance** | tier-1 (small / "fast") and tier-2 (realistic) speedup of compiled vs eager; compile time | speedup > 1.0x. (No prior-release regression check — Peng 2026-04-25: no comparable baseline; absolute numbers only.) |
| **Numerics** | `bitwise_equal` field + max_abs_diff distribution; bf16 vs fp32 stability | record per layer / per output head if model is composite |

## Why this model

DeepSeek V4 Pro is a *fresh release* not yet in the corpus. Standing eval question: when a high-profile new model lands, where do PyTorch's compile, correctness, and perf stories stand on it? This evaluation answers that — and produces actionable issues if any dimension fails.

This is the **standard model-evaluation project**, NOT WS1 skill discovery. Different from the per-case agent experiments under `discovery/experiments/`.

## What we know about the model (TBD — fill before launch)

To verify by reading the HF model card + config:

- [ ] *Architecture:* presumably MoE (DeepSeek V3 was 671B MoE w/ 37B active). Confirm V4 Pro's exact shape — total params, active params, expert count, MoE dispatch op.
- [ ] *Tokenizer:* HF tokenizer used in their inference example (DeepSeek typically uses their own).
- [ ] *Config:* hidden size, layer count, attention type (MLA?), routing strategy.
- [ ] *Memory footprint:* full-model load on devvm (1× H100 80GB) feasible, or do we need a tiny config?
- [ ] *Inference example:* HF README's canonical input + expected output (used as the eager reference).
- [ ] *Known issues:* any caveats in model card re: fp16/bf16 stability, sliding window attn, etc.
- [ ] *Comparable previous release:* DeepSeek V3 / V3-Pro baseline for perf-regression check.

## Two-phase execution (Peng 2026-04-25)

| Phase | Config | Focus | Why |
|---|---|---|---|
| **Phase 1** | Tiny config (smallest variant we can configure or HF dev-config) | **Correctness** (graph breaks + numerics) | Establish that compile produces correct output before scaling. Memory-feasible on 1× H100; fast iteration. |
| **Phase 2** | Full model (or largest config that fits 1× H100; sharded if needed and infra exists) | **Performance** (tier-1 + tier-2) + reconfirm correctness at scale | Real-world inference perf. Only meaningful if Phase 1 correctness lands. |

Phase 2 launches only after Phase 1 produces a clean correctness verdict (or with explicit Peng sign-off if Phase 1 surfaces issues we want to defer). Each phase produces its own `results.json` so downstream analysis can stage the conclusions.

## Methodology

### Evaluation harness

Build a config-driven evaluation following the existing `experiments/` pattern (`tools/run_experiment.py`). One JSON config file, runnable as:

```bash
~/envs/torch211/bin/python tools/run_experiment.py run experiments/configs/deepseek-v4-pro.json
```

The config drives a sweep of 4 backend × 2 mode combinations:

| Backend | Mode | Purpose |
|---|---|---|
| `eager` | inference | reference output for correctness; eager perf baseline |
| `inductor` (default) | inference | primary compile target; perf + correctness |
| `aot_eager` | inference | isolates dynamo from inductor (for triaging numerics divergence) |
| `inductor` | inference, `dtype=bf16` | numerics check at training-relevant precision |

(Drop `aot_eager` and the bf16 row if scope creep — they're triage tools, not gates.)

### Per-dimension protocol

**Graph breaks.** Run `torch._dynamo.explain(model)(*inputs)` and the standard `fullgraph=True` compile. Record `graph_break_count`, the unique break categories, and file-line locations. If gb>0, run the existing categorization (per `correctness/` and `tools/file_issues.py`) and tag each break with corpus's existing labels (e.g. `[dynamo] data-dependent branching`, `[dynamo] Tensor.item()`).

**Correctness.** Compute `max_abs_diff(eager_out, compiled_out)` on a fixed canonical prompt (pick from HF README example). Tolerance: **1e-4** across dtypes — same default as `pytorch/benchmarks/dynamo/common.py:1069` (`tolerance = args.xla_tolerance if args.trace_on_xla else 1e-4`). Below this is "pass"; above is "above-tolerance" and gets filed. Independently compute `bitwise_equal: bool` (`torch.equal`) so we separate "FP-rounding" from "above-tolerance" from "bitwise-different". This is the same field that's in WS4 backlog — implementing it here is partial WS4 progress.

**Performance.** Use `discovery/perf.py:measure_perf` (the primitive shipped for skill-eval — same methodology applies here: gc + cache clear, warmup separate from timing, median over reps, real `compile_times()`):
- *tier-1 ("fast"):* small input (e.g. 16-token prompt) — measures compile fixed cost + per-call overhead.
- *tier-2 ("realistic"):* representative inference workload (e.g. 2048 prompt + 256 generate) — measures steady-state.

Record: `eager_ms`, `compiled_ms`, `speedup`, `peak_mem_mb`, `compile_s`. Report absolute numbers — no prior-release regression check (Peng 2026-04-25: no comparable baseline).

**Reference benchmark stack to align with:** [pytorch/pytorch/benchmarks](https://github.com/pytorch/pytorch/tree/7a6f3270a85fefb5716d3224cf1936c07b0296e4/benchmarks) — specifically `benchmarks/dynamo/huggingface.py` for HF model handling and `benchmarks/dynamo/common.py` for the shared infra. The corpus's `discovery/perf.py` should align its methodology with this codebase (warmup, iteration counts, stat reporting) so our numbers are comparable to upstream PyTorch's benchmark dashboards. Treat divergences from upstream methodology as deliberate, documented choices, not accidents.

### RNG determinism (HF model gotcha — required for any meaningful perf or correctness number)

Per Animesh: many HF models invoke `torch.manual_seed` internally during forward (dropout init, sampling, layer-norm noise variants), so even with a fixed input you can get different outputs across two eager runs. Without addressing this, **every accuracy comparison is suspect** and tier-1 perf may show false variance.

The canonical pattern from [`benchmarks/dynamo/common.py:540`](https://github.com/pytorch/pytorch/blob/7a6f3270a85fefb5716d3224cf1936c07b0296e4/benchmarks/dynamo/common.py#L540) (referenced by `check_accuracy` at L2201):

```python
@functools.cache
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""
    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator
        seed = 1337
        if HAS_CUDA and not torch.cuda._is_in_bad_fork():
            torch.cuda.manual_seed_all(seed)
        if HAS_XPU and not torch.xpu._is_in_bad_fork():
            torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)
    torch.manual_seed = deterministic_torch_manual_seed
```

The benchmark suite also keeps a `non_deterministic_models` set ([`common.py:1897`](https://github.com/pytorch/pytorch/blob/7a6f3270a85fefb5716d3224cf1936c07b0296e4/benchmarks/dynamo/common.py#L1897)) — models that remain non-deterministic *even with the seed patch*. For those, eager-vs-eager differences are accepted ("`eager_two_runs_differ`" downgraded to "pass").

**Required infra changes for our perf primitive (`discovery/perf.py`):**
1. Apply `patch_torch_manual_seed()` (or our equivalent) before any model load + before each measurement run.
2. Run an "eager_self_check": run eager twice with the same input and compare. If outputs diverge, the model is non-deterministic — flag in results JSON, don't fail compile-vs-eager check on bitwise grounds.
3. Maintain a corpus-level `non_deterministic_models` set in the eval config for known offenders (start empty; populate as we find them).

This applies to the perf-primitive **broadly**, not just DeepSeek V4 Pro — same hardening makes every existing per-case measurement more reliable.

**Numerics.** Bitwise + max_abs_diff in bf16 specifically — DeepSeek's training/inference is bf16-native so any compile-induced fp32-bf16 mismatch is real-world relevant. If model has multiple output heads (router logits, hidden states, final logits), report per-head separately so we can isolate where divergence enters.

## Output — AutoDev Kanban board + detailed writeup

**Source-of-truth artifact:** each phase produces a `results.json` under `experiments/results/deepseek_v4_pro/<phase>-<date>/`. The AutoDev board surfaces the work-in-progress state.

**AutoDev Kanban board:** [github.com/users/penguinwu/projects/1](https://github.com/users/penguinwu/projects/1) — the same board the discovery cases (#59, #61, #62, #63) live on. Per Peng 2026-04-25 13:32 ET. No new ETL or dashboard infra needed — just track this eval as GitHub issues that auto-attach to the board (project automation handles status moves). Proposed issue structure:

- **Umbrella:** `[Eval] DeepSeek V4 Pro` — top-level tracking issue with links to phase sub-issues + the plan file.
- **Phase 1 sub-issue:** `[Eval] DeepSeek V4 Pro — Phase 1 (tiny config, correctness)` — closes when Phase 1 results land.
- **Phase 2 sub-issue:** `[Eval] DeepSeek V4 Pro — Phase 2 (full model, performance)` — closes when Phase 2 results land.
- **Per-failure issues:** filed via the existing `tools/file_issues.py` pipeline for graph-break categories and correctness divergences. These auto-attach to the board too.

Board view = the live dashboard. Status moves: Backlog → Ready → In progress → Done (project automation: closing the issue moves the card to Done, same as we saw with #59 today).

**Detailed writeup (high-profile model — people will care):**

Per Peng 2026-04-25: this is a release the broader community will be interested in. The summary needs to be more thorough than a typical per-case finding. Peng will guide the writeup process directly. Skeleton:

1. **Headline** (3–5 bullets, fits in a tweet/Workplace post): pass/fail per dimension at a glance.
2. **Graph breaks** — total count, categorized by break shape (data-dep / op-not-supported / config-flag / other), code-line locations, and which are upstream-fixable (corpus issue category) vs model-design choices.
3. **Correctness (numerics)** — max_abs_diff distribution, bitwise_equal bool, per-dtype (fp32 + bf16). Call out any layer-specific divergence (router logits vs final logits) if model is composite. Compare eager-self-consistency (deterministic? if not, flag the model in the non_deterministic set).
4. **Performance** — tier-1 + tier-2 speedup, compile time, peak memory. Absolute numbers, no regression baseline. Phase 2 only.
5. **Methodology + caveats** — config used (tiny vs full), tolerance threshold, RNG seed, env (PyTorch / transformers versions).
6. **Failures filed** — links to corpus issues, with one-line on each.
7. **What this tells us** — short interpretation, in Peng's voice (drafted with her).

Output venues: corpus repo (`experiments/results/deepseek_v4_pro/<phase>/README.md`) + AutoDev Kanban board dashboard + (per Peng's call) Workplace post to PT2 working group / Compile Q&A.

Per-eval `results.json` schema (proposed):

```json
{
  "model_id": "deepseek-ai/DeepSeek-V4-Pro",
  "model_revision": "<HF commit sha>",
  "torch_version": "2.11.0+cu128",
  "transformers_version": "5.5.3",
  "device": "cuda",
  "rows": [
    {
      "backend": "inductor", "mode": "eval", "dtype": "bf16",
      "graph_break_count": 0, "compile_ok": true,
      "max_abs_diff_vs_eager": 1.49e-04, "bitwise_equal": false,
      "eager_ms_t1": 12.3, "compiled_ms_t1": 4.1, "speedup_t1": 3.0,
      "eager_ms_t2": 145.0, "compiled_ms_t2": 52.0, "speedup_t2": 2.79,
      "peak_mem_mb": 41203, "compile_s": 18.4
    }
  ]
}
```

## Resolved decisions (Peng 2026-04-25 13:22 ET)

| # | Question | Decision |
|---|---|---|
| 1 | "Kabana" instance | **AutoDev Kanban board** ([github.com/users/penguinwu/projects/1](https://github.com/users/penguinwu/projects/1)) — same board discovery cases live on. ("Kabana" = Kanban; I'd misread it as Kibana the viz tool.) Per Peng 2026-04-25 13:32 ET. |
| 2 | Tiny-config vs full-model | Two phases: tiny + correctness FIRST, then full-model + perf. See "Two-phase execution" section. |
| 3 | Comparable baseline | None. Absolute numbers only. |
| 4 | Tolerance per dtype | **1e-4** (default per upstream `pytorch/benchmarks/dynamo/common.py:1069`). |
| 5 | Filed vs dashboard | Detailed writeup of all three dimensions (graph breaks + correctness/numerics + performance). High-profile model — community-facing report. Peng will guide the summary process. |

**All open questions resolved.** Plan is fully unblocked for execution.

## Done means

**Phase 1 (tiny-config, correctness-first):**
- `results.json` landed at `experiments/results/deepseek_v4_pro/phase1-tiny-<date>/`.
- Graph break + correctness verdict captured per the schema above.
- Failures (gb>0, max_abs_diff > 1e-4) filed as corpus issues.
- Phase 1 writeup at `experiments/results/deepseek_v4_pro/phase1-tiny-<date>/README.md`.

**Phase 2 (full-model, performance):** (gated on Phase 1 correctness)
- `results.json` landed at `experiments/results/deepseek_v4_pro/phase2-full-<date>/` with perf rows + correctness reconfirmation at scale.
- Tier-1 + tier-2 numbers captured.
- Phase 2 writeup at `experiments/results/deepseek_v4_pro/phase2-full-<date>/README.md`.

**Both phases:**
- Umbrella + sub-issues tracked on AutoDev board ([projects/1](https://github.com/users/penguinwu/projects/1)); cards auto-move to Done on issue close.
- Combined detailed writeup ready for Peng-guided community summary (per Resolved Decision #5).

## Stop conditions

- *Model won't load on devvm (memory):* fall back to tiny-config or pause and re-plan.
- *Eager itself fails* (model bug, transformers compatibility): file the eager failure as a transformers issue; don't attempt compile until eager works.
- *Compile crashes* (e.g. inductor codegen error): file as a corpus issue with the crash trace; continue with other dimensions where possible.

## Out of scope (for this eval)

- *Training-mode evaluation* — inference-only here. Training adds backward-graph + DDP/FSDP complexity that warrants its own eval.
- *Multi-GPU / sharded inference* — 1× H100 only for v1. Sharded inference is a separate workstream.
- *Latency under serving load* (vLLM, TensorRT) — different stack, different eval.
- *Skill / agent evaluation* — that's WS1; this plan is the standard eval flow.

## Execution shape

**Pre-flight (both phases):**
1. Read HF model card + config; fill in the "What we know about the model" checklist above.
2. Land RNG-determinism patch + `non_deterministic_models` set in `discovery/perf.py` (per the "RNG determinism" subsection).
3. Create the umbrella issue + Phase 1 + Phase 2 sub-issues on the AutoDev board (use `tools/queue_task.py` or hand-create with the existing per-case template style).

**Phase 1 — Tiny-config, correctness-first:**
4. Identify smallest viable config (HF dev-config, or hand-shrink the standard config to test-shape).
5. Verify model loads in eager mode on devvm.
6. Author `experiments/configs/deepseek-v4-pro-phase1.json`.
7. Run graph break check (`fullgraph=True`) — fast even if compile fails.
8. Run correctness check (`compiled vs eager`, default backend; fp32 first, then bf16).
9. Aggregate to Phase 1 `results.json`.
10. File graph-break and correctness failures as corpus issues.
11. Write Phase 1 `README.md`.
12. **Sign-off gate:** Peng reviews Phase 1 before Phase 2 launches.

**Phase 2 — Full-model, performance:**
13. Author `experiments/configs/deepseek-v4-pro-phase2.json` (full config; sharded if infra exists).
14. Re-run correctness at scale (cheap sanity check before perf).
15. Run perf check (`measure_perf` tier-1 + tier-2).
16. Aggregate to Phase 2 `results.json`; close the Phase 2 sub-issue (auto-moves the board card to Done).
17. Write Phase 2 `README.md`.
18. Combined community-facing summary (Peng-guided per Resolved Decision #5).

## Revision log

- *2026-04-25 13:15 ET:* Plan drafted from Peng's directive. 5 open questions raised.
- *2026-04-25 13:18 ET:* Folded in two new pointers from Peng — `pytorch/pytorch/benchmarks` at sha 7a6f3270 (HF benchmark suite reference) + the deterministic RNG seed pattern at `common.py:540` (Animesh's note on HF model non-determinism). Scope expanded: hardening `discovery/perf.py` is part of this work, not just DeepSeek-specific.
- *2026-04-25 13:22 ET:* Peng answered the 5 open questions. Two-phase execution adopted (tiny+correctness → full+perf). Tolerance pinned at 1e-4 (upstream default). No comparable baseline. Detailed writeup required for community release. Status moved from `draft` to `active`. One follow-up pending: "Kabana" instance.
- *2026-04-25 13:32 ET:* "Kabana" resolved — it's the **AutoDev Kanban board** ([projects/1](https://github.com/users/penguinwu/projects/1)), the same board discovery cases live on. (I'd been parsing it as "Kibana" the Elasticsearch viz tool — wrong.) No new dashboard ETL required; track via GitHub issues that auto-attach to the board. Plan is fully unblocked.
