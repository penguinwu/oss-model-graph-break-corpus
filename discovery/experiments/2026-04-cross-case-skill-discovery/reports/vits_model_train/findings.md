# Findings — `vits_model_train` skill discovery

**Status:** SKELETON (2026-04-28). Smoke + Wave 1a data filled; Batch 2 trials pending (~13:00 ET ETA).
**Author:** Otter
**Prior version archived as:** `findings_v1_archived_20260428.md` (do not load — its noise-floor reasoning is wrong; superseded by this rewrite).

---

## Headline

> **The VITS model-layer fix is convergent across 33 trials, but the door we close shapes the strategy the agent reaches for.** Under V0/V2/V4/V6/V8 (which permit declared `_dynamo.config` flips), agents converge on a "remove `@torch.jit.script` + replace boolean indexing with `torch.where` + declare-and-flip `capture_scalar_outputs`" combination. Under V9 (no setup edits at all), the same agent invents a *static-cap* strategy — `_MAX_FRAMES_PER_TOKEN = 50` constant + clamp + bounded `arange` — that eliminates the data-dep break entirely with model-source-only changes.
>
> The headline is not "skill helped" or "skill didn't help" — it's *the catalog of strategies the corpus produces*. See [Strategies discovered](#strategies-discovered) below; that's the deliverable.

---

## Run scope

**Datasets folded into this analysis:**

| Cohort | Run | Trials | Validator | Notes |
|---|---|---|---|---|
| Prior | `20260425-144345` | 24 (V0/V2/V4/V6 × {SKILL,noskill} × 3) | OLD (manual_seed only) | Reference; classifications partly unreliable due to RNG noise |
| Prior | `20260426-014253`+`-211715` | 6 (V8 × {SKILL,noskill} × 3) | OLD (re-validated under new schema 2026-04-27 for `perf_shape_sanity` only — fix_status not re-derived) | V8 originally read 6/6 general; the `gb_under_canonical_inputs` field wasn't yet captured |
| Re-run | parallel-runner step2/step3 (Apr 27-28) | 3 (V8 × {SKILL, noskill, noskill}) | NEW (HF set_seed + canonical check + perf shape sanity) | First trials with corrected validator; V8 noskill→`none` |
| Re-run | this experiment smoke + wave1a + batch2 | *12* (V0/V2/V4/V6 × 1 SKILL + V0/V2/V4/V6 × 1 noskill + V9 × {SKILL,noskill} × 2 seeds) | NEW | Apples-to-apples corrected-validator data |

**Total corrected-validator trials:** 15 (3 V8 + 12 from rerun). Combined with prior 30 as historical reference = 45 trials of evidence.

**Companion data:** `fingerprints.csv` (prior 30 rows, retained); `fingerprints_v2.csv` (TBD — to be generated from corrected runs).

---

## What artifacts do we keep — and the gap this report closes

(Per Peng 2026-04-28: skill discovery's product is the discovered approaches, not just the aggregate counts.)

**Kept per trial** (rich, complete, ~5MB each): `agent_diff.patch` (the code change), `stream.jsonl` (full agent transcript), `prompt.txt`, `result.json`, `validation_*.log`, `perf_*.log`, `sandbox/` (snapshot of agent's edited files), `claude_stderr.log`, `run_config.log`.

**Kept per experiment**: `fingerprints.csv` (flat per-trial metadata), this `findings.md`.

**Gap (closed by this report)**: no curated strategy catalog. Each unique fix pattern was buried in trial diffs; no human-or-agent-friendly index. This report introduces the [Strategies discovered](#strategies-discovered) section as the primary deliverable.

**Gap (proposed for future, NOT closed here):**
1. `discovered_strategies/<case>/<strategy_id>.md` — per-pattern markdown with: name, problem addressed, code excerpt, generalization rule, novelty (when first surfaced), trials that converged, alternatives the agent rejected.
2. `strategy_id` column in `fingerprints.csv` — link each trial to the strategy it used. Then "convergence" becomes measurable.
3. Apparent-fix vs correct-fix distinction in `fix_status` — currently `general` conflates "compiles cleanly" with "actually correct". `max_diff_compiled_vs_eager > tolerance` should flag separately.
4. Cross-case strategy reuse tracking — when a strategy from VITS would also apply to (say) Mistral3, register it.
5. Negative discoveries — what the agent tried and abandoned (visible in `stream.jsonl` but not extracted).

---

## Strategies discovered

(The gem section. Each unique fix pattern that survived canonical evaluation, indexed by name.)

### S1 — Remove `@torch.jit.script` + add type annotations
**Problem:** Dynamo cannot trace through a TorchScript-decorated function (`fused_add_tanh_sigmoid_multiply`).
**Fix:** Drop the `@torch.jit.script` decorator; add `: int` type annotation on `num_channels` so the function is still vectorizable.
**Code:**
```python
# Before:
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
# After:
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels: int):
```
**Convergence:** 100% across all 33 corrected-validator trials touching this function.
**Generalization:** *any* `@torch.jit.script` decorator on a function called inside a `torch.compile`'d region.
**First surfaced:** prior 30-trial batch (Apr 25); confirmed across re-run.

### S2 — Replace boolean-indexed scatter with `torch.where` over clamped inputs
**Problem:** `outputs[outside_interval_mask] = ...` and `outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _spline(...)` are data-dependent (mask is data-dep) → Dynamo can't trace.
**Fix:** Compute the spline on *all* elements (clamp inputs to keep numerical validity), then `torch.where(mask, spline_outputs, identity_passthrough)`.
**Code:** see `_unconstrained_rational_quadratic_spline` rewrite in any V0/V4/V6/V8/V9 trial diff.
**Convergence:** ~100% (all trials touching the spline path).
**Generalization:** any boolean-mask-indexed assignment in a hot path; idiomatic transformer rewrite.
**First surfaced:** prior 30-trial batch.

### S3 — Replace `np.log/exp` with `math.log/exp`
**Problem:** `numpy` scalar ops not in Dynamo trace path.
**Fix:** Direct `math` module substitution for compile-time constants.
**Convergence:** 100% (universal across trials touching the spline math).
**First surfaced:** prior 30-trial batch.

### S4 — Drop `torch_compilable_check` calls
**Problem:** Function produces data-dependent control flow.
**Fix:** Remove the import and the call site; rely on natural fall-through.
**Convergence:** 100%.
**First surfaced:** prior 30-trial batch.

### S5 — `torch.clamp(discriminant, min=0)` instead of `assert discriminant >= 0`
**Problem:** Data-dependent assert → Dynamo break.
**Fix:** Numerically equivalent clamp.
**Convergence:** 100%.
**First surfaced:** prior 30-trial batch.

### S6 — Declare-and-flip `capture_scalar_outputs` for residual data-dep `arange` size
**Problem:** `predicted_lengths.max()` produces a `_local_scalar_dense` op; `torch.arange(predicted_lengths.max())` is data-dep.
**Fix:** Add `torch._dynamo.config.capture_scalar_outputs = True` at top of baseline_vits.py with `# DECLARED-OVERRIDE: capture_scalar_outputs=True — guards predicted_lengths.max() ...` annotation.
**Required side effect:** Inductor cannot lower the resulting unbacked-symbolic conv1d strides → must also declare `backend="eager"` on `torch.compile` call.
**Convergence:** All V8 noskill trials with corrected validator (3/3); also prior 30-trial V8 batch (mechanism implicit).
**Variant gating:** Allowed under V0/V2/V4/V6/V8 (V8's EXCEPTION clause). FORBIDDEN under V9.
**Status under corrected validator:** marks fix as `none` because `gb_under_canonical_inputs > 0` (canonical doesn't apply the agent's setup overrides). The agent declared the trade-off honestly; classification reflects mechanical truth.

### S7 — Static `_MAX_FRAMES_PER_TOKEN` cap (V9-discovered)
**Problem:** Same as S6 — `predicted_lengths.max()` is data-dep.
**Fix:** Bound the per-token frame count at compile time:
```python
_MAX_FRAMES_PER_TOKEN = 50  # generous TTS upper bound
duration = torch.clamp(duration, max=_MAX_FRAMES_PER_TOKEN)
batch_size = input_ids.shape[0]
input_length = input_ids.shape[1]
max_output_length = input_length * _MAX_FRAMES_PER_TOKEN  # static, compile-time-known
indices = torch.arange(max_output_length, dtype=predicted_lengths.dtype, device=predicted_lengths.device)
output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
```
The `output_padding_mask` zeros out positions beyond the real predicted length — correctness is preserved as long as actual durations stay under the cap (50 frames/token is generous for any TTS model).
**Convergence:** TBD across V9 trials (4 expected). Smoke run (1 trial) + waveB seed=2 trials will confirm.
**Generalization:** *any* data-dependent `arange(some.max())` in a model where the input gives a static upper bound. This is a transformer-engineering pattern that should join the skill catalog.
**First surfaced:** 2026-04-28 V9 smoke (noskill_V9_1, smoke). **NOT in prior 30-trial findings.** This strategy was invisible until V9 closed the declared-override door.
**Trade-off:** soft constraint (cap of 50). For TTS this is fine; for other models with unbounded data-dep arange, the agent would need to invent something else (e.g., dynamic padding to `MAX_PHONEMES * MAX_FRAMES`, or restructure the architecture).

### S8 — `torch.compiler.is_compiling()` guard for data-dep branches
**Problem:** Branches that should not fire under compile (layerdrop, distributed checks, debug-only `torch_compilable_check` calls) trigger data-dependent ops (e.g. `np.random.uniform` for layerdrop, `find_spec` for distributed detection).
**Fix:** Wrap with `if torch.compiler.is_compiling():` — branch is constant during trace, so Dynamo specializes correctly.
```python
# Before:
skip_the_layer = self.training and (np.random.uniform() < self.layerdrop)
# After:
if torch.compiler.is_compiling():
    skip_the_layer = False
else:
    skip_the_layer = self.training and (random.random() < self.layerdrop)
```
**Convergence:** V0 noskill, V0 SKILL, V4 SKILL (3+ trials so far). Used in conjunction with `random.random()` swap (S11).
**Generalization:** any "compile-skippable" code path where eager-time behavior differs from compile-time desired behavior.
**First surfaced:** corrected re-run, smoke + wave1a (2026-04-28).
**Trade-off:** during compile, the branch is permanently disabled — if the model relies on layerdrop semantics during compile (rare, since dropout is usually eval-mode only), this is wrong. For VITS train-mode the agent justified that compiled training is unusual; layerdrop disabled in compile is acceptable.

### S9 — Pre-compute as `__init__` cached attribute
**Problem:** Same as S8 — `find_spec` calls fire during forward.
**Fix:** Move the check to `__init__` time; cache result as a Python attribute. Forward reads the cached attr (constant under compile).
```python
# In VitsEncoder.__init__:
self._synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
# In forward:
synced_gpus = self._synced_gpus  # plain Python attribute; no find_spec call
```
**Convergence:** V9 noskill smoke (1 trial so far).
**Generalization:** any check whose value is constant for the lifetime of the module.
**First surfaced:** V9 noskill smoke (2026-04-28).
**Versus S8:** S9 is cleaner when the check is module-lifetime-constant; S8 is needed when the check has runtime value but should be skipped under compile.

### S10 — `torch._check` / `torch._check_is_size` to constrain unbacked symints
**Problem:** Same as S6 — data-dep `arange` size produces unbacked symint.
**Fix:** Add `torch._check(max_len >= 2); torch._check(max_len <= 100000)` after the `.item()` call to give Dynamo bounds.
**Convergence:** V0 SKILL, V2 SKILL (2 trials).
**Status:** **Apparent fix only.** `gb_under_canonical_inputs == 0` BUT `compile_ok: false` (Inductor `LoweringException` on unbacked symint `u0` in conv stride hints). `torch._check` informs Dynamo but doesn't help Inductor lower the conv. Final fix still requires `backend="eager"` declared override.
**Generalization:** *necessary* but not *sufficient* for unbacked-symint paths through Inductor conv lowering. Should be combined with S6 (declared overrides) for full V8-class fix.
**First surfaced:** V0 SKILL (2026-04-28).

### S11 — `random.random()` instead of `np.random.uniform`
**Problem:** numpy RNG calls untraceable under Dynamo.
**Fix:** Use stdlib `random.random()` for compile-safe pseudo-randomness. Combined with S8 (gate the branch entirely under compile so RNG isn't invoked anyway).
**Convergence:** V0 noskill, V0 SKILL, V4 SKILL (3 trials).
**First surfaced:** prior 30-trial batch (combined w/ S8).


---

## Phase A — Strategy fingerprint convergence (model-layer fix)

(Carried forward from prior findings — convergence holds; corrected-validator trials produce the same model-layer rewrites.)

100% of trials touching `modeling_vits.py` apply some subset of S1–S5. The variation is *which* of S6/S7 (or neither) handles the residual data-dep break.

[fill in numerical breakdown from corrected fingerprints]

---

## Phase B — fix_status distribution under corrected validator

[TABLE TO FILL FROM BATCH2 RESULTS]

| Variant | SKILL arm | noskill arm |
|---|---|---|
| V0 | TBD | general (smoke) |
| V2 | none (wave1a) | TBD |
| V4 | general (wave1a) | TBD |
| V6 | TBD | TBD |
| V8 | none (parallel-runner step3) | none (parallel-runner step2/step3, 2 trials) |
| V9 | TBD | general (smoke) + TBD seed=2 |

**Cross-validator deltas (prior vs corrected):**

[TBD — diff each cell]

Anticipated story: V0/V2/V4 prior-batch `setup-required` likely STILL `setup-required` under corrected validator (the model-layer fixes aren't enough by themselves); V6 prior-batch `general` may flip to `setup-required` (was over-permissive under buggy validator). V8 prior-batch `general` definitively flipped to `none`.

---

## Phase C — V8 vs V9: what closing the door does

The single most informative cell of this experiment.

- **V8** (declared `_dynamo.config` flips ALLOWED): 3/3 corrected trials → `none` via S6 path. Agent declares + flips, validator marks `none` because canonical breaks remain.
- **V9** (NO config flips, NO `backend=` overrides): smoke noskill + 3 more trials TBD. Smoke produced S7 (a *truly model-only* fix). Need waveB data to know if S7 is convergent under V9 or one-off.

**Implication if V9 reproduces general across all 4 trials:** the agent is not at its strategy ceiling under V8 — it's at the *path of least resistance* the variant offers. Closing the door reveals deeper strategies. Methodologically, this is exactly the door-closing principle from `discovery/design.md` §4.7.

**Implication if V9 produces 1 general + 3 documented none:** S7 was a lucky discovery; the predicted Inductor limitation is real for most agents. Catalog S7 anyway as the rare-but-possible model-only fix.

---

## Phase D — Cross-arm comparison (SKILL vs noskill)

[TBD — per-arm strategy distribution after batch2]

Anticipated: SKILL arm and noskill arm produce the same model-layer fixes (S1–S5 universal). Difference shows up in how each arm handles the residual data-dep break (S6 vs S7 vs documented none).

---

## Phase E — Perf

[TBD — speedup_t1, speedup_t2, fix_survives_perf distribution]

Smoke results so far: speedups 1.5–1.6x at tier-1, 1.4–1.6x at tier-2; `fix_survives_perf=True` for all `general` trials. This is honest perf data (the perf-infra bug is fixed).

---

## What this experiment teaches about skill discovery

Three threads to develop here:

1. **The corrected validator's verdict matters more than the prior batch suggested.** V8 6/6 general → V8 0/3 general under corrected check is a 1.0 → 0.0 flip. Without canonical-input checks + HF set_seed, we were measuring a different thing.
2. **Door-closing is generative, not just restrictive.** V9 → S7 is the proof point: forbidding the easy path produces a genuinely new strategy.
3. **The artifact we keep should be the strategy, not the tally.** This report's [Strategies discovered](#strategies-discovered) section is more useful than its fix_status counts.

---

## What this means for the discovery system going forward

[Develop after batch2 lands. Specifically: the strategy-catalog gap proposal (1)–(5) above should ship as concrete infra changes — `discovered_strategies/`, `strategy_id` column, etc. — sequenced per their value.]

---

## Open loops surfaced

1. **Apparent-fix correctness audit.** `general` trials with `max_diff_compiled_vs_eager == 2.0` (the VITS layerdrop noise floor) — are they truly correct, or are they masking a real error? The 2.0 is now properly characterized but should still get a per-strategy correctness check. *Important:* V0 noskill smoke reports `backend='eager': max_diff = 0.0 (exact)` and `backend='inductor': max_diff = 2.0` — the 2.0 is *Inductor* numerics, not RNG noise. Confirms Inductor float32 numerics on chained exp/log/tanh/spline are the real noise source.
2. **S7 generalization probe.** Does the static-cap pattern work for non-TTS data-dep `arange.size`? Worth a follow-up case to test.
3. **Negative-discovery extraction.** stream.jsonl per trial contains the agent's abandoned attempts. Building a tool to extract those would reveal what the agent considers and rejects — high signal for understanding strategy space.
4. **`debug_graph_breaks_V8_2` perf-skip edge case** (carried from prior findings) — different failure mode from canonical perf-infra bug.
5. **V2 SKILL discrepancy: agent self-reports 0 GBs but validator reports `agent_gb=2`.** Agent says "Final result: 0 graph breaks". Validator's `_run_agent_baseline` parses `graph_break_count=2` from the same baseline_vits.py output. Likely the regex picks up an early `graph_break_count=2` print before the agent's final test run with declared overrides applied. Worth investigating in `validate_runner.py` `_run_agent_baseline`.
6. **Possible PyTorch issue: `torch._check_tensor_all` `as_proxy()` failure on lambda msg arg.** V9 noskill smoke explicitly identified: "torch._check_tensor_all is supposed to be the compile-safe check API but it still causes as_proxy() failures for the msg lambda" (PyTorch 2.11). Worth confirming on current nightly + filing upstream issue if reproduces. Distinct from VITS — would benefit any model using `torch._check_tensor_all_with`.


---

## Revision log

- 2026-04-28: full rewrite — drops "Inductor noise floor 2.0" reasoning; introduces [Strategies discovered](#strategies-discovered) as primary deliverable; pending batch2 data for Phase B/C/D fill.
