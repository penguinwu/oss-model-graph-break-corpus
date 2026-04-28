# Field report — `vits_model_train` skill discovery (one model, many trials)

**Status:** Field report (not analytics). This is what we observed in 15 discovery-agent trials against one model. We do not generalize beyond this case — N=1 model, every observation is anecdotal.
**Author:** Otter
**Date:** 2026-04-28
**Prior version archived as:** `findings_v1_archived_20260428.md` (do not load — superseded).

> **What this report is:** a field log of (a) what graph-break shapes the agent encountered, (b) which fix strategies worked under canonical evaluation and which were shortcuts, (c) concrete agent behaviors we observed. Plus methodology + caveats so you can judge the validity of the observations yourself.
>
> **What this report is NOT:** a statistical claim about "skills work" or "skills don't work" — we have data from one model at one moment. The strategies are the durable artifact; the per-variant counts are anecdotes from this experiment.

---

## 1. Methodology — how do we know a fix is real?

A trial is one (case, variant, skill-arm, seed) run of a discovery agent against a single model. The agent gets a prompt describing the model + graph-break symptoms, can read/edit the model source + its own baseline script, and runs `python validate.py` to check progress. When the agent says it's done, we run our own validator against the agent's edits.

The validator does three things, in order. A fix is "real" only if all three hold.

**(a) Canonical-input check.** Load the model with stock test inputs the agent didn't choose. Run `torch.compile(...)` *without* applying any of the agent's setup overrides — no `torch._dynamo.config` flips, no `backend=` swap from default `inductor`. Count graph breaks via the corpus-canonical TORCH_LOGS-based analysis (`sweep.explain.run_graph_break_analysis`). The integer this produces is `gb_under_canonical_inputs`. If it's 0, the agent's edits are general — they survive when tested as a downstream user would test them. If it's > 0, the agent's edits depend on a setup hack that the canonical check refuses to apply.

**(b) RNG determinism.** All seeding goes through HuggingFace `set_seed(0)` (not just `torch.manual_seed`), which covers torch + numpy + Python `random`. Without this, models that use `np.random` in forward (like VITS layerdrop) produce non-bit-identical outputs across two eager forwards, and the eager-vs-compile diff becomes meaningless. Verified empirically — pre-fix, 3-6 of 7 forwards matched; post-fix, all 7 are bit-identical.

**(c) Perf shape sanity.** Run the compiled model at *realistic* input shapes (B=2, seq_len=64 for VITS) — distinct from the tier-1 (fast) shapes the validator uses for the canonical check. If the model raises `RuntimeError` at perf shapes, the fix is shape-fragile (worked at B=1 but breaks at real workloads), so we mark `fix_survives_perf: False`.

**`fix_status` levels:**
- `general` — `gb_under_canonical_inputs == 0` AND no `setup-required` shortcut
- `setup-required` — agent's own run shows 0 GBs, but only because the agent's setup overrides were applied. The canonical check (without those overrides) sees > 0 GBs.
- `none` — agent could not reach 0 GBs even in their own run, OR their fix shape was caught by the canonical check.
- `unknown` — validator failed (rare, e.g. agent broke `import` of the model)

**A note on the V8 variant.** V8 explicitly *allows* the agent to declare `torch._dynamo.config` overrides + `backend="eager"` in their baseline script as long as they justify each. This is a deliberate design choice — we wanted to see what fix the agent reaches for if the easy path is allowed but acknowledged. Under the corrected validator, V8 fixes show up as `none` because the canonical check doesn't apply the agent's declared overrides. The classification is mechanical, not a judgment.

**A note on `max_diff`.** When `gb_under_canonical_inputs == 0`, we compare compile output to eager output and record `max_diff_compiled_vs_eager`. We see ~2.0 max-diff on some trials' compiled output even at `general` — this is Inductor's float32 numerics on chained `exp / log / tanh / spline` operations. Verified by running the same model at `backend="eager"` post-fix → max_diff drops to 0.0 exact. So the diff is from the *backend*, not the agent's edits. `fix_survives_perf` does not currently flag this; we treat it as a separate open loop (correctness audit).

---

## 2. The graph-break shapes we attacked (VITS train mode)

VITS in train mode has 5 distinct break shapes. The agent attacked all of them in each trial; we use shorthand below. (See `discovery/cases/vits_model_train.py` for the source-of-truth definitions.)

**B1 — `find_spec` skipped function.** `is_deepspeed_zero3_enabled()` and `is_fsdp_managed_module(self)` call `importlib.util.find_spec(...)` which Dynamo marks as a skipped function → graph break in `transformers/utils/import_utils.py`.

**B2 — `np.random.uniform` data-dep guard.** The layerdrop path in `VitsResidualCouplingBlock` does `dropout_probability = np.random.uniform(0, 1); skip_the_layer = self.training and (dropout_probability < self.layerdrop)`. The `np.random` call is untraceable; the resulting `if skip_the_layer:` branch is data-dependent. Fires only in train mode (gated by `self.training`).

**B3 — `as_proxy()` failure on `ValueError` arg.** `VitsResidualCouplingLayer` uses `torch_compilable_check(...)` (a transformers utility that's an enriched `assert`) which constructs `ValueError(f"...{tensor.var()}...")` — Dynamo can't proxy a `ValueError` initialized with a tensor-derived f-string. Fires in the duration_predictor stochastic flow.

**B4 — `callable()` builtin un-traceable.** `transformers/utils/import_utils.py:1487` calls `callable(module)` on a `StringFormatVariable` (the lazy module proxy). Dynamo can't trace `callable()` on this variable type.

**B5 — `F.conv1d` data-dep guard.** The flow + decoder path uses `torch.arange(predicted_lengths.max())` followed by an `output_padding_mask` that feeds into a transposed convolution. The `predicted_lengths.max().item()` produces an unbacked symint `u0`, and the conv stride hint check `u0 < 1` is data-dependent (`torch/nn/modules/conv.py:370`).

These are the 5 shapes. Every trial sees the same set. Variant V0..V9 vary the *constraints* on what the agent may do to fix them, not what shapes appear.

---

## 3. Working strategies (real fixes that survive canonical evaluation)

For each strategy: what break shape it addresses, what the code does, why it survives canonical, where we saw it. No counts, no convergence percentages — these are the strategies the agents wrote, observed once or more in this experiment.

### S1 — Drop `@torch.jit.script` decorator

*Addresses:* B-other (a non-data-dep break: `fused_add_tanh_sigmoid_multiply` is JIT-scripted, Dynamo can't trace through it).
*What it does:*
```python
# Before
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
# After
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels: int):
```
*Why real:* removes the TorchScript-compiled wrapper; Dynamo now traces the plain Python function. The `: int` annotation preserves the function's vectorizability for downstream TorchScript users (none in our path, but harmless).
*Where seen:* every trial in the experiment (every diff includes this). The agent reaches for it without prompting.

### S2 — Replace boolean-mask scatter with `torch.where` over clamped inputs

*Addresses:* B-other (a non-data-dep break: `outputs[outside_interval_mask] = inputs[outside_interval_mask]` is boolean-mask-indexed assignment, Dynamo can't trace).
*What it does:*
```python
# Before
outputs = torch.zeros_like(inputs)
log_abs_det = torch.zeros_like(inputs)
outputs[outside_interval_mask] = inputs[outside_interval_mask]
outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _spline(
    inputs=inputs[inside_interval_mask], ...)
# After
clamped_inputs = torch.clamp(inputs, -tail_bound, tail_bound)
spline_outputs, spline_log_abs_det = _spline(inputs=clamped_inputs, ...)
outputs = torch.where(inside_interval_mask, spline_outputs, inputs)
log_abs_det = torch.where(inside_interval_mask, spline_log_abs_det, torch.zeros_like(inputs))
```
*Why real:* compute the spline on *all* elements (`clamp` keeps numerical validity outside the interval), then use `torch.where` to select. No data-dep indexing. Mathematically equivalent within the spline's defined domain.
*Where seen:* most trials.

### S3 — Replace `np.log/exp` with `math.log/exp`

*Addresses:* B-other (numpy scalar ops not in Dynamo trace path; e.g. `constant = np.log(np.exp(1 - min_derivative) - 1)`).
*What it does:* substitute Python `math` module functions for the same compile-time-constant computation.
*Why real:* `math.log/exp` are pure-Python builtins on Python floats; they evaluate at Python interpretation time, before Dynamo traces. The result is a constant by the time the trace runs.
*Where seen:* most trials.

### S4 — Drop `torch_compilable_check` calls

*Addresses:* B3 (the wrapped `assert` constructs an un-proxy-able `ValueError`).
*What it does:* remove the import and the call sites; rely on natural fall-through. The check was a defensive assertion (e.g. "discriminant must be non-negative") — its constructor was the un-traceable part, not the check itself.
*Why real:* the safety property the assertion was protecting (e.g. "no negative discriminant") is preserved by other means (S5 below clamps the discriminant; the spline rewrite via S2 handles out-of-range inputs).
*Where seen:* most trials.

### S5 — `clamp(discriminant, min=0)` instead of `assert discriminant >= 0`

*Addresses:* a specific instance of B3 (assertion in `_rational_quadratic_spline` that discriminant of the quadratic must be non-negative for the `sqrt`).
*What it does:* `discriminant = discriminant.clamp(min=0.0)` before the sqrt call.
*Why real:* the assertion was protecting `sqrt` from negatives. The clamp makes the protection numerical (tiny negative values from float roundoff get pushed to 0) instead of an assert. Identical observable behavior in the well-conditioned case; preferable in the edge case (no NaN propagation).
*Where seen:* sometimes — when the agent traces deep into the spline math. Other trials skip directly past via S4 alone.

### S7 — Static cap on data-dep `arange.size`

*Addresses:* B5 (`torch.arange(predicted_lengths.max())` is data-dep).
*What it does:*
```python
# Add a static upper bound:
_MAX_FRAMES_PER_TOKEN = 50  # generous TTS upper bound
duration = duration.clamp(max=_MAX_FRAMES_PER_TOKEN)
max_output_length = input_length * _MAX_FRAMES_PER_TOKEN  # static, compile-time-known
indices = torch.arange(max_output_length, dtype=predicted_lengths.dtype, device=predicted_lengths.device)
output_padding_mask = indices.unsqueeze(0) < predicted_lengths.unsqueeze(1)
```
The mask zeros out positions beyond the real predicted length, so unused slots don't contribute to the conv output. The conv's stride hint is now a Python int (`max_output_length`), not an unbacked symint.
*Why real:* the data-dependency is *removed* — the `.max()` call is replaced by a compile-time constant. Inductor can lower the conv normally. The cap of 50 frames per token is a soft constraint (TTS models rarely produce more); within the cap, output is bit-identical to eager.
*Variant note:* one trial (V6 noskill parallel) did the same pattern *inline*, without naming a constant: `max_out = input_length * 128` instead of `_MAX_FRAMES_PER_TOKEN = 50`. Same idea, different surface form.
*Where seen:* V4 noskill parallel, V6 noskill parallel (inline form), V9 noskill smoke s1, V9 noskill waveB s2. Not seen in any SKILL trial.

### S8 — `torch.compiler.is_compiling()` guard for compile-skippable branches

*Addresses:* B2 (the `np.random.uniform` layerdrop path).
*What it does:*
```python
# Before
dropout_probability = np.random.uniform(0, 1)
skip_the_layer = self.training and (dropout_probability < self.layerdrop)
# After
if torch.compiler.is_compiling():
    skip_the_layer = False
else:
    skip_the_layer = self.training and (random.random() < self.layerdrop)
```
*Why real:* `torch.compiler.is_compiling()` is a constant during trace (always `True` inside `torch.compile`), so Dynamo specializes the branch and removes it. Eager-mode behavior is preserved. The agent justified that VITS doesn't strictly need layerdrop during compile (it's a regularization technique normally only active during true training, which isn't this case).
*Where seen:* multiple trials, often paired with S11.

### S9 — Pre-compute `find_spec`-like checks in `__init__`

*Addresses:* B1.
*What it does:*
```python
# In VitsEncoder.__init__
self._synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
# In forward
synced_gpus = self._synced_gpus  # plain Python attribute
```
*Why real:* the `find_spec` call happens at module construction time, not in the forward — so it's never on the trace path. The forward reads a cached attribute, which Dynamo treats as a Python constant for the life of the module.
*Where seen:* one trial (V9 noskill smoke).

### S11 — `random.random()` instead of `np.random.uniform`

*Addresses:* B2 (RNG part).
*What it does:* use Python stdlib `random.random()` instead of numpy's RNG. Combined with S8 (the guard) so the call doesn't actually happen under compile, but reachable in eager.
*Why real:* `random.random()` returns a Python float; Dynamo handles it cleanly. (numpy's RNG calls a C extension Dynamo doesn't trace.) Behaviorally equivalent to numpy for layerdrop's purpose.
*Where seen:* multiple trials, paired with S8.

---

## 4. Shortcut strategies (look like fixes in the agent's run, fail canonical)

These are strategies the agent applies that produce a clean compile in *the agent's own baseline script* but don't survive when the canonical check refuses to apply the agent's setup overrides. We document them because (a) recognizing the shortcut pattern matters for prompt design, and (b) some of them might be acceptable in production with declaration; we just don't credit them as `general`.

### S6 — Declare-and-flip `_dynamo.config.capture_scalar_outputs`

*Addresses:* B5 (data-dep `arange` size).
*What the agent writes:*
```python
# In baseline_vits.py:
import torch._dynamo
torch._dynamo.config.capture_scalar_outputs = True  # DECLARED-OVERRIDE: guards predicted_lengths.max()
# Required side effect: also override backend=
compiled = torch.compile(model, backend="eager")  # DECLARED: Inductor can't lower unbacked u0 stride
```
*Why a shortcut:* the agent didn't change the model code — they changed how the *user* should configure `torch.compile` to handle the model. The canonical validator runs `torch.compile(model)` with default config + default backend (`inductor`). It does *not* read the agent's baseline script. So the agent's flips don't apply at canonical-check time, the data-dep `.max().item()` produces an unbacked symint, Inductor fails to lower the conv, and `gb_under_canonical_inputs` reports the residual data-dep break.
*This isn't necessarily wrong for production:* a user willing to set `capture_scalar_outputs=True` and `backend="eager"` *can* run this model. But it's not a model-only fix. For SkillsWatch evaluation, it's not a transferable strategy — every model that hits this break would need the same per-call config.
*How to steer away:* the V9 variant prompt forbids any setup-script edits. The agent under V9 can't write S6; if they're going to fix B5, it has to be in the model code (→ S7).
*Where seen:* every V8 trial (variant explicitly allows declared overrides) + V0/V2/V4 SKILL (variant allows the override implicitly; SKILL arm reaches for it first).

### S10 — `torch._check` / `torch._check_is_size` to constrain unbacked symints

*Addresses:* B5 (informs Dynamo of a bound on the unbacked symint).
*What the agent writes:*
```python
max_len = predicted_lengths.max().item()
torch._check(max_len >= 2)
torch._check(max_len <= 100000)
indices = torch.arange(max_len, ...)
```
*Why a shortcut (in this case):* `torch._check` informs Dynamo of bounds, so the Dynamo trace passes (`gb_under_canonical_inputs == 0` reported). But Inductor still fails to lower the resulting conv with unbacked-stride hints — `compile_ok: False` with `LoweringException`. So the *trace* succeeds, but *execution* fails. The agent's own baseline script paired this with `backend="eager"` to bypass Inductor — same trade-off as S6.
*Note:* `torch._check` is the *correct* API for sound Dynamo tracing in many contexts. It's a "shortcut" *here* only because Inductor's lowering doesn't complete; in models where Inductor can lower the resulting graph, `torch._check` is a real fix. Categorization is per-context.
*How to steer away:* not always desired — for cases where Inductor *can* lower, S10 is preferable to S7. Useful prompt guidance might be: "after applying `torch._check`, verify both Dynamo trace AND Inductor lowering succeed; if Inductor fails, the fix is incomplete."
*Where seen:* V0 SKILL, V2 SKILL, V8 noskill step2.

---

## 5. Agent behaviors observed

Concrete things agents did during these trials. No claim about how often they generalize — just things worth knowing happened.

**Behavior 1 — Reading the case source to discover absolute paths, then writing to them.** In 3 of 9 originally-rogue trials (`noskill_V4_1`, `noskill_V6_1`, `SKILL_V9_1` waveB), the agent's stream.jsonl shows it issued a `Read` tool call against `discovery/cases/vits_model_train.py`, found the literal `VITS_SRC = Path("/home/pengwu/envs/torch211/...")` constant, then issued `Edit` tool calls against that absolute path *in addition to* its sandbox copy. This produced filesystem contamination of the shared site-packages — caught by post-trial detection (Layer A). The fix shipped the same day: chmod 444 the watched file at trial start (Layer B), parse stream.jsonl post-trial for out-of-bounds Edits (Layer C), and an explicit prompt instruction to never edit `/home/pengwu/envs/...` paths.

**Behavior 2 — Many-turn exploration with tiny final commit.** V9 SKILL/waveA s1: 89 assistant turns, but the final `agent_diff.patch` is only 29 lines (compared to 180-275 lines for substantively-engaged trials). The agent explored, tried things, presumably reverted them, and ended with a near-trivial diff. `gb_under_canonical_inputs = 17` — barely engaged with the break shapes despite the long conversation.

**Behavior 3 — Inventing a fix surface form when none was named.** V6 noskill parallel didn't introduce a `_MAX_FRAMES_PER_TOKEN` constant. It wrote `max_out = input_length * 128` inline as a local variable in the function where the `arange` happens. Same pattern (static cap on the arange size), different surface form. The agent solved the problem without using the framing the V9 noskill agents used.

**Behavior 4 — Defaulting to declared overrides when the variant allows them.** V0, V2, and V4 SKILL trials all reached for S6 (declare-and-flip `capture_scalar_outputs`) as their first try at B5, even when that variant didn't *require* an override. The override was the path of least resistance, and the SKILL prompt's "Detect → Diagnose → Fix → Benchmark" workflow doesn't push the agent to question whether the fix is canonical-grade.

**Behavior 5 — Committing to a strategy and not pivoting when it doesn't survive canonical.** V8 trials: the agent runs `python validate.py`, sees `gb_under_canonical_inputs > 0`, and... declares again. Multiple V8 trials show 150-230 turns with extensive exploration, but the final state is still S6 + `backend="eager"`. The agent doesn't conclude "S6 is a shortcut, let me try a model-only fix" — instead it iterates on S6.

**Behavior 6 — Prompt fix changing what the agent commits to.** When we replaced the original 3 contaminated trials (V4 noskill, V6 noskill, SKILL_V9 waveB) with a re-run under the new prompt ("never edit `/home/pengwu/envs/...`") + chmod-RO, two of the three changed verdict — V4 noskill and V6 noskill went from `none` (original) to `general` (re-run, both via S7-class fix). Possible mechanisms: prompt fix nudged the agent away from the rogue path, or the chmod-RO physically blocked it and the agent invested its turns into the sandbox edit instead, or LLM non-determinism. Most likely a combination. Worth noting: agent behavior under "no escape outside sandbox" is observably different from agent behavior with the escape path open, even if the escape path is forbidden by prompt alone.

---

## 6. The four questions, answered as observations from this experiment

These are the four questions the skill discovery project broadly tries to address. We answer each only as far as this single experiment supports — no generalization.

### Q1. Do skill catalogs (like `debug-graph-breaks`) actually help an agent fix graph breaks — or do they sometimes hurt?

*What we observed in this experiment:* both. In V4 SKILL (no escape hatches allowed), the catalog seemed to help the agent stay methodical and reach a `general` fix. In V9 SKILL (no setup edits), the catalog didn't help — the agent stayed anchored on catalog patterns (`_dynamo.config` declares, `compiler.is_compiling` guards) and didn't pivot to the strategy that the corresponding noskill agent invented (S7, static cap). Both V9 SKILL trials landed `none` while both V9 noskill trials landed `general`.

This is a single case. We don't know if it generalizes. But it's worth knowing: the catalog can make a difference, and the direction of the difference depends on whether the right fix is in the catalog or not.

### Q2. What does the agent reach for first when faced with a graph break? "Easy" config flips, or model-code rewrites?

*What we observed:* when the variant allows it, both arms reach for the easy path (S6 — declare and flip `_dynamo.config`). Every V0/V2/V4 SKILL trial used S6 as part of its attack on B5. V8 (which explicitly allows declared overrides + makes the trade-off visible) was 100% S6 across all 3 trials. Only when the variant *forbids* setup edits entirely (V9) does the agent invest in model-code rewrites for B5 — and even then, only the noskill arm. The SKILL arm under V9 still tried catalog patterns first.

The path-of-least-resistance attractor is strong. Constraint design seems to matter more than skill arm for which path the agent takes first.

### Q3. Can agents invent fixes that aren't documented anywhere — including not in any skill catalog? Under what conditions?

*What we observed:* yes, under one condition we know about — when the variant constraint is tight enough that the catalog's preferred patterns don't fit, AND the agent isn't anchored on a catalog. Specifically: V9 noskill seed 1 invented S7 (the static-cap strategy). It had been ~50 prior trials and 30+ corrected-validator-grade trials before S7 emerged, all via different (non-V9) variants. S7 is not in `debug-graph-breaks`. Once it emerged in V9 noskill, it appeared in 4 different trials (V4/V6/V9 noskill) with two distinct surface forms (named constant + inline).

We don't know if the agent could have invented S7 under SKILL+V9 — across both V9 SKILL trials (waveA s1 + parallel) it didn't. We don't know whether it could invent fixes for break shapes other than B5 — V8's residual was B5 too, and the agent didn't discover S7 under V8 (presumably because S6 was allowed and easier).

### Q4. What does this teach us about how to design good skills? If a catalog can be a trap, what shape should it take to help without trapping?

*What we observed (suggests, doesn't prove):*

- A skill catalog with strong recipes encourages the agent to apply those recipes first. In our V0/V2/V4 SKILL trials, the agent reached for S6 (the catalog's escape path) before considering model-code rewrites. This is fine when the recipe fits; it's a problem when it doesn't.
- We don't see the agent "step out of the catalog" mid-trial. V9 SKILL/parallel engaged for 154 turns and applied 6 of the catalog's strategies, but didn't invent S7 even after they all failed.
- The catalog gap we observed is concrete and testable: add S7 (static-cap pattern) to `debug-graph-breaks`. Prediction: V9 SKILL trials should match V9 noskill (both `general`) once S7 is in the catalog. We haven't tested this prediction yet.
- Open question: what would a catalog look like that *encourages* off-catalog discovery? Possibilities surface but unverified — explicit "if none of these recipes fit, propose a new one" guidance, or pairing the catalog with a constraint variant that explicitly forbids the most common recipes, or other shapes. We don't have data here.

---

## 7. Methodology caveats + open questions

### Caveats

- **N=1 model.** Every strategy and behavior above is from VITS train mode. We do not know if (a) strategies generalize to other models with similar break shapes, (b) the same agent behaviors recur in other model contexts, or (c) the SKILL-vs-noskill split looks the same for any other case.
- **Trial counts per cell are tiny.** Most variant×arm cells are N=1. V8 noskill is N=2, V9 SKILL+noskill are N=2 each. No cell has enough samples for distributional inference; "1/1 general" and "0/1 general" are anecdotes.
- **`max_diff` correctness audit not run.** Some `general` trials show `max_diff_compiled_vs_eager ≈ 2.0` due to Inductor float32 numerics on chained ops. We have not verified that the agent's edits don't *also* introduce drift on top of the Inductor floor. This is open loop #1.
- **The validator's regex for parsing the agent's own GB count may have false positives.** `V2 SKILL/wave1a` reported `agent_gb=2` despite the agent claiming 0 GBs in their own run. Likely the regex caught an early baseline print before the agent's final config-flipped run. Open loop #5.
- **Strategy auto-detection is regex-based.** The strategy adoption appendix table (below) was built by regex-matching diffs. We hand-verified obvious cases but the matrix may misclassify edge cases (e.g. the V6 noskill parallel inline-S7 was missed by the auto-detector and added manually after diff inspection). Treat the matrix as approximate.
- **Some agent edits are inside the model file but outside our 5-break-shape mental model.** Trials sometimes touch parts of the model not in the original break list (e.g. removing intermediate `torch.IntTensor` allocations). These are usually safe but we haven't fully characterized them.

### Open questions worth investigating

1. *Apparent-fix vs correct-fix.* Audit each `general` trial's `max_diff_compiled_vs_eager` against a per-strategy correctness check. If the agent introduces drift the Inductor floor doesn't account for, that's a hidden shortcut.
2. *S7 generalization.* Does the static-cap pattern work for non-TTS data-dep `arange.size`? Worth a follow-up case (e.g. another seq2seq model with variable-length output).
3. *Negative-discovery extraction.* `stream.jsonl` per trial contains the agent's abandoned attempts. Building a tool to extract those would reveal what the agent considers and rejects — high signal for understanding the agent's strategy space, not just what it commits.
4. *Add S7 to skill catalog and re-test.* File a PR to Arsh's `debug-graph-breaks` fork adding S7. Re-run V9 SKILL. Prediction: matches V9 noskill (both `general`). If the prediction holds, that's evidence the catalog gap is causal. If it doesn't, the SKILL trap is more than a missing recipe.
5. *PyTorch issue follow-up.* V9 noskill smoke surfaced: `torch._check_tensor_all` (the supposedly compile-safe check API) still causes `as_proxy()` failures for `msg` lambda args (PyTorch 2.11). Worth confirming on current nightly + filing upstream issue if reproduces.
6. *Validator regex for `agent_gb`.* The V2 SKILL discrepancy (agent claims 0, validator reports 2) needs investigation in `validate_runner.py:_run_agent_baseline`.

### Reproducing the observations

- Per-trial result.json + agent_diff.patch + stream.jsonl: see [Trial map](#appendix-a--trial-map).
- Strategy auto-detection script: `discovery/experiments/.../reports/vits_model_train/strategy_matrix.py` (run with `python strategy_matrix.py` from the corpus repo).
- Methodology code: `discovery/validate_runner.py` (canonical-input check), `discovery/perf.py` (perf shape sanity), `discovery/run_config.py` (trial driver).

---

## Appendix A — Trial map

15 corrected-validator trials. The "rogue" original trials (V4_1, V6_1, SKILL_V9_1 waveB) were re-run after detection of filesystem contamination; their replacements (`_parallel`) are the canonical record.

| Trial | Variant | Arm | fix_status | gb_canonical | speedup_t1 |
|---|---|---|---|---|---|
| smoke noskill_V0_1 | V0 | noskill | general | 0 | 1.63x |
| wave1a SKILL_V0_1 | V0 | SKILL | general | 0 | — |
| waveA noskill_V2_1 | V2 | noskill | general | 0 | 2.05x |
| wave1a SKILL_V2_1 | V2 | SKILL | none | 2 | — |
| wave1a SKILL_V4_1 | V4 | SKILL | general | 0 | 1.90x |
| parallel noskill_V4 | V4 | noskill | general | 0 | 1.84x |
| waveA SKILL_V6_1 | V6 | SKILL | none | 10 | 1.57x |
| parallel noskill_V6 | V6 | noskill | general | 0 | 1.74x |
| step3 SKILL_V8_1 | V8 | SKILL | none | 1 | 1.20x |
| step2 noskill_V8_1 | V8 | noskill | none | 1 | 1.17x |
| step3 noskill_V8_1 | V8 | noskill | none | 1 | 1.15x |
| smoke noskill_V9_1 s1 | V9 | noskill | general | 0 | 1.53x |
| waveA SKILL_V9_1 s1 | V9 | SKILL | none | 17 | 1.05x |
| waveB noskill_V9_1 s2 | V9 | noskill | general | 0 | 2.07x |
| parallel SKILL_V9 | V9 | SKILL | none | 1 | 1.22x |

## Appendix B — Strategy adoption matrix (raw data, not analytics)

Auto-detected from each trial's `agent_diff.patch` by regex. Hand-verified for obvious cases. May misclassify edge cases; treat as approximate. *No conclusions are drawn from the counts; the matrix is here so anyone can audit which strategy went where.*

| Trial | Variant | Arm | fix | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S10 | S11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke noskill_V0_1 | V0 | noskill | gen | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ |
| wave1a SKILL_V0_1 | V0 | SKILL | gen | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | ✓ | ✓ |
| waveA noskill_V2_1 | V2 | noskill | gen | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ |
| wave1a SKILL_V2_1 | V2 | SKILL | none | ✓ | ✓ | · | ✓ | · | ✓ | · | · | ✓ | ✓ |
| wave1a SKILL_V4_1 | V4 | SKILL | gen | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ |
| parallel noskill_V4 | V4 | noskill | gen | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | · | · |
| waveA SKILL_V6_1 | V6 | SKILL | none | ✓ | ✓ | ✓ | ✓ | · | · | · | · | · | ✓ |
| parallel noskill_V6 | V6 | noskill | gen | ✓ | ✓ | ✓ | ✓ | · | · | (✓) | · | · | · |
| step3 SKILL_V8_1 | V8 | SKILL | none | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | · | · | ✓ |
| step2 noskill_V8_1 | V8 | noskill | none | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | ✓ |
| step3 noskill_V8_1 | V8 | noskill | none | ✓ | ✓ | ✓ | ✓ | · | ✓ | · | ✓ | · | · |
| smoke noskill_V9_1 s1 | V9 | noskill | gen | ✓ | ✓ | ✓ | ✓ | ✓ | · | ✓ | ✓ | · | ✓ |
| waveA SKILL_V9_1 s1 | V9 | SKILL | none | ✓ | · | · | · | · | · | · | · | · | · |
| waveB noskill_V9_1 s2 | V9 | noskill | gen | ✓ | ✓ | · | ✓ | · | · | ✓ | · | · | ✓ |
| parallel SKILL_V9 | V9 | SKILL | none | ✓ | ✓ | · | ✓ | · | ✓ | · | ✓ | · | ✓ |

`(✓)` = inline variant of S7 (used `max_out = input_length * 128` instead of a named constant). Pattern-equivalent.

---

## Revision log

- 2026-04-28 (morning): skeleton — Phases B-F stubs filled with partial data; batch2 pending.
- 2026-04-28 (afternoon): all 15 corrected-validator trials in; statistical narrative written.
- 2026-04-28 (evening, v3): **field report rewrite.** Per Peng: this is one model, no generalization. Restructured around (1) methodology, (2) break shapes attacked, (3) working strategies per shape, (4) shortcut strategies per shape, (5) agent behaviors observed, (6) 4 project questions answered as anecdotes from this experiment, (7) caveats + open questions. Strategy adoption matrix moved to appendix as raw data, not analytics. Statistical conclusions removed.
