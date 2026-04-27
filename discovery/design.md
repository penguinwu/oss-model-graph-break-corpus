# Discovery Agent — Design Doc (v0.6, harness-bug retraction + set_seed correction)

**Status:** Phase 1 complete on Dbrx; cross-case skill discovery in progress (Mistral3, VitsModel). v0.6 retracts the v0.5 "Inductor noise floor 2.0" attribution after empirical disproof, and corrects the corpus's reseed pattern.
**Author:** Otter
**Date:** 2026-04-27

> **Before running ANY discovery experiment** (multi-trial run, harness change, schema change, case addition), read [`EXPERIMENT_LIFECYCLE.md`](EXPERIMENT_LIFECYCLE.md) — the 5-gate methodology that's mandatory for all discovery work. The gates exist because we kept rushing launches and burning attention on bugs that small-scale validation would have caught in seconds.

> ⚠️ **2026-04-27 retraction summary** — read this BEFORE acting on §2.1 / §4.3 / §4.7 below. The "Inductor noise floor 2.0" framing throughout v0.5 was empirically wrong. The actual root cause was our validator using `torch.manual_seed` alone, which doesn't seed numpy/python.random. Many HF models (VITS layerdrop) use `np.random` in forward — outputs were intermittently non-deterministic. With HF's full `set_seed` (torch + numpy + python.random + cuda), eager VITS is bit-identical across forwards. The 2.0 magnitude is the saturation max for [-1, 1]-bounded waveform output, not codegen drift. *V0.5's manual_seed-as-L-non-functional-defensive reclassification is therefore also wrong* — manual_seed insertions ARE incomplete, not non-functional. Full-rewrite of §2.1/§4.3/§4.7 deferred until next iteration; treat them as superseded.

## Revision Log

- **v0.1 (2026-04-24 morning)** — initial draft, sent to Peng for review.
- **v0.2 (2026-04-24 mid-morning)** — folded in 4 implications from Pilot 4 forensic re-run:
  1. Strategy fingerprint gains a 5th axis (`dynamo_config_changes`).
  2. New constraint variant V6 (no config flags).
  3. Trial harness must restore *both* model source AND test/baseline file between trials; post-trial diff check flags any test-file mutation not reflected in agent's diff.
  4. Pilot 4 strategy distribution to be re-tallied splitting clean / deliberate-flag / contaminated trials; lean is to skip re-running the contaminated 3 (discovery agent will re-cover the same case).
- **v0.3 (2026-04-24 noon, per Peng)** — perf primitive scope correction: drop compile-time measurement. Compile is a one-time cost that varies with cache state and is unrelated to steady-state fix quality. The first warmup iteration triggers compilation and is discarded along with the rest of warmup.
- **v0.4 (2026-04-24 evening, post-Phase-1)** — closed §8 open questions with answers Phase 1 produced on Dbrx. M=variant-driven (V0/V1/V6 sufficed for Dbrx), N=3 confirmed adequate for strategy-distribution signal, fingerprint stayed heuristic (no LLM-as-judge needed for Dbrx), output landed at `discovery/runs/<case>/<run-id>/` + report at `discovery/reports/<case>_<date>.md`, Phase 1 case = Dbrx as recommended.
- **v0.5 (2026-04-26, post-VITS-V8 + empirical noise-floor test)** — three changes:
  1. **§2.1 sharpened with proof.** Inductor noise-floor magnitude and cause are now empirical, not estimated. A fixed-model eager-vs-inductor test produced `backend=eager → max_diff=0`, `backend=inductor → max_diff=2.0` on the same code path. Confirms the floor is *Inductor codegen numerical drift*, not RNG. Old "~1e-4" estimate replaced.
  2. **New §4.7 — Variant-rule design principle: "declared fallback with required justification."** Variants forbid escape hatches by default. Agent may use a forbidden mechanism *only if* declared + justified in the findings doc. Post-trial classifier sorts trials into "legit declared fallback" vs "undeclared shortcut." Applies uniformly to backend choice, capture_* flags, and any future agent-action category. Through-line confirmed by Peng three times this session.
  3. **§4.3 amended.** Canonical check now reseeds (`validate_runner.py:_run_canonical_check`) and per-case baselines reseed before each leg of the eager/compiled comparison. `manual_seed` insertions empirically reclassified as **L non-functional defensive** — they don't flatten anything because the noise floor is not RNG.
- **v0.6 (2026-04-27, post-V8 deep-dive infra audit)** — methodology corrections after multiple harness bugs surfaced during V8 follow-up:
  1. **§2.1 RETRACTED.** The "Inductor noise floor 2.0" attribution was incorrect. Confirmed empirically: with HF's full `set_seed` (torch + numpy + python.random + cuda) reseeded before each forward, VITS is bit-identical across 7/7 forwards in 4/4 fresh process runs. The original test that "proved" Inductor drift used `torch.manual_seed` alone, which left numpy and python.random adrift; VITS layerdrop's `np.random.uniform` produced different layer-drop patterns each forward → different output sequence lengths → 2.0 max diff (saturation magnitude for [-1, 1]-bounded waveform). The 2.0 was not Inductor codegen drift; it was un-reseeded numpy RNG.
  2. **§4.3 RETRACTED partially.** `manual_seed` insertions are NOT "L non-functional defensive" — they're INCOMPLETE (cover only torch RNG, not numpy/python.random). Reclassification to L was wrong. Awaiting full rewrite.
  3. **`_dynamo.explain` → `sweep.explain` migration.** Validator now uses corpus-canonical `sweep.explain.run_graph_break_analysis` (TORCH_LOGS-based) instead of deprecated `_dynamo.explain` (known to segfault on VITS in nightly). Methodology consistency restored across sweep + discovery.
  4. **`_eager_self_check` shape-mismatch crash fixed.** Used `_safe_max_diff` prefix-clamp pattern, matching validate_runner. Stochastic-output models like VITS train no longer crash perf measurement.
  5. **New schema fields** for distinguishing "model broken at perf shapes" from "measurement noise":
     - `validation.details.gb_call_sites` — `[{reason, type, file, line, location}]` per break, classified by `sweep.explain` (Tensor.item() / data-dependent-branch / aten.nonzero / other)
     - `validation.details.eager_self_diff`, `eager_deterministic` — lifted from perf to validator; now means "is set_seed sufficient for this model?" rather than "does the model use any RNG?"
     - `perf.perf_shape_sanity` — `"ok" | "runtime_failure" | "unknown"`. Pre-timing forward at perf shapes catches model-broken-at-perf-shapes failures distinctly from measurement noise.
     - `perf.runtime_failure_msg` — RuntimeError message when sanity fails
     - `fix_survives_perf` (top-level on TrialResult) — `True | False | None`. Integrated verdict combining validator's fix_status with perf shape-sanity from both tiers.
  6. **`smoke_test.py` added** — Layer 1 synthetic harness self-tests (tiny MLP + clean / broken / np.random patterns) + Layer 2 per-case smoke. Mandatory pre-launch check. Crucially includes the broken-PATH tests (perf RuntimeError, np.random regression) — the absence of these catches "happy path works, broken path silently fails" regressions like the `_eager_self_check` bug we discovered after multi-hour V8 wait.
  7. **Validator now uses HF `set_seed`** at all reseed sites (was `torch.manual_seed`-only). Smoke test `test_validator_seeding_covers_nprandom` is the regression guard.
  8. **Corpus-wide audit pending** — 5 other discovery cases, `experiments/scripts/run_eval.py`, `discovery/perf.py`'s `patch_torch_manual_seed`, and `sweep/worker.py`'s fallback all use incomplete reseed patterns. Filed in OPEN-LOOPS.

---

## 1. Goal

Move from "pilot studies that produce one fix per case" to a **discovery agent** that, per case, surfaces the *strategy space* — runs M constraint variants, collects the strategies that emerge, and scores them on a 4-axis rubric (compute / accuracy / code complexity / perf).

The output of one discovery run is a **trade-off matrix** for that case, not a single fix.

## 2. Non-goals

- Not a replacement for `sweep/` — sweep is breadth-first across the corpus; discovery is depth-first per case.
- Not a fix-shipping tool — the output is decision-support material for humans (and, eventually, for skill catalog curation).
- Not a sweep-scale infrastructure project (yet) — first ships scoped to one case at a time, manually invoked.

### 2.1 Backend choice — inductor (default), not eager

Discovery uses `torch.compile(model)` with the *default* backend (inductor). `sweep/` uses `backend="eager"` for fast breadth-first surveying with no codegen. Discovery is the opposite: depth-per-case where production-realistic perf matters, and codegen is part of what we're studying. A successful "fix" at the discovery layer means the agent improved compile-time + runtime perf, which only inductor can measure.

**Empirical noise floor (v0.5 update, 2026-04-26).** The inductor backend introduces a per-case `max_diff` floor that is *not* RNG. Verified on a fixed VITS model: same code path, same inputs, two backends —

| Backend | `max_diff` (eager vs compiled) |
|---|---|
| `eager` | 0.0 |
| `inductor` (default) | ~2.0 |

The 2.0 magnitude is Inductor codegen numerical drift (op reordering, fused kernel arithmetic). It does not change between runs, is not RNG-related, and cannot be flattened by `manual_seed` insertion. This floor must not be confused with a correctness regression, but also must not be used as cover for a real correctness regression — see §4.7 for how variants enforce that.

Trade-off accepted: the per-case noise floor is real (case-dependent, can be calibrated), but the perf signal Inductor produces is essential to the trade-off matrix. Worth it.

Decision: 2026-04-25, per Peng. Empirical confirmation: 2026-04-26. Documented here so future contributors don't accidentally swap backends, and so future readers don't mistake the floor for a regression.

## 3. Why now

Pilot 4 surfaced two things our current setup can't see:

1. *Strategy-space invisibility.* Pilot 4 asked "did the constraint sentence steer the agent away from masked-dense?" Yes. But it didn't ask "what other strategies exist, and how do they compare?" 3/6 trials produced bmm; 3/6 produced partial range-loop. We have no framework for comparing those.

2. *No perf signal.* When the agent generalized the bmm pattern to Dbrx (where the human maintainer chose masked-dense), we had no way to assess whether the agent's choice was better, worse, or just different. Perf is the missing axis.

The discovery agent makes the strategy space and the trade-offs explicit.

## 4. Design

### 4.1 Per-case workflow

```
Case (model + break-shape + constraint context)
  → run M constraint variants (each variant = different prompt steering)
  → collect K successful trials (K ≤ M; some variants may fail)
  → cluster trials by strategy fingerprint
  → assessor pass scores each unique strategy on 4 axes
  → synthesis report (trade-off matrix)
```

### 4.2 Constraint variant catalog

Each variant = a prompt body with a different *steering constraint*. Initial catalog (extensible):

- *V0 — bare:* no constraint, "fix the graph break"
- *V1 — sparsity preserved:* "must not collapse sparse compute to dense" (Pilot 4's constraint)
- *V2 — bitwise equivalent:* "output must be bitwise equal to eager" (forces escape-hatch family)
- *V3 — minimal diff:* "smallest possible code change to fix"
- *V4 — no escape hatches:* "do not use custom_op / disable / allow_in_graph / cond"
- *V5 — escape hatch only:* "use only custom_op or disable" (negative-space variant)
- *V6 — no config flags:* "do not modify torch._dynamo.config; fix in source code only" (added v0.2 — see §4.4 for why)

Recommended starting set: V0, V1, V2, V4, V6. (V3 and V5 are nice-to-have.) V0/V1/V2/V4 implicitly allow flag changes; the fingerprint surfaces them. V6 forces a code-only fix so we can see whether agents have a fallback when flag-set is taken away.

### 4.3 Trial execution

Reuse the Pilot 4 harness shape: per-trial isolated state restore, GPU pre-check, hard timeout, validate.py, agent_diff.patch capture.

**v0.2 hardening (from Pilot 4 forensic finding):**
- Per-trial restore covers *both* model source AND test/baseline file from `.original` backups. (Pilot 4's harness only restored model source — a flag added by `with_skill_2` to `baseline_dbrx.py` persisted into 4 subsequent trials and silently contaminated them.)
- Post-trial diff check: if any restored file differs from `.original` and the diff is not present in `agent_diff.patch`, the trial is flagged `test-file-mutated` and its strategy fingerprint is marked invalid.
- All file mutations the agent makes — including ones to the test harness — must show up in the captured diff.

**v0.5 hardening (canonical-check determinism, 2026-04-26):**
- `validate_runner.py:_run_canonical_check` reseeds (torch + python + numpy) before each leg of the eager-vs-compiled comparison. Systemic — applies to every case.
- Per-case baselines reseed analogously inside their `.original` files (currently shipped: `baseline_vits.py.original`; the other 5 baselines are an open loop, see Decision 1 in `~/.myclaw/spaces/AAQANraxXE4/threads/2026-04-26-determinism-decisions.md`).
- This pins eager-vs-eager `max_diff` to deterministic 0.0, exposing Inductor's intrinsic noise floor (§2.1) cleanly.
- **Implication for the L/M/S/R taxonomy (§4.7):** `manual_seed` insertions in agent diffs are now empirically classified **L non-functional defensive** — they don't flatten max_diff because the floor isn't RNG. They are not measurement-affecting (M) and not shortcuts (S).

Every trial runs with `--output-format stream-json --include-partial-messages` so we capture the full trace for downstream classification.

### 4.4 Strategy fingerprint

A "strategy" is the *shape* of the fix, not the exact diff. Fingerprint = a tuple over the diff:

- *control flow change:* range-loop / bmm / vectorized-bucketize / unchanged
- *data dependency removal:* gather-scatter / mask-and-multiply / branch-removal / escape-hatch
- *escape hatch used:* none / custom_op / disable / allow_in_graph / cond
- *dynamo config changes:* none / capture_dynamic_output_shape_ops / suppress_errors / other (added v0.2)
- *bitwise preservation:* yes / no (from validate.py)

Two trials with the same fingerprint cluster to one strategy. Initial classifier: regex + AST inspection on the agent_diff.patch. Punt to LLM-as-judge if heuristics fail.

**Why config-flag is its own axis (v0.2):** in Pilot 4, `with_skill_2` set `torch._dynamo.config.capture_dynamic_output_shape_ops = True` as a deliberate, documented strategy. Treating "set a config flag" as an invisible no-op or as part of the "escape hatch" axis would lose signal — flags are a *distinct* class of fix (config-level vs. code-level) with different deployability properties (config can be set at runtime; code change requires a release). The fingerprint must surface them as their own dimension.

### 4.5 Assessor pass

For each unique strategy, score on 4 axes:

| Axis | How |
|------|-----|
| Compute | Static analysis: O(top_k) vs O(num_experts) vs unchanged. From fingerprint. |
| Accuracy | `validate.py` already produces `max_abs_diff` + `bitwise_equal`. Use directly. |
| Code complexity | Cyclomatic complexity of the patch + lines changed. Cheap, mechanical. |
| Perf | `measure_perf(model, inputs)` from the WS1 perf primitive — eager vs compiled steady-state latency + peak mem. *Compile time intentionally not measured* (one-time cost, varies with cache state, unrelated to fix quality). |

Output: JSON record per strategy with all 4 axis scores.

### 4.6 Synthesis report

Per case, produces:
- *Strategy table:* one row per unique strategy, all 4 axis scores, count of trials that produced it, exemplar diff
- *Headline:* which strategy wins on which axis, where there's tension
- *Catalog gap flag:* if any strategy isn't represented in the current skill catalog, flag explicitly

### 4.7 Variant-rule design principle: declared fallback with required justification

Variants are *intent specifications*, not hard bans. The default for every forbidden mechanism is forbidden — but the agent may override iff it declares the override and justifies why. This keeps variants honest about their purpose (probing what the agent reaches for and how it reasons about constraints) without refusing to produce a fix on cases where the forbidden mechanism is the only honest answer.

**Three rules.**

1. *Default = forbidden.* Each variant enumerates mechanisms it forbids. Without declaration, those mechanisms count as constraint violations.
2. *Override = declared + justified.* The agent may use a forbidden mechanism by writing a one-line declaration in the trial output naming (a) the mechanism used and (b) the specific reason it was unavoidable. Format: `# DECLARED-OVERRIDE: <mechanism> — <reason>` either inline in the diff or in the agent's final summary.
3. *Post-hoc classification.* Per-case Phase B (`per-case-analysis` skill) runs a classifier matching declarations against the diff. Every trial receives an L/M/S/R tag.

**L/M/S/R taxonomy (per-trial post-hoc classification).**

- *L (Legitimate)* — override was declared AND the diff matches the declaration. The agent honored the variant's intent: it knew it was breaking the constraint and named why. The fix is admissible into strategy aggregation.
- *M (Measurement-affecting)* — override changes what the experiment is measuring. Example: `backend="eager"` hides Inductor's noise floor (§2.1) so the perf and accuracy axes mean something different from every other trial. M-tagged trials are reported separately; their numbers do not aggregate with the rest.
- *S (Shortcut)* — the diff uses a forbidden mechanism with no matching declaration. Counts as an undeclared constraint violation. Surfaces in the strategy distribution as "agent routed around the constraint without acknowledging it" — itself a signal worth measuring.
- *R (Refused)* — the agent declared it could not produce a fix under the constraint and exited cleanly without writing a forbidden-mechanism diff. Valid outcome — surfaces "this constraint is too tight for this case" cleanly.

**Classifier sketch.** For each declared override in the trial's stream-final-summary or `# DECLARED-OVERRIDE:` comment in the diff, check the diff for the named mechanism. Match → L (or M if the mechanism is on the measurement-affecting list). Mechanism present in diff but no declaration → S. No fix attempted + clean refusal → R.

**Two current applications.**

- *Backend choice (default = inductor).* `backend="eager"` is M-tagged — it hides the Inductor noise floor (§2.1). Permitted only with a one-line declaration naming the Inductor numerics issue being sidestepped.
- *Capture flags (`capture_scalar_outputs`, `capture_dynamic_output_shape_ops`).* These count as a strategy axis (§4.4 axis 4: `dynamo_config_changes`). Permitted only with a one-line declaration naming the specific data-dep op being guarded.

**Why this beats hard bans.** A hard ban makes a variant unrunnable on cases where the forbidden mechanism is the only honest fix. The trial either fails (lost signal) or the agent silently violates (lost signal). Declared-fallback converts the variant from a syntactic gate into an intent probe: we see whether the agent reaches for the mechanism, AND we see how it justifies the reach. The classifier sorts the trials post-hoc; the agent never has to refuse to fix the case.

**Through-line.** This principle applies uniformly to backend choice, capture_* flags, escape-hatch usage, future agent-action categories, and any variant V_n added after this date. New variants must specify their forbidden mechanism set + which mechanisms (if any) are M-tagged.

## 5. Output schema

```json
{
  "case_id": "dbrx_moe_data_dep",
  "model": "DbrxModel",
  "break_shape_id": "BS-103",
  "discovery_run_id": "20260424-093200",
  "variants_attempted": ["V0", "V1", "V2", "V4"],
  "trials_per_variant": 3,
  "strategies": [
    {
      "fingerprint": "range-loop+mask-multiply+none+no",
      "n_trials": 4,
      "variants_seen_in": ["V0", "V1"],
      "exemplar_diff_path": "...",
      "compute_class": "O(num_experts)",
      "accuracy": {"max_abs_diff": 1.49e-7, "bitwise_equal": false},
      "complexity": {"loc_changed": 23, "cyclomatic_delta": 2},
      "perf": {"eager_ms": 12.4, "compiled_ms": 8.1, "speedup": 1.53, "eager_peak_mem_mb": 4892, "compiled_peak_mem_mb": 4920}
    },
    { "fingerprint": "bmm+gather-scatter+none+no", ... }
  ],
  "headline": "bmm wins on perf+complexity; range-loop+mask wins on conservatism (no shape change to MoE pattern)",
  "catalog_gaps": ["bmm+gather-scatter not in current skill catalog"]
}
```

## 6. Repo layout

```
discovery/
├── runner.py          # config-driven trial runner (M variants × N trials)
├── variants.py        # constraint variant catalog
├── fingerprint.py     # strategy fingerprint extractor
├── assessor.py        # 4-axis scoring (calls measure_perf, validate.py, etc.)
├── synthesizer.py     # per-case report generator
├── perf.py            # measure_perf primitive (WS1 perf reframe)
└── cases/             # per-case config files (model + inputs + initial constraint set)
    └── dbrx_moe_data_dep.yaml
```

Sibling to `sweep/`. Owns its own subcommand: `python discovery/run.py --case dbrx_moe_data_dep`.

Per Peng 2026-04-24: tucks under corpus repo for now (one-repo manageability); clean boundary so we can `git mv` to a sibling repo later.

## 7. Phasing

**Phase 1 (~2 days):** measure_perf primitive + runner + variants V0/V1 + naive fingerprint. Single case (Dbrx). Manual scoring.

**Phase 2 (~2 days):** assessor pass automation + synthesis report. Re-run Phase 1 case end-to-end automated.

**Phase 3 (~1 day):** add V2 (bitwise) + V4 (no escape hatches). Run on second case (Jamba or T5).

**Phase 4 (deferred):** add bitwise as a sweep-level constraint, scale to corpus. This is the level-4 corpus extension — not Phase 1.

## 8. Open questions — closed by Phase 1 (Dbrx)

1. **Scope of M.** Phase 1 ran V0/V1/V6 on Dbrx (3 variants, not 4). V2 (bitwise) was deferred — accuracy axis already showed max_diff ~1e-7 on V0/V1/V6 so bitwise wasn't a separate axis worth a variant. V4 (no escape hatches) folded into V6's broader scope. **Conclusion:** M is variant-driven, not a fixed number. Pick variants that probe distinct *constraint axes*, not just "more is better." For Dbrx, 3 variants × 3 trials = 9 trials surfaced the full strategy space (masked-dense, bmm, config-flag).

2. **Trials per variant (N).** N=3 was sufficient for Dbrx — V0×3 all converged to masked-dense (low strategy variance under no constraint), V6×3 all converged to masked-dense (V0 attractor confirmed under tightened constraint), V1×3 split 1 bmm + 2 config-flag (constraint-induced variance). The split was visible at N=3. **Conclusion:** N=3 for first case of any new break-shape. Scale to N=5 only if N=3 leaves the strategy distribution ambiguous.

3. **Strategy fingerprint sophistication.** Heuristic regex (`for expert_idx in range(...)` for masked-dense, `torch.bmm` for bmm, `_dynamo.config` mutation for escape-flag) + post-trial diff inspection sufficed for all 9 Dbrx trials. **Conclusion:** Heuristic-first works for cases where the strategies are syntactically distinguishable. LLM-as-judge unused so far; reserve for the first case where heuristics can't classify a trial.

4. **Where the discovery output lands.** Settled on: per-trial artifacts at `discovery/runs/<case>/<run-id>/V<n>_<trial>/` (agent_diff.patch, prompt.txt, result.json, stream.jsonl), per-run summary at `discovery/runs/<case>/<run-id>/summary.json`, narrative report at `discovery/reports/<case>_<date>.md`. Drive upload deferred until report stabilizes; reports live in repo first.

5. **Phase 1 case selection.** Confirmed Dbrx — Pilot 4 forensic data made it possible to validate the harness against known ground truth (V1 contamination caught by harness v0.2 dual-file restore + post-trial diff check exactly as designed).

### New questions opened by Phase 1 (for Phase 3 case selection)

- **Cross-case strategy generalization.** Does masked-dense's "strong attractor" property hold for other MoE cases (T5-MoE, Mixtral)? Or is it Dbrx-specific?
- **Different break-shape, same methodology?** Jamba (data-dependent branch, BS-104) is a different break-shape than Dbrx (data-dependent for-loop, BS-103). Does the V0–V6 catalog need shape-specific variants, or do the existing variants generalize?
- **When is N>3 needed?** Phase 1 didn't hit it. The first case where N=3 leaves strategy distribution ambiguous will be the trigger.

## 9. What this enables

- A reusable per-case methodology that can be applied to any graph-break case in the corpus.
- A trade-off matrix per case that turns "did the agent fix it" into "which strategies exist and what do they cost on the axes that matter."
- A cleaner discovery loop for the *novel-to-catalog vs novel-to-the-world* question — the strategy clusters make catalog gaps mechanical to surface.

## 10. What this does NOT enable (yet)

- Cross-case strategy patterns (e.g., "bmm shows up in 4 different MoE cases — should it be a skill?") — that's a separate aggregation layer on top of multiple discovery runs.
- Automatic skill catalog generation — the human curator still picks what becomes a skill.
- Real-time agent steering — discovery is offline, batch.

---

*Phase 1 complete on Dbrx (commit 6684ffa, 2026-04-24). §8 closed. Phase 3 (second case) awaiting Peng's go on case selection.*
