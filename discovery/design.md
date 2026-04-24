# Discovery Agent — Design Doc (v0.2, baked-in implications)

**Status:** Pre-implementation. v0.2 folds in the 4 implications from the Pilot 4 forensic re-run.
**Author:** Otter
**Date:** 2026-04-24

## Revision Log

- **v0.1 (2026-04-24 morning)** — initial draft, sent to Peng for review.
- **v0.2 (2026-04-24 mid-morning)** — folded in 4 implications from Pilot 4 forensic re-run:
  1. Strategy fingerprint gains a 5th axis (`dynamo_config_changes`).
  2. New constraint variant V6 (no config flags).
  3. Trial harness must restore *both* model source AND test/baseline file between trials; post-trial diff check flags any test-file mutation not reflected in agent's diff.
  4. Pilot 4 strategy distribution to be re-tallied splitting clean / deliberate-flag / contaminated trials; lean is to skip re-running the contaminated 3 (discovery agent will re-cover the same case).
- **v0.3 (2026-04-24 noon, per Peng)** — perf primitive scope correction: drop compile-time measurement. Compile is a one-time cost that varies with cache state and is unrelated to steady-state fix quality. The first warmup iteration triggers compilation and is discarded along with the rest of warmup.

---

## 1. Goal

Move from "pilot studies that produce one fix per case" to a **discovery agent** that, per case, surfaces the *strategy space* — runs M constraint variants, collects the strategies that emerge, and scores them on a 4-axis rubric (compute / accuracy / code complexity / perf).

The output of one discovery run is a **trade-off matrix** for that case, not a single fix.

## 2. Non-goals

- Not a replacement for `sweep/` — sweep is breadth-first across the corpus; discovery is depth-first per case.
- Not a fix-shipping tool — the output is decision-support material for humans (and, eventually, for skill catalog curation).
- Not a sweep-scale infrastructure project (yet) — first ships scoped to one case at a time, manually invoked.

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

## 8. Open questions for Peng

1. **Scope of M.** How many constraint variants per case is enough? My lean: 4 to start (V0/V1/V2/V4), expand if a case has obvious axes the catalog doesn't cover. **Recommend 4.**

2. **Trials per variant (N).** Pilot 4 ran 3 per arm; that's enough for "did it converge" but small for "what's the strategy distribution." My lean: N=3 for first case, scale to N=5 once cost is understood. **Recommend 3.**

3. **Strategy fingerprint sophistication.** Heuristic regex+AST is cheap. LLM-as-judge is more robust but costs $0.10–0.50 per trial classification. My lean: heuristics first, LLM-as-judge for the cases heuristics can't classify. **Recommend hybrid.**

4. **Where the discovery output lands.** Local file? Drive doc? Both? My lean: JSON to `discovery_results/<case>/<run_id>/`, narrative report uploaded to Drive PARA. **Recommend both.**

5. **Phase 1 case selection.** Start with Dbrx (we have rich Pilot 4 + forensic data) or pick a fresh case to avoid confirmation bias? **Recommend Dbrx for Phase 1** — having forensic data lets us validate the harness against known ground truth before scaling.

## 9. What this enables

- A reusable per-case methodology that can be applied to any graph-break case in the corpus.
- A trade-off matrix per case that turns "did the agent fix it" into "which strategies exist and what do they cost on the axes that matter."
- A cleaner discovery loop for the *novel-to-catalog vs novel-to-the-world* question — the strategy clusters make catalog gaps mechanical to surface.

## 10. What this does NOT enable (yet)

- Cross-case strategy patterns (e.g., "bmm shows up in 4 different MoE cases — should it be a skill?") — that's a separate aggregation layer on top of multiple discovery runs.
- Automatic skill catalog generation — the human curator still picks what becomes a skill.
- Real-time agent steering — discovery is offline, batch.

---

*Awaiting Peng's review on §8 open questions before any code lands.*
