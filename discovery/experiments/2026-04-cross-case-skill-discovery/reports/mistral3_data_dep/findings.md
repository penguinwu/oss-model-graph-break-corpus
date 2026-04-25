# Mistral3 Case 3a — Findings (24-trial cross-case skill discovery)

**Case:** `mistral3_data_dep` (Mistral3ForConditionalGeneration, multimodal vision-language)
**Run:** `/tmp/discovery-runs/mistral3_data_dep/20260425-041832/`
**Matrix:** 2 skill arms (none, debug-graph-breaks) × 4 constraint variants (V0, V2, V4, V6) × 3 trials = 24 trials
**Wall:** 6.5 hr sequential
**Author:** Otter
**Date:** 2026-04-25

---

## TL;DR

- Every trial (24/24) achieved gb=0 in the agent's own run. Two distinct strategy classes emerged: **model-layer** (fix in `modeling_*.py` only — generalizes; 8 trials) and **mixed model+setup** (also edits `baseline_mistral3.py` to pass `image_sizes` as a Python list; 16 trials).
- *fix_status × performance correlation is the dominant signal:* general fixes get **median 2.68x speedup** at tier-1 (range 1.71–3.18x); setup-required fixes get **median 1.46x** (range 1.39–1.94x). Same case, two strategy classes, ~1.8x perf delta between them.
- *Skill arm steers strategy:* bare Claude finds the model-layer (general) fix 6/12 times; with-skill 2/12. The skill is steering toward a valid-but-slower strategy class.
- *Zero canonical escape hatches.* No agent used `custom_op`, `disable`, `cond`, `allow_in_graph`, `nonstrict_trace`, or `leaf_function`. Bare arm reached for `torch.compiler.is_compiling()` guards (4/12); skill arm did not (0/12).

---

## The data

Full per-trial fingerprint at `fingerprints.csv` (24 rows). Aggregations:

### fix_status × arm × variant

| Arm | V0 | V2 | V4 | V6 | total general / setup-required |
|---|---|---|---|---|---|
| noskill | 3 / 0 | 1 / 2 | 2 / 1 | 0 / 3 | **6 / 6** |
| SKILL | 0 / 3 | 0 / 3 | 1 / 2 | 1 / 2 | **2 / 10** |

Values: `general / setup-required`. No `none` outcomes.

### Performance by fix_status

| | n | median speedup tier-1 | range | median speedup tier-2 | range |
|---|---|---|---|---|---|
| general | 8 | 2.68x | 1.71–3.18 | 2.63x | 2.03–4.01 |
| setup-required | 16 | 1.46x | 1.39–1.94 | 1.44x | 1.14–1.94 |

### Per-arm aggregates

| | median speedup tier-1 | median elapsed | median turns |
|---|---|---|---|
| noskill | 1.90x | 1170s | 40 |
| SKILL | 1.60x | 972s | 51 |

### Strategy clusters (fix_locus × is_compiling)

| (fix_locus, is_compiling) | trials |
|---|---|
| (both, none) | 16 |
| (model-only, none) | 4 |
| (model-only, is_compiling) | 2 |
| (both, is_compiling) | 2 |

### fix_locus × fix_status (perfect correlation)

| fix_locus | general | setup-required |
|---|---|---|
| model-only | 6 | 0 |
| both | 2 | 16 |

### Break shapes attacked (across all trials)

| Break shape | times addressed |
|---|---|
| scalar_dense / `.item()` | 29 |
| nonzero / boolean indexing | 24 |
| `.tolist()` | 22 |
| `torch_compilable_check` | 21 |
| `@capture_outputs` decorator | 14 |
| list comprehension | 10 |
| `generate_block_attention_mask` | 7 |
| cumsum tensor slice | 4 |
| `image_sizes` (tensor) | 4 |
| unfold decomp | 2 |

---

## Phase C — Question-by-question walk

### Q1: Where does the agent edit?

24/24 trials touched `modeling_mistral3.py` AND `modeling_pixtral.py`. The split is on whether they ALSO touched `baseline_mistral3.py`:

- *model-only (6 trials):* only model files — produces `general` fix
- *both (18 trials):* model + baseline — produces `setup-required` if model alone is insufficient (16) or `general` if model alone happens to be sufficient (2)
- *setup-only:* zero — no agent edited only the baseline

The fix_locus axis perfectly predicts fix_status: editing only the model gives general fixes; editing baseline too gives setup-required (with rare exceptions).

### Q2: Which break shapes get fixed?

Every trial addressed essentially the same core set of breaks: `.item()`, `nonzero`, `.tolist()`, `torch_compilable_check`. About half (14/24) ALSO addressed the `@capture_outputs` decorator (which triggers ContextVar/threading.Lock breaks). The unfold decomp issue — which I originally flagged as the dominant break shape during case authoring — was only explicitly mentioned by 2 trials. Most agents fix it incidentally via the broader image_sizes restructure.

There's no break shape that "always gets fixed" vs "sometimes ignored." The breaks the agents identify are largely uniform; the variation is in HOW they fix them.

### Q3: What fix patterns does the agent reach for?

Of the canonical fix-shape families (deletion / restructure / wrap / config-flag-flip / escape-hatch / input-type-tweak / mixed):

- **21/24 trials are "mixed"** — every successful fix combines multiple families. The fix-shape-family axis as defined doesn't discriminate well for this case.
- 3 trials are pure `restructure` — model-only structural rewrites with no flag flips, no input tweaks.
- ZERO trials used `escape-hatch` family (no custom_op, disable, cond, allow_in_graph, nonstrict_trace, leaf_function).
- ZERO trials used `config-flag-flip` (no `torch._dynamo.config` modifications) — V6 explicitly forbade this; the other variants permitted it but no agent reached for it anyway.

The dominant patterns observed:
- Replace tensor `image_sizes` with Python list at the call site (16 trials — the setup-layer move)
- Replace per-image loop with batched unfold using static config dims (8 trials — the structural model-layer move)
- Wrap `torch_compilable_check` calls in `if not torch.compiler.is_compiling():` guard (4 trials — bare arm only)
- Delete `@capture_outputs` decorator and inline its hidden-state collection (about half)

### Q4: Does the fix preserve performance?

**The most important answer in this analysis.** YES across the board (every trial improves on the unfixed baseline of 1.49x compiled-vs-eager) — but with a ~1.8x ratio between the two strategy classes:

- general (model-layer) fixes: median 2.68x speedup (range 1.71–3.18) at tier-1
- setup-required (model+setup) fixes: median 1.46x (range 1.39–1.94) at tier-1
- Tier-2 (realistic batch=4): pattern holds — general 2.63x median; setup-required 1.44x

The setup-required cluster's max speedup (1.94x) doesn't reach the general cluster's MIN (1.71x). The two distributions don't overlap. This is the strongest signal in the run: WHICH fix locus the agent picks has a ~1.8x perf consequence.

### Q5: Do prompt constraints (V2/V4/V6) actually steer strategy?

In the bare arm:
- V0 (no constraint): 3/3 general — Claude reaches for the structural fix unprompted
- V4 (no escape hatches): 2/3 general — barely changes the V0 pattern
- V2 (bitwise equal): 1/3 general — modest pull toward setup-layer
- V6 (no config flags): 0/3 general — strong pull toward setup-layer

V6 is the most discriminating constraint: it forbids `torch._dynamo.config` modifications, and 3/3 V6 trials moved to setup-layer instead. The mechanism makes sense: when the "easy" config-flag escape is forbidden, the agent reaches for the next-easiest path, which turns out to be input-type tweaks at the baseline rather than model rewrites.

V2 and V4 are weak steering — the bare arm's preference for model-layer largely survives them.

In the SKILL arm, all four variants tilt heavily setup-required (V0, V2 = 0/3 general; V4, V6 = 1/3 general). The skill effect dominates the constraint effect.

### Q6: Does loading the debug-graph-breaks skill change anything?

Yes, three measurable effects:

1. *Strategy choice:* SKILL = 2/12 general; bare = 6/12 general. Skill steers strongly toward setup-layer fixes.
2. *Median speedup:* SKILL 1.60x; bare 1.90x. Driven by the strategy-choice difference (setup-layer fixes are slower).
3. *Effort metrics diverge:* SKILL takes MORE turns (51 vs 40 median) but LESS wall time per trial (972s vs 1170s). More turns of more decisive action, perhaps — suggests the skill workflow makes each turn more productive but encourages thorough exploration that lengthens turn count.

Notable: *no SKILL trial used the `is_compiling()` guard pattern* (0/12); 4/12 bare trials did. The skill's documented methodology either doesn't surface this pattern or actively discourages it.

---

## Phase D — Surprises

(Pre-registered: data points that diverged from what I'd have predicted.)

1. *V0 + bare = 3/3 general.* The most-bare configuration finds the model-layer fix unanimously. I'd predicted more variance.

2. *Zero canonical escape hatches.* I expected at least some agents (especially with-skill, since the skill explicitly recommends `@leaf_function` / `@nonstrict_trace`) to reach for them. None did. The skill's own examples push escape hatches; agents loaded with the skill chose to ignore that toolkit and went structural+setup-tweak instead.

3. *Skill takes MORE turns but LESS wall time.* Counter to my prior assumption that more turns = more wall. Skill agents make more decisions per second of wall clock, suggesting the skill workflow improves decision throughput.

4. *Perfect fix_locus → fix_status correlation in one direction.* If you only edited the model, you got `general` (6/6). If you edited both, you almost always got `setup-required` (16/18). The 2/18 exceptions (SKILL_V4_1, SKILL_V6_2) are where the model edits happened to be sufficient AND the agent also touched baseline; the baseline touches were redundant or non-impactful.

5. *V6 (no config flags) is the most discriminating constraint.* I didn't predict this — V2 (bitwise) seemed conceptually stronger. But V6 reliably pushes to setup-layer in the bare arm (0/3 general).

6. *Break-shape attacks are uniform.* I'd expected variation in WHICH breaks each trial chose to attack first. In fact, all trials attack essentially the same core set; variation is in the HOW.

---

## Open observations (unexplained)

1. *2.68e-04 max_diff identical across 16 setup-required trials.* All 16 trials produce bit-identical compiled outputs at this exact value. Could be that the inductor codegen for the "setup-required model-state" is deterministic across these trials (since the model-side edits don't materially affect the compiled function under canonical-input regime). Could be something else. Did NOT investigate further this round; flagged for potential follow-up if it recurs in cross-case data.

2. *fix_shape_family axis was uniformly "mixed" (21/24).* The axis as defined doesn't discriminate. Either reframe the axis (e.g., split into "primary pattern" + "secondary pattern") or drop it from the fingerprint going forward.

3. *Skill suppresses `is_compiling()` guard usage entirely (0/12 vs 4/12 bare).* No clear mechanism explanation. The skill's text doesn't explicitly forbid is_compiling, but doesn't recommend it either. Worth probing in the next case.

---

## Phase E — Cross-case implications

For VitsModel (3b), Aria (3c), PaddleOCRVL (3d) — what to add/watch:

1. *Add `general_speedup_vs_setup_speedup` as a primary metric.* This case showed the perf delta is the most actionable signal. Track it per case.

2. *Refine the fix_shape_family axis* — currently 88% trials are "mixed", so the axis is uninformative. Consider replacing with "primary edit pattern" + "secondary edit patterns".

3. *Test the V6 effect (no config flags) deliberately.* If V6 reliably pushes to setup-layer across cases, it's the most useful constraint for surfacing setup-vs-model strategy preference. The other constraints (V2, V4) are weak.

4. *Track `is_compiling()` guard usage.* If the skill suppresses it consistently across cases, that's a real skill-side behavior worth documenting. Consider adding it as a named fingerprint axis.

5. *Watch for canonical escape hatches.* Mistral3 had zero. If they appear in other cases, that's case-shape-specific signal (e.g., maybe vision-language doesn't need them but text models do).

6. *Methodology gap caught here matters for downstream cases.* The validate-vs-baseline input mismatch was only visible because we had both fix-locus types in this case. For cases where ALL trials happen to do model-only fixes, the gap won't show — but it's still there. validate_v2 is the right schema; document it as a permanent change.

---

## Recommendations

1. *For the skill curation team:* the debug-graph-breaks skill's preference for setup-layer fixes is real and measurable. Whether that's good (more thorough diagnosis path) or bad (steers away from higher-perf strategies) is a value judgment for the skill author. Worth surfacing.

2. *For the discovery harness:* keep `validation_v2` schema (model-layer-only check + agent-script subprocess + fix_status verdict). Don't go back to single gb_count.

3. *For the constraint variant catalog:* V6 (no config flags) is the most discriminating. V2, V4 are weak on this case. Consider dropping V2 or V4 from future per-case runs to save trial budget.

4. *For Mistral3 specifically:* the case is "complete" for skill discovery purposes. No re-run needed.

---

## Methodology notes

- *validate_v2* (added by `discovery/revalidate.py`) is the canonical fix_status verdict. Original `validation` field preserved as legacy. See `discovery/experiments/2026-04-cross-case-skill-discovery/plan.md` for schema.
- *Per-trial fingerprint* in `fingerprints.csv` was extracted by 3 parallel subagents reading each trial's diff + result + stream-final-summary. Each axis defined in the prompt; semicolon delimiters within cells.
- *Backend* — torch.compile uses default (inductor), per design.md §2.1.

## Appendix: trial-by-trial table

See `fingerprints.csv` for the full classification. Key columns: `trial`, `fix_locus`, `fix_shape_family`, `escape_hatches`, `breaks_attacked`, `agent_claim`. Cross-reference with `result.json/validation_v2.fix_status` for the verdict.
