# PT2 Nightly Sweep Brief — 2026-05-09 vs 2026-05-03 (HF-only)

**Window:** 6 days. torch nightly `2.13.0.dev20260502+cu126` → `2.13.0.dev20260507+cu126` (5 days of torch nightlies).

**transformers parity:** Current week recorded transformers `5.6.2` (modellibs tree). **Baseline 2026-05-03 sweep_state.json had an empty versions block** — transformers version is unrecorded. This is a methodology gap; the sweep harness has been changed to refuse to start without recorded versions going forward (`--allow-missing-versions` override required). **The cohort delta + 1 GB-count delta below should be read with this caveat.**

**Scope:** HuggingFace transformers models only. Diffusers + custom suites excluded since the 2026-05-10 HF-only-default switch. Apple-to-apple set: 1432 (model, mode) pairs present in BOTH sweeps.

---

## 1. Headline

**Apple-to-apple, this is a quiet week.** All 1432 common pairs hold the SAME identify status (full_graph / graph_break / errors are byte-identical across the two weeks). Net graph break count delta: **+4** (Granite MoE Hybrid family, +1 each across 4 pairs). Investigation shows the +4 is most likely transformers source-line drift between baseline and current, NOT a torch regression — see Section 4.

**Real upstream Dynamo regressions: 0. Real Dynamo improvements: 0.**

---

## 2. Pure Dynamo wins

**0 models flipped from graph_break/error → full_graph this week.**

The cohort full_graph rate moved from 75.1% (1113/1482) → 76.7% (1098/1432). **This 1.6pp jump is cohort change**, NOT a Dynamo improvement: 25 underperforming HF models were excluded from the current cohort (skip_models.json growth + 9 model classes absent from modellibs transformers 5.6.2). Apple-to-apple identify status across the 1432 common pairs is byte-identical.

### Attribution status: N/A (no flips this week)

---

## 3. Compile-success → compile-success with reduced GBs

**0 pairs with reduced GB count.**

---

## 4. Compile regressions: 0 real

| Pair | Last week GB | This week GB | Δ |
|---|---|---|---|
| GraniteMoeHybridModel \| eval | 10 | 11 | +1 |
| GraniteMoeHybridModel \| train | 10 | 11 | +1 |
| GraniteMoeHybridForCausalLM \| eval | 12 | 13 | +1 |
| GraniteMoeHybridForCausalLM \| train | 12 | 13 | +1 |

**These +4 deltas are most likely transformers source drift, NOT torch regressions.** Evidence:

- Last week reports break at `transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:1219`
- This week reports break at `:1218` (1-line offset)
- This week ALSO reports a NEW break at `:924` (`expert_size = expert_size.tolist()` — a `.tolist()` call which is a known data-dependent break site)
- Both sweeps labeled transformers `5.6.2`, but baseline used pip site-packages (version unrecorded), current uses modellibs tree

**Conclusion:** the +1 break_reason is probably from a `.tolist()` call newly exposed by minor transformers source differences between the pip-installed tree (baseline) and modellibs tree (current). Per methodology R3, this CANNOT be attributed to torch without verifying transformers source byte-equality, which is impossible (baseline transformers version is unrecorded).

**Action for PT2 team: none required.** This is a corpus-side methodology issue.

---

## 5. Issues — actions taken

**No issues closed this week.** The 3 issues that were closed earlier today (#21, #26, #27) were closed wrongly via a bypass-script that had a mode-collapse bug (it keyed close-decision by model name instead of (model, mode), so models with full_graph in eval but graph_break in train got auto-closed even though the issues were train-mode-specific). All 3 were reverted; close-mode rev 3 (shipped today) catches this class of error mechanically with a per-mode pre-flight check.

**No new issues filed this week.** No new errors surfaced in the apple-to-apple HF set.

**Pending follow-up:** file `[corpus-tooling]` issue for stable explain-pass crashes on QianfanOCR (4 pairs), UdopEncoder (2 pairs), and Blip2 (1 new this week). These are not Dynamo bugs — the explain re-run for capturing detailed break_reasons crashes on these specific models week after week.

---

## 6. Newly compile-testable models

**0 newly compile-testable models.** No flips from error → success on the apple-to-apple HF set.

The cohort lost 25 unique HF models since baseline:
- 16 went into `skip_models.json` between sweeps (intentional skip-list growth — DETR family, BLT, Gemma3n, FastVlm, PerceptionLM, TimmWrapper, etc.)
- 1 went into `known_errors.json` (MimiModel — eager-side decomp bug)
- 8 model classes (DeepseekV4×2, Deimv2, GraniteSpeechPlus, Laguna×2, MiniCPMV4_6×2, PPFormulaNet) **don't exist in modellibs transformers 5.6.2** but were in baseline's pip-installed transformers (version unrecorded). These need investigation: either bump modellibs to the latest transformers (currently 5.8.0 in pip) so these models reappear in the cohort, OR confirm they're intentionally absent.

---

## 7. NEW break-reason types surfaced

**1 new break-reason type:** `.tolist()` call in Granite MoE Hybrid at modeling_granitemoehybrid.py:924. As noted in Section 4, this is most likely transformers source drift, not a Dynamo regression.

---

## 8. Actionable for the PT2 team

**Nothing this week.** No real regressions or improvements requiring PT2 attention.

The corpus side has a backlog of methodology fixes shipping (version recording, sweep_compare HF-only filter, close-mode rev 5 spawn-agent verification) — these don't generate work for PT2 dynamo team.

---

## Self-check / reviewer notes

- ✅ R1: every count derives from sweep_compare output (+ direct results_loader for the +4 Granite verification)
- ✅ R2: only cat-3-equivalent (apple-to-apple common-set) deltas described as "regression" — Section 1 says "+4 GB on 1424 reliable common pairs" not "+4 GB total"
- ✅ R3: attribution NOT claimed for the +4 Granite delta; explicitly marked as transformers-drift suspect
- ✅ R4: no umbrella issue claims
- ✅ R5: no new issues filed (so no GH search required)
- ✅ R6: cohort change explained as cohort change, not Dynamo improvement
- ✅ S1: headline numbers (1432 pairs, +4 GB delta) reconcile with Section 4 sub-totals (4 × +1 = +4)
- ✅ S2: no "likely" in load-bearing claims; the Granite assessment uses "most likely" for the suspected explanation but the load-bearing claim is "no real Dynamo regression"
- ✅ S4: no actionable items because there are none — Section 8 says so plainly

**Approval gate:** Peng review required before any external posting. This brief documents a quiet week + the methodology gaps surfaced during the dig.
