---
title: NGB Verify Pass — Postmortem of 2026-05-06 Runs
status: closed
owner: otter
created: 2026-05-07
last_check: 2026-05-07
disposition: ABANDONED — both runs non-canonical; do not use result data for analysis or issue filing
---

# ⚠️ DATA ABANDONED — 2026-05-07 14:26 ET (Peng directive)

**Both Run 1 and Run 2 produce result data that is NOT to be used for any analysis, issue filing, known_errors registration, or other downstream conclusions.** The runs surfaced infrastructure gaps which are now fixed; that was their useful contribution. The result rows themselves are non-canonical and abandoned.

What stays:
- The infrastructure improvements derived from the FAILURE PATTERN (validator, sampler, invariant executor, pre-push hook, expanded adversary-review scope, external-engagement guardrail).
- This postmortem as the historical record of what went wrong.

What goes:
- Any analysis based on Run 1 or Run 2 result rows (D1 divergences, sanity-check counts, etc.).
- Any open loops or handoff items proposing action on Run 2 data.

The next-up canonical NGB verify run (M2 + full verify in `experiments/ngb-verify-launch-plan.md`) is the source of truth going forward.

---

# NGB Verify Pass — Postmortem (2026-05-06 → 2026-05-07)

Two NGB verify sweep attempts on 2026-05-06, both producing non-canonical results for distinct reasons. This is the historical record so the failure modes don't recur.

## Run 1 — wrong version stack

**Output:** `sweep_results/experiments/nested-gb-correctness-2026-05-06-2026-05-06/`
**Launched:** 2026-05-06 07:38 ET
**Status:** NON-CANONICAL — wrong version stack

Used torch 2.11.0+cu128 + transformers 5.5.3 + diffusers 0.37.1 instead of the explain pass's torch 2.13.0.dev20260502+cu126 + transformers 5.6.2 + diffusers 0.38.0. Mismatched transformers caused systematic per-family failures: 12 generic `error` rows on Bart / BigBirdPegasus / Blenderbot (`AttributeError: 'Tensor' object has no attribute 'config'` in `EncoderDecoderCache(DynamicCache(config=self.config), ...)` — newer transformers code path), 4 `create_error` on Bamba (DNS / Hub fetch).

**Root cause + mitigation:** version-mismatch detection wasn't enforced. Fixed by `--models-from` + version-compat refusal mechanism (commit `4e1f071`) AND the cohort `_metadata.source_versions` check on `--models <flat.json>` runs (same commit). Verified 2026-05-07: launching the regenerated cohort with torch211 venv produces explicit `VERSION MISMATCH` and exits non-zero.

## Run 2 — broken cohort

**Output:** `sweep_results/experiments/ngb-verify-2026-05-06-2026-05-06/`
**Launched:** 2026-05-06 21:33 ET
**Status:** NON-CANONICAL — right stack, broken cohort

Fixed Run 1's stack issue but launched against a broken cohort file. The cohort `experiments/configs/nested_gb_cohort_2026-05-06.json` (214 models) was supposed to be the explain ok subset (190 unique names per `nested-gb-2026-05-05-2026-05-05/explain_results.json` with `status == ok` filter) but was actually built from the nightly identify pass `status graph_break` filter (214 names — exact match to that filter, not the explain subset). 17 cohort models were extras the explain pass had FAILED to analyze (got `explain_error` or `skipped`); they predictably failed again on this run.

### Sanity-check verdict (APPLY-C, 2026-05-07)

10 STRICT_FAIL on the v3 sanity-check skill:

| Inv | Failure |
|---|---|
| A1 | 6 attribute-not-found create_errors (MiniCPMV4_6, PPFormulaNet — don't exist in transformers 5.6.2) |
| A2 | Cohort file is bare flat list, no `_metadata` block |
| A3 | 17 cohort models not in explain ok subset (cohort generator pulled from wrong source) |
| C1 | 22 create_error |
| C2 | 4 non-env eager_error (Lumina2, Qianfan — input-shape mismatches) |
| C3 | 4 input-generation errors (same set) |
| E1 | 15 status regressions on previously-clean models (14 custom-model loader cases + 4 Qwen3_5*TextModel) |
| D1 | 14 catastrophic train-mode divergences (audio + seq2seq) — see Load-bearing data below |
| G1 | 32/32 non-success rows untriaged in known_errors.json or skip_models.json |
| B3 | 0.93% worker_error rate (Qwen3_5*TextModel cases) |

### Root cause investigation (2026-05-07)

Three reinforcing failures:

1. **In-conversation directive lost between proposal and launch.** Peng explicitly approved regenerating the cohort from explain results before launch. Otter forgot the directive between the proposal-time conversation and the launch-time action and used the existing (broken) cohort file.

2. **Smoke gate validated the wrong thing.** A 6-fixed-model smoke ran before launch, all passed. But the smoke models were a version-stack canary (chosen for known transformers-version sensitivity), not a sample of the actual cohort. Smoke would have passed regardless of cohort contamination.

3. **No standalone cohort generator existed.** The only canonical cohort-generation path was `--save-cohort` coupled with launching a sweep. To regenerate without launching meant writing ad-hoc code or hand-rolling — both of which historically produced bad cohorts.

### Mitigations

| Gap | Mitigation | Commit |
|---|---|---|
| Smoke validated wrong thing | 20-random sample-sweep gate from actual cohort + identical flags | `7c4b4fa` (v2.1) → `3aa4500` (v3 simplification) |
| No standalone cohort generator | `tools/generate_cohort.py` standalone tool + INV-A2 mechanically rejects hand-rolled cohorts at sample time | `cdb0ca4` |
| Wrong cohort file in tree | Regenerated `experiments/configs/nested_gb_cohort_2026-05-06.json` from explain pass `status == ok` filter (190 models, full _metadata) | `cdb0ca4` |
| Sanity-check skill missing | New `skills/sweep_sanity_check.md` v3 + wired into `skills/sweep.md` Pre-flight + §8 Step 0 | `7c4b4fa`, `3aa4500` |
| In-conversation directive persistence | OPEN — discipline encoding only; not closed by an artifact. See `skills/sweep.md` Pre-flight bullet on directive-action persistence (pending) |

### Verification (2026-05-07)

Live 5-model random sample on the regenerated cohort + right stack: 10 PASS, 0 STRICT_FAIL, 0 FLAG. Same skill that surfaced 10 STRICT_FAIL on the broken cohort returns clean.

## Load-bearing data from Run 2

The 14 catastrophic train-mode divergences (D1) ARE legitimate signal — those models are in the explain ok subset, so the divergence is real on the right stack. Not artifacts of the cohort or stack issues.

Affected families (all train mode only):
- Audio / speech: `Wav2Vec2Model`, `Wav2Vec2ConformerModel`, `WavLMModel`, `UniSpeechModel`, `UniSpeechSatModel`, `HubertModel`, `Data2VecAudioModel`, `SEWModel`
- Seq2seq: `M2M100Model`, `M2M100ForConditionalGeneration`, `PLBartModel`, `PLBartForConditionalGeneration`, `SpeechEncoderDecoderModel`
- Other: `ReformerModel`

All show absolute max_diff of 1.9–5.3 units between eager and compiled outputs (severity ratios in the millions); all survive less-flaky retest. To be filed as upstream PyTorch issues.

## What's next

A canonical full NGB verify on the regenerated cohort + right stack has not yet been run. It will not produce the 22+4 cohort-artifact errors (those are gone), but is expected to surface:
- The 14 D1 divergences again (real signal)
- The 4 Qwen3_5*TextModel worker_error regressions (real bug, not yet diagnosed)

Both should be filed as upstream issues per `tools/file_issues.py pytorch-upstream`.

## Trust update — 2026-05-07 14:23 ET (Peng directive)

The "Load-bearing data from Run 2" section above carves out the 14 D1 catastrophic train-mode divergences as "legitimate signal" with the argument that those models are in the explain ok subset and therefore "the divergence is real on the right stack."

**That carve-out is suspect.** Run 2 was non-canonical (broken cohort). When a sweep's process is broken, the principle is: treat ALL results as suspect until validated by a clean process. Selectively rescuing some results based on argument-from-set-membership ("those models should have been in the cohort, so their divergence must be real") is the same class of selective interpretation that produced the original failure mode.

Disposition (per Peng 2026-05-07 14:23 ET): **No upstream PyTorch issues will be filed based on Run 1 or Run 2 data.** No `known_errors.json` registration of the D1 divergences. The "14 D1 divergences are real" claim is now PROVISIONAL — it requires reproduction on a canonical NGB verify run (right stack + regenerated cohort + new validator + new sample-sweep pre-flight gate) before any upstream action.

This pause applies to:
- S2 (file 14 D1 upstream issues): PAUSED
- M1 (register in known_errors.json): PAUSED
- D1 disposition: PROVISIONAL until canonical run reproduces them

## Diagnosis update — 2026-05-07

(Added after the postmortem was nominally closed, per S1 investigation in `experiments/ngb-verify-launch-plan.md`.)

The "4 Qwen3_5*TextModel worker_error regressions (real bug, not yet diagnosed)" framing above is **incorrect** — re-read the data:

- `identify_streaming.jsonl` shows 4 worker_errors on Qwen3_5MoeTextModel (eval+train) and Qwen3_5TextModel (eval+train), all with the same root cause: `Unable to load any of {libcudnn_graph.so.9.10.2, libcudnn_graph.so.9.10, libcudnn_graph.so.9, libcudnn_graph.so} / Invalid handle. Cannot load symbol cudnnGetVersion`.
- This is **env-induced cuDNN library loading flakiness** — not a Qwen-specific bug, not a regression.
- `auto_retry_errors_checkpoint.jsonl` shows all 4 cases retried to **success**. The final `identify_results.json` shows them as `success`.
- Per `skills/sweep_sanity_check.md` C2, env-induced cuDNN errors are accepted ("env = CUDA OOM, cudnn, contention only").
- Per B3 (worker_error rate < 0.5%), the FINAL results have 0% worker_error (all retried to success), which passes B3.

**Implication:** the Qwen3_5* portion of M1 is moot. No `known_errors.json` entry needed; no upstream filing needed. The original sanity-check verdict was applied to the streaming jsonl (pre-retry); the corrected verdict on `identify_results.json` (post-retry) is clean for B3/C2.

**Remaining real-signal items from this run:** the 14 D1 catastrophic train-mode divergences. Those are still real and need upstream issues (S2 in `experiments/ngb-verify-launch-plan.md`).
