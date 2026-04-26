# Phase 0 Audit — vits_model_train (run 20260425-144345)

**Verdict:** Data is **trustworthy for fix_status / strategy fingerprinting** (the verdicts that drive the analysis). Perf data is **partially unreliable** — 12/24 trials have nan perf due to an infra bug. Two methodology gaps surfaced; both are documented below and do not invalidate the run.

Phase A may proceed with caveats noted in the per-question scope below.

## Setup recap

24 trials = 2 skill arms × {V0, V2, V4, V6} × N=3, run 2026-04-25 14:43–~20:00 ET. All `agent_exit_code=0`. Run dir: `/tmp/discovery-runs/vits_model_train/20260425-144345/`.

## Check 1 — Artifact completeness ✓

All 24 trials have: `agent_diff.patch`, `result.json`, `prompt.txt`, `validation_stdout.log`, `validation_stderr.log`, `perf_stdout.log`, `perf_stderr.log`, `perf_tier2_stdout.log`, `perf_tier2_stderr.log`, `stream.jsonl`, `claude_stderr.log`. No missing artifacts.

## Check 2 — `.original` backup integrity ✓

`/tmp/discovery-runs/vits_model_train/modeling_vits.py.original` md5 = `13c324dc3517f5c0b6fe941c42550e19` matches pristine `~/envs/torch211/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py`. No post-hoc tool mutated the backup.

## Check 3 — Trial completion sanity ⚠️ (with caveats)

All 24 trials completed cleanly from the runner's POV (`agent_exit_code=0`, all flags benign — `file-mutated:modeling_vits.py` and `file-mutated:baseline_vits.py` are expected when agent edits). No `validate-crashed`, `perf-parse-error`, or `watched-file-missing` anomalous flags.

**However:** 12/24 trials have nan perf. Distribution:

| Cell | nan-perf count | Clean-perf count |
|---|---|---|
| noskill V0/V2 (6) | 0 | 6 |
| noskill V4 (3) | 1 (V4_3) | 2 |
| noskill V6 (3) | 2 (V6_2, V6_3) | 1 (V6_1) |
| with-skill V0/V2/V4 (9) | **9 (all)** | 0 |
| with-skill V6 (3) | 0 | 3 |

The with-skill arm V0/V2/V4 cells are completely perf-blocked.

## Check 4 — Internal consistency ⚠️ (`fix_status` reliable, perf decoupled)

`fix_status` verdict comes from `validate.py` (uses canonical inputs from the case file). Validate ran cleanly on all 24 trials. Verdicts: 4 `general` (all V6, both arms — V6_1/V6_2/V6_3 with-skill + V6_1 noskill), 19 `setup-required`, 1 `none` (noskill V6_3).

`perf` data comes from `_measure_case.py` subprocess (also uses canonical inputs from the case file) — but crashes on a tensor-shape mismatch on 12/24 trials. Validate and perf run independently, hence verdict-reliable / perf-broken decoupling is real.

`gb_in_agent_run` vs `gb_under_canonical_inputs` divergence is the expected setup-required signature: 0 in agent's setup, 1–6 under canonical inputs. Consistent across all setup-required trials.

## Check 5 — Reasonable value bounds ⚠️

- 12/24 trials: speedup = nan (out of bounds, see Check 3)
- 12/12 valid trials: speedup ∈ [1.013, 2.241] ✓
- max_diff: 0 (not measurable on nan-perf trials), 2.00 on all 4 `general` V6 trials. **2.00 looks alarming but is a noise-floor artifact** — see Methodology Gap 2.
- elapsed_s: all in (60s, 1800s+5min) ✓

## Check 6 — Stream integrity ✓

24/24 `stream.jsonl` files parse cleanly to last line. No truncation.

---

## Root cause — Infra Bug 1: perf measurement uses case inputs, not agent's edited setup

**Symptom:** 12 trials produce `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension 1` from `_measure_case.py`.

**Mechanism:** The agent's diff modifies `modeling_vits.py` to assume specific tensor shapes (e.g., hardcoding seq_len-derived dims). The agent also modifies `baseline_vits.py` to call the model with matching inputs — that's why their own validation log records gb=0. But `measure_perf` runs in a separate subprocess (`_measure_case.py`) that imports the case file (`discovery/cases/vits_model_train.py`) and calls `make_inputs(B=1, seq_len=16)` directly. It does **not** use the agent's edited `baseline_vits.py`. The model with hardcoded shape assumptions then fails to handle the case's canonical inputs.

**Why concentrated in with-skill arm:** the `debug-graph-breaks` skill teaches more aggressive shape-specialization techniques (hardcoded sizes, fixed-length unfold/conv1d operations), which more often produce model-layer edits incompatible with canonical inputs. The noskill arm tends to do less aggressive shape specialization, so its model edits more often happen to remain compatible.

**Fix proposal (separate from this audit, file as open loop):** `_measure_case.py` should either (a) execute the agent's edited `baseline_vits.py` to obtain inputs, or (b) `make_inputs` should be parameterized so the agent can declare the inputs they fixed for. Option (a) preserves the test as "what the agent built must run on the harness's canonical inputs"; option (b) admits the agent's setup as part of the fix surface. Decision needed before next case (3c Aria).

**Impact on this analysis:** Q4 ("does the fix preserve performance?") is only answerable on the 12 clean trials. Strategy fingerprinting (Q1, Q2, Q3, Q5, Q6) is unaffected — `agent_diff.patch`, `stream.jsonl`, `fix_status` all intact across 24/24.

## Root cause — Methodology Gap 2: `max_diff` lacks a train-mode noise floor

**Symptom:** All 4 `general` V6 trials report `max_diff_compiled_vs_eager = 2.00`. Initially looked like math is broken (case docstring sets 1e-3 tolerance).

**Mechanism:** `noskill_V6_1`'s perf log records `eager_self_diff = 2.0` and `eager_deterministic = false`. Train mode (`model.train()`) means dropout layers fire on every forward pass. Two consecutive eager forward passes on the same inputs/seed produce outputs differing by 2.0 — that's the noise floor. The `compiled_vs_eager` max_diff of 2.0 is consistent with this floor; compile didn't change the math.

**Implication:** the `general` verdict is valid. The fixes are mathematically correct.

**Fix proposal (separate, file as open loop):** `validation_v2` should record `eager_self_diff` (or `eager_deterministic` flag) alongside `max_diff_compiled_vs_eager`. For train-mode cases, the meaningful comparison is `max_diff_compiled_vs_eager - eager_self_diff` or a quantile-based check. The case docstring's 1e-3 tolerance is unmeetable in train mode and should be loosened or qualified. This affects any future train-mode case (current run is the first).

**Impact on this analysis:** the per-trial fingerprint table can record `max_diff` honestly but the findings doc must explicitly note that train-mode max_diff includes dropout noise floor.

---

## Caveats to carry into Phase A and findings

1. Q4 (perf preserved?) — answered on 12/24 trials (the clean cells). Document the per-cell coverage table.
2. The `general` verdicts on V6 (4 trials) are math-correct — the max_diff=2.0 is dropout noise floor, not a divergence.
3. The 9 with-skill V0/V2/V4 trials lack perf but have full diff/stream/fix_status — strategy fingerprint analysis unaffected.
4. Two infra/methodology issues filed as open loops for the next case (3c) — neither blocks this analysis.
