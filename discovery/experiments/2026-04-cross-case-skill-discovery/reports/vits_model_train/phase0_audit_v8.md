# Phase 0 Audit — vits_model_train V8 follow-up (run 20260426-014253)

**Verdict:** Data is **trustworthy for fix_status / strategy fingerprinting** (the verdicts that drive the analysis). Perf data is **partially unreliable** — 4/6 trials have nan/missing perf (same `_measure_case.py` infra bug as the original 24-trial run, plus one trial with perf step entirely skipped). The V8 fix_status finding (6/6 `general`) is reliable and is the headline.

Phase A may proceed with caveats noted in the per-question scope below.

## Setup recap

6 trials = 2 skill arms × V8 × N=3, run 2026-04-25 19:38–21:49 ET. V8 was added evening of 2026-04-25 in response to the original 24-trial run showing 12/12 `setup-required` on V0/V2/V4 (door-closing for the setup-edit attractor). Run dir: `/tmp/discovery-runs/vits_model_train/20260426-014253/`. Summary: `summary.json` present, 6 entries.

## Check 1 — Artifact completeness ⚠️

5/6 trials have all 11 expected artifacts. **Exception:** `debug_graph_breaks_V8_2` is missing all 4 perf logs (`perf_stdout.log`, `perf_stderr.log`, `perf_tier2_stdout.log`, `perf_tier2_stderr.log`). Other artifacts present (agent_diff.patch, result.json, prompt.txt, validation logs, stream.jsonl, claude_stderr.log).

Mechanism: agent timed out at 1800s; result.json was written at 21:19 (right after validation); stream.jsonl kept being flushed until 21:38; perf step appears to have been skipped entirely without producing logs. Different failure mode from the canonical perf-infra bug (which produces logs + a `RuntimeError` in `perf.error`). Worth noting as a separate edge case.

## Check 2 — `.original` backup integrity ✓

`/tmp/discovery-runs/vits_model_train/modeling_vits.py.original` md5 = `13c324dc3517f5c0b6fe941c42550e19` matches pristine `~/envs/torch211/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py` — same as the original 24-trial audit. No post-hoc tool mutated the backup between runs.

## Check 3 — Trial completion sanity ⚠️ (timeout pattern is real signal)

**5/6 trials hit the 1800s wall budget (`agent_exit_code=124`).** Only `noskill_V8_2` completed naturally (`exit=0` at 1616s); `debug_graph_breaks_V8_3` was the only with-skill trial below the wall (`exit=0` at 1696s — barely under).

Per-trial:

| Trial | exit | elapsed | flags |
|---|---|---|---|
| `debug_graph_breaks_V8_1` | 124 (timeout) | 1800s | `file-mutated:modeling_vits.py` |
| `debug_graph_breaks_V8_2` | 124 (timeout) | 1800s | `file-mutated:modeling_vits.py` |
| `debug_graph_breaks_V8_3` | 0 | 1696s | `file-mutated:modeling_vits.py` |
| `noskill_V8_1` | 124 (timeout) | 1800s | `file-mutated:modeling_vits.py`, `diff-promoted-post-validate` |
| `noskill_V8_2` | 0 | 1616s | `file-mutated:modeling_vits.py` |
| `noskill_V8_3` | 124 (timeout) | 1800s | `file-mutated:modeling_vits.py` |

Validation runs *post-agent* in the runner harness, so timeouts do not invalidate the verdict — `fix_status` is read from the agent's final state of `modeling_vits.py`. All 6 trials produced a complete `validation.fix_status` field.

**Timeout pattern is interpretable:** agents found the model-layer fix early but kept iterating (probably refining, optimizing, or hitting repeated tool calls). They didn't crater. Phase A should record per-trial when (which turn) the agent first achieves gb=0 to separate "fix attained early then ran out clock" from "fix attained at the buzzer."

No anomalous `validate-crashed`, `perf-parse-error`, or `watched-file-missing` flags. The `diff-promoted-post-validate` flag on `noskill_V8_1` is benign (means the post-validate diff was larger than the pre-validate one — agent kept editing past validation's snapshot, expected when the trial runs to timeout).

## Check 4 — Internal consistency ✓

`summary.json` is a list of 6 entries, each carrying the same `validation.fix_status` as the corresponding `result.json`. Cross-check: 6/6 match (all `general`). Validate.py was the canonical reader for both — no divergence between summary aggregation and per-trial result.

`validation_v2.json` has NOT been run on V8 (post-hoc `revalidate.py` step not yet invoked). Not a blocker — `validation.fix_status` already comes from the runner's validate.py call which uses the same canonical inputs. validation_v2 would be a redundant sanity check at this stage.

## Check 5 — Reasonable value bounds ⚠️

Per-trial:

| Trial | fix_status | gb agent→canon | max_diff | sp_t1 | sp_t2 | eager_self_diff | eager_det |
|---|---|---|---|---|---|---|---|
| `debug_graph_breaks_V8_1` | general | 0→0 | None | nan | nan | None | None |
| `debug_graph_breaks_V8_2` | general | 0→0 | None | None | None | None | None |
| `debug_graph_breaks_V8_3` | general | 0→0 | None | nan | nan | None | None |
| `noskill_V8_1` | general | 0→0 | 2.000 | 1.828 | 1.564 | 2.000 | False |
| `noskill_V8_2` | general | 0→0 | 2.000 | 1.756 | 1.461 | 2.000 | False |
| `noskill_V8_3` | general | 0→0 | None | nan | nan | None | None |

- `fix_status`: **6/6 general** — every trial achieved a model-layer-only fix that produces gb=0 under canonical inputs. Headline finding.
- `gb_in_agent_run` and `gb_under_canonical_inputs` both 0 across the board — agent's setup matches canonical, no divergence to explain.
- `max_diff_compiled_vs_eager` = 2.0 on the 2 valid-perf trials, consistent with the train-mode dropout noise floor identified in the original audit (Methodology Gap 2). Confirms compile didn't change the math.
- Speedup: tier-1 ∈ [1.756, 1.828], tier-2 ∈ [1.461, 1.564] on the 2 valid-perf trials. In bounds, both speedups > 1.4×. Suggests model-layer-only fixes preserve perf despite being more constrained than setup-edit shortcuts.
- `eager_self_diff = 2.0`, `eager_deterministic = False` — same dropout signature as the original run (Methodology Gap 2 still applies; train-mode noise floor is real).
- 4/6 trials lack perf data: 3 (`debug_graph_breaks_V8_1`, `debug_graph_breaks_V8_3`, `noskill_V8_3`) hit the canonical `_measure_case.py` infra bug (`RuntimeError: tensor size mismatch`); 1 (`debug_graph_breaks_V8_2`) had perf step entirely skipped (see Check 1).

## Check 6 — Stream integrity ✓

6/6 `stream.jsonl` files parse cleanly to last line. Last event type = `result` for all. Line counts: 10662–33868 (no truncation).

---

## Caveats to carry into Phase A and findings

1. **Q4 (perf preserved?)** — answered on 2/6 V8 trials only (both noskill arm). The 2 valid samples both show > 1.4× speedup, suggesting model-layer-only fixes preserve perf, but n=2 is thin. Document the per-cell coverage table; flag the with-skill arm as unanswered for V8 perf.
2. **Train-mode max_diff = 2.0 is dropout noise floor** (same caveat as original audit). The `general` verdict on V8 trials is math-correct.
3. **5/6 timeouts are V8-specific signal, not infra failure.** Phase A should examine per-trial: when did the agent first achieve `gb=0`? If they all reached the fix early and then iterated until killed, that's "agent found the fix, kept polishing" — not crater. If some only reached `gb=0` at the wall, that's "barely-in-time fix."
4. **`debug_graph_breaks_V8_2` perf-skip is a separate edge case** from the canonical perf-infra bug. Worth noting in the followup loop tracking perf-infra issues but doesn't block analysis.
5. **The V0/V2/V4 perf-infra bug and the determinism gap are the same two issues already filed as in-flight loops** — V8 inherits both, no new issues surfaced.

## Headline (preview, not the analysis itself)

V8 produced what V8 was designed to surface: agents *can* find model-layer-only fixes when the setup-edit door is closed. 6/6 `general` is qualitatively different from the original 24-trial result (19/24 setup-required, 4/24 general all on V6). The "shut the door to shortcut solutions" methodology produced its first cross-arm-uniform model-layer outcome.

The 5/6 timeout rate is the headline tension: agents found the fix but didn't terminate cleanly. Phase A needs to characterize *when* they reached the fix relative to the wall.
