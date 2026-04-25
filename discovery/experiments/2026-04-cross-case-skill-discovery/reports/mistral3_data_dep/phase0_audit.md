# Mistral3 Case 3a — Phase 0 Audit (Data Trustworthiness Gate)

**Run dir:** `/tmp/discovery-runs/mistral3_data_dep/20260425-041832/`
**Trials audited:** 24
**Audit date:** 2026-04-25 (retroactive — backfilled after the case wrap; original audit was done in conversation but not captured as a file)
**Verdict:** **CLEAN — no serious issues; analysis stands.**

This is the Phase 0 audit prescribed by `discovery/skills/per-case-analysis/SKILL.md`. Phase 0 establishes that the trial data is sound enough to draw conclusions from. Phase A (per-trial fingerprint) and beyond depend on Phase 0 passing.

## Check 1 — Artifact completeness

For every trial dir, the required files are present and non-zero where it matters:

- `agent_diff.patch` — present, non-zero in all 24 trials
- `result.json` — present, non-zero in all 24 trials
- `prompt.txt` — present, non-zero in all 24 trials
- `stream.jsonl` — present, non-zero in all 24 trials
- `claude_stderr.log` — present in all 24 trials (zero-length in some, which is benign — means the agent ran without warnings to stderr)
- `validation_*.log`, `perf_*.log` — present in all 24 trials

**Verdict:** PASS. 0/24 trials missing required artifacts. 0/24 trials with zero-length critical files.

## Check 2 — `.original` backup integrity (vs pristine source)

The discovery harness saves a `.original` copy of each watched source file at run setup, so the agent's diff can be cleanly applied/reverted. Critical: prior post-hoc tools (an early version of `revalidate.py`) once mutated `.original` files via `patch -p0` — that bug was caught and the files were recovered, but the audit needs to verify recovery succeeded.

`.original` files in `/tmp/discovery-runs/mistral3_data_dep/`:

| File | SHA256 | Pristine source SHA256 | Match? |
|---|---|---|---|
| `modeling_mistral3.py.original` | `9886c6ef55239e37c27f79b05d779e43c155902d44775719d9dc45a0f64ac880` | `9886c6ef55239e37c27f79b05d779e43c155902d44775719d9dc45a0f64ac880` | ✅ |
| `modeling_pixtral.py.original` | `e713605fcc0623cb49b2c25b96ae7d7d385fb6e4e2553e54871e3b5a32f9a699` | `e713605fcc0623cb49b2c25b96ae7d7d385fb6e4e2553e54871e3b5a32f9a699` | ✅ |
| `baseline_mistral3.py.original` | `64f92d3fb8de03b7b5b9f16e6c6d9955a4a0197f00007db09dbcd14de30b082b` | (case-specific, no upstream) | n/a |

Pristine reference: `transformers` package source at `/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/{mistral3,pixtral}/modeling_*.py` — the same env the trials ran in.

**Verdict:** PASS. The two upstream `modeling_*.py.original` files are byte-identical to the installed transformers source. The recovery from the earlier `revalidate.py` mutation succeeded; no post-hoc corruption remains. The case-specific `baseline_mistral3.py.original` has no upstream pristine to compare against (it's authored per-case), but its SHA is recorded above as a future cross-check anchor.

## Check 3 — Trial completion anomaly flags

Per the SKILL: serious flags (`validate-crashed`, `perf-parse-error`, `watched-file-missing`) gate the analysis; benign flags (`file-mutated:*`) are expected when the agent edited.

- **Serious flags:** 0 trials
- **Benign `file-mutated:*` flags:** 66 instances across 24 trials (median ~3 per trial — agents typically mutated 2–3 of {modeling_mistral3.py, modeling_pixtral.py, baseline_mistral3.py})

**Verdict:** PASS.

## Check 4 — Internal consistency (validation_v2 vs legacy validation)

The legacy `validation` field in `result.json` already replays under canonical inputs (the case's standard input set), so its `graph_break_count` should equal `validation_v2.details.gb_under_canonical_inputs`. (The `validation_v2` schema additionally surfaces `gb_in_agent_run`, which was not previously broken out.)

Cross-checked all 24 trials: **0 mismatches.** Examples:

| Trial | legacy `graph_break_count` | v2 `gb_under_canonical_inputs` | v2 `fix_status` |
|---|---|---|---|
| `noskill_V0_1` | 0 | 0 | general |
| `noskill_V2_2` | 11 | 11 | setup-required |
| `SKILL_V0_1` | 12 | 12 | setup-required |
| `SKILL_V6_2` | 0 | 0 | general |

**Verdict:** PASS. The two schemas agree on all 24 trials.

## Check 5 — Reasonable value bounds

Per the SKILL:

- `speedup` ∈ (0, 100)
- `max_diff_compiled_vs_eager` ∈ [0, 1)
- `elapsed_s` ∈ (60s, timeout+5min) — i.e. (60s, 2100s) for the 1800s budget

Cross-checked all 24 trials' `perf.speedup`, `validation.max_diff_compiled_vs_eager_now`, `elapsed_s`. **0/24 outside bounds.**

Observed ranges in this dataset:
- speedup: 1.39x – 3.18x (tier-1)
- max_diff: 0 – 2.68e-04
- elapsed: 603s – 1720s

**Verdict:** PASS. All values within expected bounds.

## Check 6 — Stream integrity

For each `stream.jsonl`: parse cleanly to last line, and the last event must be of type `result` (the agent's final summary). A truncated stream would mean the trial died mid-flight.

Cross-checked all 24 trials: **0 broken streams.** Every stream parses cleanly and ends with a `result` event.

**Verdict:** PASS.

## Audit summary

| Check | Verdict | Notes |
|---|---|---|
| 1. Artifact completeness | ✅ PASS | 0/24 missing |
| 2. .original integrity | ✅ PASS | upstream files byte-identical to installed transformers |
| 3. Anomaly flags | ✅ PASS | 0 serious, 66 benign (expected) |
| 4. validation_v2 vs legacy consistency | ✅ PASS | 0 mismatches |
| 5. Value bounds | ✅ PASS | 0 out-of-bounds |
| 6. Stream integrity | ✅ PASS | 0 truncated/malformed |

**Overall verdict: CLEAN.** Phase A (per-trial fingerprint), Phase B (aggregations), and Phases C/D/E in [`findings.md`](findings.md) all proceeded on data that passes Phase 0.

## Caveats

- **Audit was retroactive** (backfilled 2026-04-25 after the case wrap). The original conversation-time audit was done by reading individual artifacts and spot-checking; this file documents the same checks as a re-runnable script. Future cases will produce `phase0_audit.md` at the time of analysis, not after.
- **No `.original` for the case-specific `baseline_*.py`** can be cross-verified against an upstream pristine source — by design, baseline_*.py is authored per-case and has no canonical upstream. Its SHA is recorded above as a future-cross-check anchor: if anyone modifies it, the SHA will change and re-running this audit will surface the difference.
