---
plan: ngb-verify-launch
status: active
owner: otter
created: 2026-05-07
last_check: 2026-05-07
---

# NGB Verify — Pre-Launch Tracking

Forward-looking tracking list for items that must / should be addressed before the canonical NGB verify pass on the regenerated cohort + right stack. Companion to the postmortem at `experiments/2026-05-06-ngb-verify-postmortem.md` (which is closed/historical).

Created 2026-05-07 at Peng's request: "I am nervous that we may lose track of things on this list." This file IS the track list.

## Status legend

- ⏳ = pending (not started)
- 🚧 = in progress
- ✅ = done
- ⏸️ = deferred (with reason)
- 🚫 = blocked (with blocker)

## Must-close before launch

| # | Item | Status | Owner | Notes |
|---|---|---|---|---|
| M1 | G1 — register expected signal in `known_errors.json` / `skip_models.json` | 🚫 PAUSED | otter | **Paused 2026-05-07 14:23 ET (Peng).** Qwen3_5* portion already moot per S1. D1 portion paused: no registration until D1 divergences are reproduced on a canonical NGB verify run. Reasoning: Run 2 was non-canonical (broken cohort); selective trust of D1 ("those models should have been in cohort → divergence must be real") is the same class of selective interpretation that produced the original failure. |
| M2 | Run 20-random sample-sweep pre-flight gate on regenerated cohort + right stack | ⏳ | otter | Now scriptable via `tools/sample_cohort.py --n 20 --output <sample>` (built X1.1). Workflow: sample → run identify on sample → check_cohort_invariants → if green, proceed to canonical full NGB verify. **This is now THE primary path forward** — without a clean run, every result from Runs 1+2 stays provisional. |

## Should-close before launch

| # | Item | Status | Owner | Notes |
|---|---|---|---|---|
| S1 | Diagnose Qwen3_5*TextModel worker_error root cause | ✅ | otter | Diagnosed 2026-05-07 13:50 ET as env-induced cuDNN library loading flakiness (`Unable to load any of {libcudnn_graph.so.9...}`). Auto-retry handles it; final identify_results.json shows 0 worker_errors. NOT a real bug, NOT a regression. Postmortem updated with diagnosis addendum. |
| S2 | File 14 D1 catastrophic train-mode divergences as upstream PyTorch issues | 🚫 PAUSED | otter | **Paused 2026-05-07 14:23 ET (Peng).** Run 1 + Run 2 are both non-canonical; D1 divergences observed there are PROVISIONAL until reproduced on a canonical run. Filing upstream based on suspect data wastes maintainer time and our credibility. Will revisit after M2 + canonical full verify produce clean reproductions. |

## Meta — system hardening (this morning's adversary-review work)

| # | Item | Status | Notes |
|---|---|---|---|
| X1 | Adversary review of cohort-regen fix bundle | ✅ | Invoked 2026-05-07 12:41 ET (case_id `2026-05-07-124100-cohort-regen-fix`); 9 gaps + 12 tests surfaced. Full output preserved at `/tmp/raw_output_cohort_regen.txt` (sha256 `5f4a88d4...`); summary in `skills/adversary-review/reviews_log.md`. |
| X1.1 | Address all 9 gaps + 12 tests | ✅ | Built `sweep/cohort_validator.py` + `tools/sample_cohort.py` + `tools/check_cohort_invariants.py` + 35 net-new tests + skill drift fix + regression-test fail-loud + Python-version guards. Full suite 55/55 PASS. See log dispositions per gap. |
| X2 | External engagement strong-guardrail rule encoded in local CLAUDE.md | ✅ | Approved 2026-05-07 12:35; encoded in local CLAUDE.md; supersedes prior feedback-space thread-reply leniency. |

## Process-discipline (encoded but unproven)

| # | Item | Status | Notes |
|---|---|---|---|
| P1 | Approval-Triggered Action Discipline rule (in-conversation directive persistence) | ✅ encoded; ⏳ unproven | Local CLAUDE.md addition this morning. Track whether it actually holds for next launch. |

## Suggested order of operations (updated 2026-05-07 14:23 ET after Peng's pause directive)

1. ✅ X1 + X2 (adversary review + external engagement guardrail)
2. ✅ X1.1 (address all 9 adversary gaps + 35 net-new tests)
3. ✅ S1 (Qwen3_5* diagnosis = env cuDNN flakiness, no action needed)
4. ✅ Pre-push test hook installed (commit 51edd13)
5. **NEXT:** M2 (run 20-random sample-sweep pre-flight using new tooling). This is now the only forward path — without a clean run, every conclusion from Runs 1+2 is provisional. Sweep launch follows `skills/sweep.md` §4 + `skills/test-sweep-changes/SKILL.md` 5 gates.
6. After M2 PASS: launch canonical full NGB verify on regenerated cohort + right stack
7. After canonical verify PASS: re-evaluate D1 divergences — if they reproduce, M1 + S2 unpause; if they don't, the postmortem's "Load-bearing data" section was a Run-2 artifact and we update accordingly.

**Blocked items (paused per Peng 2026-05-07 14:23):**
- S2 (file upstream D1 issues) — paused pending canonical reproduction
- M1 (register D1 in known_errors.json) — paused pending canonical reproduction

## Closure log

| date | item | outcome |
|---|---|---|
| 2026-05-07 12:35 | X2 external engagement guardrail | encoded in local CLAUDE.md; supersedes prior thread-reply leniency |
| 2026-05-07 12:41 | X1 adversary review of cohort-regen fix | 9 gaps + 12 tests surfaced |
| 2026-05-07 13:30 | X1.1 fix all 9 gaps | 2 commits pushed (8b6df53 + 90ad1b4); 35 net-new tests; 55/55 PASS |
| 2026-05-07 13:50 | S1 Qwen3_5* worker_error diagnosis | env-induced cuDNN flakiness; auto-retry handles; postmortem updated; M1 Qwen3_5* portion moot |
| 2026-05-07 14:00 | Pre-push test hook | scripts/pre-push installed (commit 51edd13); replaces discipline-only test-running |
| 2026-05-07 14:23 | S2 + M1 PAUSED by Peng | broken-process runs → all results suspect; D1 divergences PROVISIONAL until canonical reproduction; postmortem updated with trust-update addendum |
