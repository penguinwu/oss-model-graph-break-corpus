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
| M1 | G1 — register expected signal in `known_errors.json` / `skip_models.json` (14 D1 train-mode divergences + 4 Qwen3_5*TextModel worker_errors) | ⏳ partial | otter | **Qwen3_5* portion MOOT** per S1 diagnosis (env-induced cuDNN flakiness; auto-retry handles; final results are clean for B3/C2). Remaining: register the 14 D1 divergences AFTER S2 files them upstream so we can reference the issue numbers. |
| M2 | Run 20-random sample-sweep pre-flight gate on regenerated cohort + right stack | ⏳ | otter | Now scriptable via `tools/sample_cohort.py --n 20 --output <sample>` (built X1.1). Workflow: sample → run identify on sample → check_cohort_invariants → if green, full sweep |

## Should-close before launch

| # | Item | Status | Owner | Notes |
|---|---|---|---|---|
| S1 | Diagnose Qwen3_5*TextModel worker_error root cause | ✅ | otter | Diagnosed 2026-05-07 13:50 ET as env-induced cuDNN library loading flakiness (`Unable to load any of {libcudnn_graph.so.9...}`). Auto-retry handles it; final identify_results.json shows 0 worker_errors. NOT a real bug, NOT a regression. Postmortem updated with diagnosis addendum. |
| S2 | File 14 D1 catastrophic train-mode divergences as upstream PyTorch issues | ⏳ | otter | Use `tools/file_issues.py pytorch-upstream`; batch by family (audio/seq2seq/Reformer). Each issue is EXTERNAL ENGAGEMENT — requires Peng's pre-post approval per the new strong guardrail. Otter must propose with verbatim content, target, audience, ask before posting. |

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

## Suggested order of operations (updated 2026-05-07 13:50 ET after S1)

1. ✅ X1 + X2 (adversary review + external engagement guardrail)
2. ✅ X1.1 (address all 9 adversary gaps + 35 net-new tests)
3. ✅ S1 (Qwen3_5* diagnosis = env cuDNN flakiness, no action needed)
4. **NEXT:** Either (a) S2 (prepare D1 upstream issue proposals for Peng's pre-post approval — strong-guardrail gated) OR (b) M2 (run 20-random sample-sweep pre-flight using new tooling — sweep launch, follows sweep.md §4 + test-sweep-changes 5 gates). Order is Peng's call.
5. M1 (register D1 divergences in known_errors.json with `pytorch_issue: <number>` after S2 files upstream)
6. Launch full NGB verify

## Closure log

| date | item | outcome |
|---|---|---|
| 2026-05-07 12:35 | X2 external engagement guardrail | encoded in local CLAUDE.md; supersedes prior thread-reply leniency |
| 2026-05-07 12:41 | X1 adversary review of cohort-regen fix | 9 gaps + 12 tests surfaced |
| 2026-05-07 13:30 | X1.1 fix all 9 gaps | 2 commits pushed (8b6df53 + 90ad1b4); 35 net-new tests; 55/55 PASS |
| 2026-05-07 13:50 | S1 Qwen3_5* worker_error diagnosis | env-induced cuDNN flakiness; auto-retry handles; postmortem updated; M1 Qwen3_5* portion moot |
