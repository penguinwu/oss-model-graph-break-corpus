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
| M1 | G1 — register expected signal in `known_errors.json` / `skip_models.json` (14 D1 train-mode divergences + 4 Qwen3_5*TextModel worker_errors) | ⏳ | otter | Without this, next sanity-check STRICT_FAILs on G1. Now mechanically checkable via `tools/check_cohort_invariants.py --post-sweep <results>` (built X1.1) |
| M2 | Run 20-random sample-sweep pre-flight gate on regenerated cohort + right stack | ⏳ | otter | Now scriptable via `tools/sample_cohort.py --n 20 --output <sample>` (built X1.1). Workflow: sample → run identify on sample → check_cohort_invariants → if green, full sweep |

## Should-close before launch

| # | Item | Status | Owner | Notes |
|---|---|---|---|---|
| S1 | Diagnose Qwen3_5*TextModel worker_error root cause | ⏳ | otter | Real bug, undiagnosed; informs M1 (register-as-known vs fix-locally) |
| S2 | File 14 D1 catastrophic train-mode divergences as upstream PyTorch issues | ⏳ | otter | Use `tools/file_issues.py pytorch-upstream`; can batch by family (audio/seq2seq/Reformer) |

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

## Suggested order of operations (for reference)

1. X1 + X2 reviewed by Peng (current blocker) → invoke adversary, encode guardrail
2. S1 (Qwen3_5* diagnosis) → informs M1
3. M1 (known_errors / skip_models registration)
4. S2 (D1 upstream issue filing) — parallel-safe
5. M2 (20-sample pre-flight gate)
6. Launch full NGB verify

## Closure log

(Items move here when status → ✅. Keeps the active table small.)

(none yet)
