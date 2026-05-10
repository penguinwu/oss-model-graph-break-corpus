# PLAN.md — corpus project working plan

**Last updated:** 2026-05-10 11:40 ET (Otter)
**Current focus:** weekly sweep workflow build-out (Steps 1-2d, per Peng's process spec)

Read this FIRST when starting a session. Update after every completed task, every new task added, every scope change.

---

## WS1 — Weekly sweep workflow infrastructure

The standard repeated weekly process: Step 1 (run sweep) → Step 2a (new errors triage) → Step 2b (new models triage) → Step 2c (issue actions: close/edit/open) → Step 2d (brief report). Goal: autonomous with little Peng intervention.

- [x] Step 1: watchdog v3 simple design — stateless observer + tier-aware threshold + `.resume_in_flight` marker. Shipped 2026-05-10.
- [x] Step 1: auto-retry whitelist — only OOM/subprocess-crash/CUDA-assert retried. Shipped 2026-05-10. Design: `sweep/AUTO_RETRY_REFINEMENT.md`.
- [x] Step 1: HF-only default for nightly sweep — `tools/run_experiment.py` defaults to `--source hf`. Shipped 2026-05-10.
- [x] Step 1: skip refresh-nightly on `--resume` — saves 2-15 min per resume cycle. Shipped 2026-05-10.
- [ ] Step 1: per-model timeout propagation — design at `sweep/TIMEOUT_PROPAGATION_DESIGN.md`. **Pending Peng review.** Reminder cron `timeout-propagation-task-2026-05-10` fires Sunday 9 AM ET; surfaces design link to GChat for approval before implementation.
- [ ] Step 2a: `tools/audit_new_errors.py` — design + build + test on 2026-05-09 sweep. **Design pending.**
- [ ] Step 2b: `tools/audit_new_models.py` — design + build + test on 2026-05-09 sweep. **Design pending.**
- [ ] Step 2c: file-issue close-mode (`subagents/file-issue/` extension + `corpus-issue --close <num>`) — gated CLOSE workflow with adversary review + MRE re-verification (R3). **Design pending.** Replaces today's bypass-prone close-stale.
- [ ] Step 2d: `tools/compose_brief.py` — pre-fill 8-section brief from `sweep_compare.py` output. **Design pending.**

## WS2 — This week's sweep (2026-05-09)

Execution of WS1's workflow on real data. Each WS1 tool gets tested here.

- [x] Identify pass complete (1432 HF rows, post-processed from mixed-suite original).
- [ ] Explain pass — running, ~73/1474 last check 2026-05-10 11:00 ET.
- [ ] Step 2a applied (once tool ships from WS1).
- [ ] Step 2b applied (once tool ships from WS1).
- [ ] Step 2c applied to identified candidates (once close-mode ships from WS1).
- [ ] Step 2d brief composed + surfaced for Peng approval.

---

## Backlog (not yet started, no design even)

- Workflow retrospective doc (`sweep/WEEKLY_RETROSPECTIVE.md`) — log mistakes per cycle, encode learning.
- Cluster-plan dashboard — `subagents/file-issue/cluster-plans/` view of pending Peng approvals.
- Per-week sweep-compare automation — `sweep_compare.py` wired into nightly pipeline (currently manual).

---

## Closed

- 2026-05-10: 3 dynamo issues closed via one-off script (#21, #26, #27) → reverted + reopened; awaiting close-mode workflow.
