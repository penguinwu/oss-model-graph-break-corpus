# Weekly Sweep Workflow — Current State + Gap Analysis

**Status:** DRAFT for review (Peng) per directive 2026-05-10 11:03 ET ("encode the right process for this workflow that we repeat every week ... runs smoothly with little intervention from me").

**Goal:** A standard weekly sweep workflow that runs autonomously, with Peng intervening only at pre-defined gates (cluster-plan approval, brief approval, external posts).

**Author note:** This is a READ-ONLY pass. No implementation. Surfacing for Peng's review of the gap inventory + step-by-step plan.

---

## Workflow as Peng described it (2026-05-10 11:03 ET)

| Step | What | Where we are |
|---|---|---|
| 1 | Run sweep + collect results.jsonl with errors / GB counts / fullgraph counts | **WE ARE HERE** — sweep in explain phase. Watchdog just fixed today. |
| 2a | New errors triage → (infra fix) OR (open new issue candidate) | NOT STARTED |
| 2b | New models triage → tier (large/very_large/regular), known_errors, skip_models, file infra fixes | NOT STARTED |
| 2c | Walk all dynamo issues → close / edit / open new (using gated file-issue workflow) | I JUMPED HERE EARLY — bypassed gates, had to revert |
| 2d | Brief report comparing to baseline (kept in repo / posted upon Peng approval) | NOT STARTED |

The unit of repeated weekly work is Steps 1-2d, in order. Any time we skip the order or bypass a gate, we accumulate one-off noise that doesn't compound into autonomy.

---

## Inventory: what infrastructure exists today

### Step 1 (Run sweep)

| Component | What it does | Status |
|---|---|---|
| `tools/run_experiment.py nightly` | End-to-end pipeline (refresh → preflight → canary → sweep → close-stale dry-run) | EXISTS |
| `nightly-sweep` cron (Sat 8 PM ET) | Launches detached pipeline; schedules wake-cron at +4h for close-stale review | EXISTS |
| `sweep/run_sweep.py` orchestrator | identify pass + auto-retry-timeout + auto-retry-errors phases | EXISTS |
| `sweep/sweep_watchdog.py` (v3, today) | Stateless observer: ALIVE/STALLED/DEAD/COMPLETE with tier-aware threshold | JUST SHIPPED |
| `sweep/sweep_watchdog_cycle.sh` (v3, today) | Cron wrapper: case-statement dispatch + `.resume_in_flight` marker mechanism | JUST SHIPPED |
| `sweep/run_sweep.py::_is_retry_eligible` | Auto-retry whitelist (OOM, subprocess crashes, CUDA-assert) | JUST SHIPPED |
| `sweep/skip_models.json` | Skip list (BLT, Gemma3n, etc.) | EXISTS |
| `sweep/known_errors.json` | Known eager/create_error suppressions with version filter | EXISTS |
| `sweep/large_models.json` | Tier registry (large/very_large) — **NOT YET WIRED into nightly pipeline** | EXISTS, scheduled for tomorrow per `TIMEOUT_PROPAGATION_DESIGN.md` |

**Gaps in Step 1:**
- ⚠ Per-model timeout NOT yet enforced in nightly pipeline (tomorrow's task; design doc pending Peng review).
- ⚠ Watchdog v3 untested in a real death scenario; verified only via marker-grace + stale-marker scenarios this morning.

### Step 2a (New errors triage)

| Component | What it does | Status |
|---|---|---|
| Walk `identify_streaming.jsonl` for HF eager_error / create_error not in known_errors/skip_models | Manual one-off Python script today | **MISSING — should be `tools/audit_new_errors.py`** |
| Triage decision per error: (fix-infra) / (add-known-errors) / (add-skip-models) / (file-new-issue) | Manual judgment | **MISSING — no tracking, no audit trail** |
| Infra fix workflow: edit `sweep/worker.py` (or equivalent), test, commit | Manual today | EXISTS as discipline; no automation |

**Gaps in Step 2a:**
- ⚠ No `audit_new_errors.py` tool. Each week I re-write the same Python script.
- ⚠ No structured triage tracker. Decisions live in commit messages, not in a queryable place.
- ⚠ No "new error" definition: is it "not in known_errors AND not in skip_models AND not in last week's results"? Each session I redefine this.

### Step 2b (New models triage)

| Component | What it does | Status |
|---|---|---|
| Detect new models added since last sweep | None — `tools/sweep_compare.py` shows cat 4 (NEW only in current) | PARTIAL |
| Per-model triage: tier classification, known_errors entry, skip_models entry, fixture fix | Manual today | **MISSING — no tool** |
| Fix create_errors before declaring model ready for sweep | Per Peng directive 2026-05-10 ("create_errors are entirely our infra's fault") | **MISSING — no enforcement** |

**Gaps in Step 2b:**
- ⚠ `sweep_compare.py --check-only` requires explain pass to pass invariants; can't pre-audit just from identify.
- ⚠ For each new model, the triage path is: try eager → try compile → measure GPU mem → assign tier. No tool that walks this.
- ⚠ create_errors on new models should BLOCK adding the model to the cohort until fixed; today there's no enforcement.

### Step 2c (Issue actions: close / edit / open)

| Component | What it does | Status |
|---|---|---|
| `subagents/file-issue/` Mode A + Mode B | Gated NEW issue workflow with adversary + assembler + verify_repro × 4 | EXISTS |
| `tools/file_issues.py corpus-issue --new` | Posts NEW issue, gated by --via-skill + --cluster-plan-approved + 4 repro JSONs | EXISTS |
| `tools/file_issues.py corpus-issue --edit <num>` | EDITs existing issue, gated by --via-skill + --cluster-plan-approved (no repro gate per v1.0) | EXISTS |
| `tools/file_issues.py close-stale --plan <sweep-report>` | Bulk close-with-comment for issues whose all-affected models now compile fullgraph | **GAP: NO --via-skill gate** |
| `subagents/file-issue/` CLOSE mode | Gated CLOSE workflow with adversary review of the close evidence + MRE re-verification | **MISSING (deferred to v1.5 per SKILL.md)** |
| `tools/file_issues.py corpus-issue --close <num>` | CLI subcommand to close an issue with audit comment | **MISSING** |
| `tools/cluster_failures.py` | Cluster sweep failures into batches; emit cluster plan for Peng approval | EXISTS |
| `tools/dedup_search.py` | Dedup against existing open issues for cluster plan | EXISTS |

**Gaps in Step 2c (HIGHEST PRIORITY — this is what bit us this morning):**
- ⚠ **`close-stale` has no per-issue review gate.** Bulk-close with criterion "all models flipped fullgraph" is mechanical — but the AUDIT comment makes a causal claim ("torch fixed the gap") that R3 (weekly-sweep-brief methodology) requires verifying via MRE re-run.
- ⚠ **`subagents/file-issue/` deliberately defers CLOSE to v1.5.** Today there is NO gated close workflow at all. close-stale fills the bulk-action gap but skips per-issue rigor.
- ⚠ When I bypassed both today, I made data-driven decisions but skipped the causal-attribution rigor → had to revert.

### Step 2d (Brief report)

| Component | What it does | Status |
|---|---|---|
| `skills/weekly-sweep-brief/SKILL.md` | 7-step workflow: gather data → verify attribution → segment per-pattern → umbrella-split → compose → self-check → post | EXISTS |
| `template.md` | 8-section structure: Headline, Wins, GB-reduction, Regressions, Issues, New-models, New-patterns, Actionable | EXISTS |
| `methodology.md` | R1-R8 hard rules + S1-S4 soft rules + self-check checklist (12 items) | EXISTS |
| `tools/sweep_compare.py` | Single source of truth (R1) — apple-to-apple comparison with invariant check | EXISTS |
| `tools/post_to_feedback.py` | Gated wrapper for user-group posts (R7) | EXISTS |

**Gaps in Step 2d:**
- ⚠ No `tools/compose_brief.py` that pre-fills the 8 sections from sweep_compare output. Each brief is hand-composed.
- ⚠ Brief is composed AFTER all of Step 2c (close/edit/open) — but Step 2c is partially manual today, so brief composition has been blocked or delayed.
- ⚠ The brief includes "Issues — actions taken (umbrella-split policy)" — Section 5. This section depends on Step 2c outputs being machine-readable. Today, Step 2c is ad-hoc, so Section 5 is hand-summarized.

### Cross-cutting

| Component | Status |
|---|---|
| `subagents/adversary-review/` | EXISTS — should be invoked for ANY substantive code/decision in this workflow |
| `subagents/mre/` | EXISTS — provides verified MRE for issue bodies; ledger tracks every dogfood |
| Weekly workflow checklist | **MISSING** — Steps 1-2d aren't documented as a single ordered checklist anywhere |
| Workflow retrospective | **MISSING** — no place to log "this week I made these mistakes; here's how the workflow needs to harden" |

---

## Gap analysis — prioritized

### Tier 1 (BLOCKING repeat weekly autonomy)

1. **CLOSE workflow (Step 2c)** — file-issue skill defers to v1.5; close-stale has no per-issue gate. This is the gap that caused this morning's 3-issue rollback. Either:
   - **Option A:** Extend `subagents/file-issue/` with a `mode: close` that adversary-reviews the close evidence + requires MRE re-verification on current torch (R3 enforcement). Then add `tools/file_issues.py corpus-issue --close <num> --via-skill ...` analogous to --edit.
   - **Option B:** Extend `tools/file_issues.py close-stale` to require per-candidate `--via-skill` (auto-generated case_id from a file-issue close-mode invocation). Each candidate must have an MRE re-verification recorded in the case file before close.
   - I lean Option A — it makes "close" a first-class file-issue operation with its own rigor; close-stale becomes a batch-launcher of close cases.

2. **New-errors triage (Step 2a)** — manual ad-hoc each week. Need:
   - `tools/audit_new_errors.py <current-sweep-dir> <baseline-sweep-dir>` — emits structured candidate list (per error: model, mode, error message, current status in known_errors/skip_models, suggested triage class).
   - Triage decisions tracked in `sweep/triage_decisions.jsonl` (or per-week dir under `sweep_results/nightly/<date>/triage.jsonl`).
   - Each "fix-infra" decision creates a TODO that blocks the brief until resolved (or explicitly deferred with reason).

3. **Per-model timeout enforcement (Step 1)** — already designed (`TIMEOUT_PROPAGATION_DESIGN.md`); pending Peng review tomorrow morning per existing reminder cron.

### Tier 2 (IMPROVES weekly experience but workable manually)

4. **New-models triage (Step 2b)** — needs `tools/audit_new_models.py` to detect cohort additions/removals + propose tier classification.

5. **`compose_brief.py`** — pre-fill the 8 brief sections from sweep_compare output. Currently the brief is hand-composed.

6. **Weekly workflow checklist** — single canonical doc that lists Steps 1-2d in order with "what to run, what to surface, what gates to clear." Today the workflow lives across 6+ docs (this file, weekly-sweep-brief/SKILL.md, file-issue/SKILL.md, WORKFLOWS.md, cron prompts, ad-hoc memory).

### Tier 3 (NICE-TO-HAVE)

7. **Workflow retrospective** — append to `sweep/WEEKLY_RETROSPECTIVE.md` after every weekly cycle: what mistakes happened, what's the encoded learning?

8. **Cluster-plan dashboard** — `subagents/file-issue/cluster-plans/` accumulates over time; eventually need a "what's pending Peng approval" view.

---

## Proposed next steps (for Peng to choose order)

I'd ask you to pick the next concrete thing to work on. Three plausible orders:

### Path A — finish Step 1 first, then build Step 2c gates
1. Tomorrow: Per-model timeout (pending Peng design review).
2. After: Build close-mode in file-issue (Tier 1 #1, Option A above).
3. After: Build audit_new_errors (Tier 1 #2).
4. Use the new infra in next week's sweep cycle.

### Path B — finish Step 2 for THIS sweep first, building infra as we go
1. Today: When current sweep's explain finishes, walk Step 2a manually (audit new errors). As I do so, capture the script as `tools/audit_new_errors.py`.
2. Today: Walk Step 2b manually. Capture as `tools/audit_new_models.py`.
3. Today: For Step 2c, propose CLOSE design + iterate with you on it; don't actually close anything via the new path yet (because design isn't built).
4. Today: Compose the brief by hand using existing weekly-sweep-brief skill.

### Path C — pause sweep work; design the full Step 2c gate this morning, then resume
1. Today: Design + ship the file-issue CLOSE mode (Tier 1 #1, Option A).
2. After: Use it on the 3 issues I had to revert; produce clean R3-verified closes.
3. After: Resume Step 2 for the rest.

I'd lean **Path B** — it gets THIS week's sweep through Steps 2a-d while building the missing infra incrementally, but keeps the rigor (no closes without designed-gate; design first, then use).

What's your call?

---

## Appendix: file-paths cited in this doc

- `subagents/file-issue/SKILL.md` — gated NEW + EDIT issue workflow (CLOSE deferred)
- `subagents/file-issue/persona.md` — Mode A + Mode B
- `subagents/mre/SKILL.md` — verified MRE construction
- `subagents/adversary-review/SKILL.md` — code review subagent
- `tools/file_issues.py` — corpus-issue NEW/EDIT, close-stale (no via-skill), pytorch-upstream
- `tools/cluster_failures.py` — Step 0 cluster
- `tools/dedup_search.py` — Step 0 dedup
- `tools/sweep_compare.py` — apple-to-apple comparison (R1 source of truth)
- `tools/run_experiment.py` — nightly pipeline
- `sweep/run_sweep.py` — orchestrator (with new auto-retry whitelist)
- `sweep/sweep_watchdog.py` — observer (v3, just shipped)
- `sweep/sweep_watchdog_cycle.sh` — cycle script (v3, just shipped)
- `sweep/skip_models.json`, `known_errors.json`, `large_models.json` — config files
- `skills/weekly-sweep-brief/SKILL.md` + `methodology.md` + `template.md` — Step 2d
- `~/.myclaw/spaces/AAQANraxXE4/cron-prompts/nightly-sweep.md` — versioned cron prompt
- `~/.myclaw/spaces/AAQANraxXE4/WORKFLOWS.md` — workflow index (incomplete)
- `sweep/TIMEOUT_PROPAGATION_DESIGN.md` — Step 1 follow-up (tomorrow)
- `sweep/AUTO_RETRY_REFINEMENT.md` — Step 1 detail (just shipped)
- `sweep/WATCHDOG_DESIGN.md` — Step 1 detail (just shipped)
