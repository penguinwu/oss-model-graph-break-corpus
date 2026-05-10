# Weekly Sweep Workflow

The standard repeated weekly process for the corpus project. Source of truth for what happens, in what order, with what tools and what manual gates.

**Goal:** the workflow runs autonomously from Saturday-night sweep launch to Sunday brief post. Manual approvals required only at predefined gates (cluster plan, brief approval, external posts).

---

## The steps

```
Step 1  — Run sweep + collect results
Step 2a — New errors triage
Step 2b — New models triage
Step 2c — Walk all dynamo issues (close / edit / open)
Step 2d — Compose brief report
Step 3  — Learning + encoding
```

PLAN.md tracks which tools are built; this doc describes what each step DOES.

---

## Step 1 — Run sweep + collect results

**Trigger:** cron `nightly-sweep` fires Saturday 17:00 PT = 20:00 ET.

**What runs today** (per current `tools/run_experiment.py nightly`):
1. `refresh-nightly` (skipped on `--resume`) — pip-update venv to latest torch nightly.
2. Preflight + canary — verify venv + GPU + corpus health on 1 model.
3. Identify pass — every model in cohort × 2 modes (eval/train), `fullgraph=True`. Each row classified as `full_graph` / `graph_break` / `eager_error` / `create_error` / `worker_error` / `timeout`.
4. Auto-retry-timeout pass — re-run timed-out models. **Today uses single `--timeout 180` for all models; the per-tier extension (3× / 9× from `large_models.json`) is NOT YET wired into this code path. Target state: extended per-tier timeout. See `sweep/TIMEOUT_PROPAGATION_DESIGN.md` for the implementation plan.**
5. Auto-retry-errors pass — re-run only RETRY_ELIGIBLE failures (OOM, subprocess crashes, CUDA device-side assert) serially with 1 worker. Whitelist policy in `sweep/run_sweep.py::_is_retry_eligible`. Other eager/create errors are NOT retried (deterministic; fix root cause).
6. Explain pass — for every `graph_break` model, capture detailed break-reason text via `TORCH_LOGS=graph_breaks` re-run.

**Convention for documenting target-vs-current state:** wherever the doc says "Today X; Target Y; see Z" it means X is what the implementation actually does AND a future agent must NOT assume Y is in effect. Y is what the WS1 task in PLAN.md will deliver.

**Watchdog:** cron `sweep-watchdog-<DATE>` runs every 15 min. Reports ALIVE / STALLED / DEAD / COMPLETE. On DEAD: writes `.resume_in_flight` marker, auto-resumes via `tools/run_experiment.py nightly --force --resume`. Death-spiral guard: 3 consecutive failed resumes → escalate to human reviewer, no further auto-resume. Self-disables when `explain_results.json` is present.

**Output files** (in `sweep_results/nightly/<date>/`):
- `identify_results.json` — final per-model status.
- `identify_streaming.jsonl` — append-only as workers complete.
- `explain_results.json` / `explain_checkpoint.jsonl` — break-reason details.
- `auto_retry_timeout_checkpoint.jsonl`, `auto_retry_errors_checkpoint.jsonl`.
- `retry_classification.jsonl` — what was retried vs skipped (whitelist decisions).
- `sweep_state.json` — orchestrator status snapshot.

**Manual intervention required:** NONE in steady state. Watchdog handles death/stall; auto-retry handles transients. manual review only required on death-spiral escalation (rare).

**Gate to Step 2:** cron prompt's wake-cron at sweep-launch +4h triggers post-sweep step (close-stale dry-run today; eventually full Step 2 sequence).

---

## Step 2a — New errors triage

**Trigger:** Step 1 explain pass complete (`explain_results.json` present).

**What it does:** for each `eager_error` / `create_error` row in current sweep that is NOT in `known_errors.json`, NOT in `skip_models.json`, AND NOT present at the same status in baseline sweep — classify and propose triage decision.

**Tool:** `tools/audit_new_errors.py` (NOT YET BUILT — WS1 task).

**Triage classes** (heuristic-driven, with manual fallback):
- `fixture-bug` → action: fix `sweep/worker.py` input synthesis. Examples: "Audio must be mono" (HiggsAudio), "Image features and image tokens do not match" (Mistral3 FCG variant). Bias toward fix-infra over escape-hatch.
- `gpu-contention` (CUDA OOM in multi-worker) → action: wait for auto-retry serial pass; if it still OOMs, classify as `large` or `very_large` tier in `large_models.json`.
- `cuda-context-pollution` (device-side assert) → action: wait for auto-retry; if still fails serially, file as upstream bug.
- `subprocess-crash` (worker died before completing) → action: wait for auto-retry.
- `upstream-bug` (model code bug we can't fix) → action: file new issue via `subagents/file-issue/` workflow.
- `unknown` → action: manual triage (surface to human reviewer if persistent).

**Principle:** create_errors are entirely infra's fault and MUST be fixed (not added to known_errors as escape hatch). Bias toward fix-infra over add-to-list.

**Output:** markdown report + JSON sidecar with per-candidate triage class + suggested action.

**Re-run scope rule (load-bearing — addresses adversary concern #1):** if any `fixture-bug` candidate gets a `worker.py` fix committed during Step 2a, the affected models' identify_results.json rows are STALE. Step 2c MUST NOT proceed on stale rows. The rule:
- After any fixture commit, re-run JUST the affected models via `tools/run_experiment.py sweep --models '<list>' --output-dir <current-sweep-dir> --resume`.
- The resume-into-existing-sweep-dir merges the fresh rows (orchestrator deduplicates by `(name, mode)` — newest write wins).
- Step 2c reads identify_results.json AFTER all in-cycle fixture fixes have been re-run.

**Manual intervention:** for each `unknown` candidate, human reviewer reviews. For `upstream-bug` candidates, human reviewer approves issue filing per Step 2c gates.

---

## Step 2b — New models triage

**Trigger:** Step 1 explain pass complete (parallel with Step 2a).

**What it does:** for each model in current cohort that was NOT in baseline cohort (NEW model, e.g., from transformers library upgrade), perform tier classification + error triage.

**Tool:** `tools/audit_new_models.py` (NOT YET BUILT — WS1 task).

**Per-NEW-model checks:**
1. Status in identify pass: `full_graph` / `graph_break` / errors → continue with normal tracking.
2. If `eager_error` or `create_error`: needs fixture fix (infra's fault; must be fixed before model is "ready").
3. Wall-clock + GPU memory measurement → propose tier:
   - `regular` (default): no entry needed.
   - `large` (> 60s wall OR > 5GB GPU mem): add to `large_models.json` with `timeout_tier: "large"`.
   - `very_large` (> 300s wall OR > 15GB GPU mem): add to `large_models.json` with `timeout_tier: "very_large"`.
4. If unfixable in time for sweep: add to `skip_models.json` with reason (TEMPORARY — must have a follow-up fix task).

**Per-REMOVED-model checks:** confirm in `skip_models.json` (intentional). If not, investigate as transformers refactor (model renamed or deprecated).

**Output:** markdown report listing per-model decisions + proposed config-file edits.

**Manual intervention:** human reviewer reviews and approves config-file edits before commit. New-model fixture fixes are normal `worker.py` commits (no special gate beyond standard adversary-review for `sweep/` changes).

---

## Step 2c — Walk all dynamo issues (close / edit / open)

**Trigger:** Steps 2a + 2b complete (because issue-walk needs to know which models are stable / new / errored).

**What it does:** for each open dynamo issue + every error pattern surfaced this week, decide one of CLOSE / EDIT / OPEN-NEW / NO-ACTION, then route all action decisions through the file-issue subagent.

**Source of truth for the workflow:** `subagents/file-issue/SKILL.md`. That doc owns the gating mechanism (Mode A adversary review, Mode B assembler, cluster plan + human approval, `--via-skill` argparse enforcement, case-file audit chain). This Step does NOT repeat that workflow — when the file-issue skill changes, those changes apply here.

**What this Step adds on top of file-issue:**

**Per-open-issue decision tree (Step 2c is the routing layer; file-issue is the action layer):**
1. From Step 1 identify, look up affected models' current statuses:
   - All flipped to `full_graph`: invoke file-issue **close-mode** (NOT YET BUILT — WS1 task).
   - Some flipped, some still failing: invoke file-issue **edit-mode** (body update reflects new statuses).
   - None flipped: NO-ACTION.

**Close-mode attribution tests (the routing layer's check before file-issue close-mode runs):**
A sweep-data flip alone is insufficient evidence for a close. The router runs three tests; close-mode receives the verdict + framing:
- **Test A:** re-run issue's MRE on CURRENT torch. If MRE still reproduces → NOT a close candidate (the gap is real; sweep flip is misleading).
- **Test B:** re-run MRE on PRIOR torch + CURRENT transformers. If MRE no longer reproduces here either → root cause is transformers code drift. Close framing: `closed-as-not-applicable`.
- **Test C:** re-run MRE on PRIOR torch + PRIOR transformers (original repro env). If still reproduces → torch attribution confirmed; close framing: `closed-as-completed (torch fixed)`. If no longer reproduces → model removed/rewritten; close framing: `closed-as-vacuous`.

Only Test C confirming "still reproduces in original env, no longer reproduces on current torch" justifies attribution to a torch fix. Other outcomes use different framings.

**For NEW patterns** (not tracked by any open issue): invoke file-issue Step 0 (cluster + dedup). All ceremony — cluster plan content, human approval mechanism, case-file schema, posting commands — lives in `subagents/file-issue/SKILL.md`.

**No bypass paths — enforcement:**
- All close/edit/open operations MUST go through `tools/file_issues.py corpus-issue --via-skill <case_id>`. The `--via-skill` flag is `argparse required=True` (mechanically enforced; tested by `tools/test_file_issues.py`).
- Direct GitHub API calls (curl, urllib, gh CLI) for issue close/edit/open from this repo are FORBIDDEN.
- Today the `tools/file_issues.py close-stale` subcommand has no `--via-skill` gate (open exception). Until close-mode ships, `close-stale --apply` is invoked ONLY by the cron prompt's wake step — never by an agent ad-hoc.

---

## Step 2d — Compose brief report

**Trigger:** Step 2c complete (so the brief reflects the post-action state).

**What it does:** compose the brief comparing current sweep to baseline, then post to the team group after manual approval.

**Source of truth for brief content + composition rules:** `~/projects/oss-model-graph-break-corpus/skills/weekly-sweep-brief/`. That dir owns:
- `template.md` — section structure
- `methodology.md` — R1-R8 hard rules + S1-S4 soft rules + 12-item self-check
- `SKILL.md` — 7-step composition workflow

This Step does NOT repeat what's in those files — when the skill changes, those changes apply here.

**What this Step adds on top of the brief skill:**
- **Tool (NOT YET BUILT — WS1 task):** `tools/compose_brief.py` pre-fills sections from `tools/sweep_compare.py` output and runs the methodology self-check mechanically.
- **Bridge (NOT YET WIRED — WS1 task):** `tools/sweep_compare.py` is auto-invoked at end of Step 1 nightly pipeline so its output is ready when Step 2d runs.
- **Manual gate:** human reviewer approves the draft brief before post. Posting routes through `~/.myclaw/spaces/AAQANraxXE4/tools/post_to_feedback.py` (gated wrapper).

---

## Step 3 — Learning + encoding

**Trigger:** Steps 2a-2d complete (or Step 2 stalled at a gate — also a learning trigger).

**What it does:** reflect on what gaps surfaced during this sweep cycle, propose specific actions, and either address them in-cycle OR add tasks to PLAN.md.

**Per cycle, capture:**
1. **Gaps surfaced** — anything where the workflow needed manual intervention beyond the predefined gates, anything where a tool was missing and we improvised, anything where a step's spec was ambiguous and we had to decide.
2. **Proposed action per gap** — either "encode now into <tool/spec>" or "add task to PLAN.md WS1/WS2".
3. **Disposition per gap** — either fixed in-cycle (with commit ref) or queued (with PLAN.md task ref).

**Output destination:** included in the per-sweep summary surfaced to human reviewer (the same surface as the brief approval gate). NOT included in the dynamo-team brief or user-group post — internal to the corpus project workflow improvement loop.

**Principle:** learnings → specific actions, not journal entries. If a gap can be fixed before the brief posts, fix it. If it needs design or larger scope, queue it as a PLAN.md task. Never "note for later" — that's pacifier.

---

## Manual gates summary (where action is required)

| Step | Gate | Frequency |
|---|---|---|
| Step 1 | Death-spiral escalation | Rare (only when watchdog has tried 3 resumes) |
| Step 2a | Triage `unknown` candidates | Per sweep, ~0-3 candidates |
| Step 2a | Approve `upstream-bug` filings (per Step 2c gate) | Per sweep, ~0-2 |
| Step 2b | Approve config-file edits for new tier classifications | Per sweep, when transformers ships new models |
| Step 2c | Approve cluster plan | Per sweep, 1 per filing batch |
| Step 2c | Approve Mode B body (NEW + EDIT) | Per posted issue |
| Step 2c | Approve close-mode decision | Per closed issue |
| Step 2d | Approve brief before post | Per sweep, 1 |

**Goal:** these gates are the ONLY manual touchpoints in steady state. Everything else automated.

---

## Gaps in current state (each is a WS1 task in PLAN.md)

The workflow above describes what SHOULD happen. Today, several pieces are missing:

| Gap | Where in workflow | WS1 task in PLAN.md |
|---|---|---|
| Per-model timeout not enforced in nightly | Step 1 #4 | "Per-model timeout propagation" |
| `audit_new_errors.py` doesn't exist | Step 2a | "Build `tools/audit_new_errors.py`" |
| `audit_new_models.py` doesn't exist | Step 2b | "Build `tools/audit_new_models.py`" |
| close-stale lacks `--via-skill` gate; close-mode doesn't exist | Step 2c | "Build file-issue close-mode workflow" |
| Brief is hand-composed | Step 2d | "Build `tools/compose_brief.py`" |
| `sweep_compare.py` invoked manually | Step 1 → Step 2 bridge | "Wire `tools/sweep_compare.py` into nightly pipeline" |

---

## Discoverability (addresses adversary concern #5)

This doc is useless if no one reads it. Wired into:
- **Project `CLAUDE.md`** (top): "When about to perform any post-sweep step (2a/2b/2c/2d), READ `sweep/WEEKLY_SWEEP_WORKFLOW.md` first. It is the canonical spec for the weekly sweep workflow."
- **Space `CLAUDE.md`** (`~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md`): session-start checklist references this doc as the workflow source-of-truth alongside PLAN.md.
- **PLAN.md**: WS1 task "Maintain `sweep/WEEKLY_SWEEP_WORKFLOW.md`" tracks updates; TASK 0 in WS2 cites this doc as the spec to walk.
- **`nightly-sweep` cron prompt**: the +4h post-sweep wake's prompt begins with "read `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2 sequence before doing anything else."

If you're an agent reading this and you didn't get here through one of those paths, one of the entry points is broken — fix it before proceeding.

## Known-errors rot audit (quarterly — addresses adversary concern #2)

Every ~3 months (or when `known_errors.json` exceeds 25 entries, whichever first), audit existing entries:
- For each entry with `applies_to_versions` covering a version older than 6 months: re-run on current torch + current transformers. If the bug clears, remove the entry.
- For each entry whose original reason cites "fixture" or "input" or vague "investigate later": flag for re-triage as fixture-bug candidate.
- Goal: prevent corpus rot where escape-hatch entries accumulate forever.

This is NOT a per-sweep step. Surfaced separately when the audit cadence triggers.

## Update cadence for THIS doc

This doc describes the standard process. Update when:
- A new step or gate is added to the workflow.
- A tool replaces a manual intervention (e.g., when `audit_new_errors.py` ships, Step 2a's "(NOT YET BUILT)" goes away + the Manual Gates row updates).
- the workflow definition changes definition.

NOT updated for: per-week findings (those go in the weekly brief), per-issue decisions (those live in issue comments / file-issue case files), implementation details (those live in tool docstrings).
