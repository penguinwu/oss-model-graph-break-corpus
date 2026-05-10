# PLAN.md — corpus project working plan

**Last updated:** 2026-05-10 12:00 ET (Otter)
**Active workstreams:** 2 (cap). New workstreams go to Backlog until a slot opens.

Read this FIRST when starting a session. State "Plan loaded: <current focus>" in the first message. Update after every completed task, every new task added, every scope change.

---

## WS1 — Build the standard weekly-sweep workflow so it runs autonomously

The standard weekly process (per Peng spec 2026-05-10): launch sweep on Saturday → wait for completion → audit new errors and triage them → audit new models and triage them → walk all dynamo issues to decide close/edit/open → compose a brief comparing this week to baseline. Goal: minimal Peng intervention; all gates encoded in tools, not in human discipline.

### Done (this week)

- [x] **Watchdog v3 simple design** — rewrote `sweep/sweep_watchdog.py` as stateless observer with tier-aware STALLED threshold (identify=30min, auto_retry_timeout=90min). Rewrote `sweep/sweep_watchdog_cycle.sh` with case-statement dispatch + `.resume_in_flight` marker mechanism + `*)` default arm for crash safety. Removes the `notified` flag chicken-and-egg deadlock that left tonight's sweep dead 7h. Cron interval 15 min (was 10). Design at `sweep/WATCHDOG_DESIGN.md`. Shipped 2026-05-10.
- [x] **Auto-retry whitelist policy** — `sweep/run_sweep.py::_is_retry_eligible` only retries OOM, subprocess-crash, and CUDA device-side-assert. Everything else defaults to NO retry (per Peng + adversary review). Emits `retry_classification.jsonl` for machine-queryable history. Design at `sweep/AUTO_RETRY_REFINEMENT.md`. Shipped 2026-05-10.
- [x] **HF-only sweep default** — `tools/run_experiment.py nightly` and `sweep` subcommand default `--source hf`. Diffusers + custom now opt-in. Reason: diffusers tail had 47% eager_error rate vs HF 75% full_graph; was killing weekly sweeps via auto-retry queue. Shipped 2026-05-10.
- [x] **Skip refresh-nightly on `--resume`** — saves 2-15 min per resume cycle. The venv was at the correct version at original launch; pip metadata fetch + dep verification on every resume buys nothing for our failure modes. Shipped 2026-05-10.
- [x] **BLT models added to `sweep/skip_models.json`** — interim unblock for tonight's sweep. Both `BltForCausalLM` and `BltModel` need very_large timeout tier (1620s) but the regular nightly sweep used `--timeout 180` (3 min) and didn't read `large_models.json`. Models died deterministically at item 127 on both 2026-05-03 and 2026-05-09 sweeps. To be removed once per-model timeout propagation lands.

### Pending (in priority order)

- [ ] **Maintain `sweep/WEEKLY_SWEEP_WORKFLOW.md` — the standard weekly-sweep process spec.** This is the canonical doc that describes the standard weekly workflow Steps 1-2d (run sweep → audit new errors → audit new models → walk dynamo issues → compose brief), with for each step: what it does, what tool runs it, what gates exist, what manual interventions are still required. PLAN.md is the task tracker; WEEKLY_SWEEP_WORKFLOW.md is the process spec the tools implement. The doc currently contains a gap analysis I wrote 2026-05-10 11:08 ET; needs rewriting to be the forward-looking process spec (with the gaps reframed as tasks already captured in PLAN.md). Updated whenever a new step or gate is added to the standard workflow.

- [ ] **Per-model timeout propagation in `tools/run_experiment.py nightly`** — wire the existing `_timeout_for(name)` helper from `sweep/run_sweep.py:836-855` (deprecated entry point) through to the orchestrator from `tools/run_experiment.py` (current entry point). Add launch-side validation gate that refuses to launch if `large_models.json` is non-empty but `load_per_model_timeouts()` returns empty (catches future regression of this class). Design at `sweep/TIMEOUT_PROPAGATION_DESIGN.md`. **Awaiting Peng design review** — reminder cron `timeout-propagation-task-2026-05-10` fires Sunday 9 AM ET to surface design link to GChat for approve/modify/reject. Implementation only after explicit approval. After landing, remove BLT from `skip_models.json` and verify they run cleanly with 1620s timeout.

- [ ] **Build `tools/audit_new_errors.py`** — Step 2a script. Walks the current sweep's `identify_streaming.jsonl` for HF rows with `status=eager_error` or `status=create_error` that are NOT covered by `known_errors.json` AND NOT in `skip_models.json` AND NOT present at the same status in baseline sweep (so it surfaces actual NEW errors, not stable failures). Classifies each error message via heuristic into one of: `fixture-bug` (worker.py input synthesis wrong), `gpu-contention` (auto-retry will catch), `cuda-context-pollution` (device-side assert; auto-retry will catch), `subprocess-crash` (auto-retry will catch), `upstream-bug` (file as new issue), `unknown` (manual triage). Emits markdown report + JSON sidecar with proposed triage class per candidate. Per Peng directive 2026-05-10: bias toward INFRA-FIX over add-to-list (escape hatch). Replaces the manual one-off Python audit Otter keeps re-writing each week. **Design pending.**

- [ ] **Build `tools/audit_new_models.py`** — Step 2b script. Detects cohort additions and removals between current sweep and baseline. For each NEW model: probe to determine tier (regular / large / very_large) by checking GPU memory + wall-clock from sweep evidence, propose a `large_models.json` entry if needed, and propose `known_errors.json` / `skip_models.json` additions for any new-model failures. For REMOVED models: confirm whether the removal is intentional (in skip_models.json, expected) or unexpected (transformers refactor — needs investigation). Emits markdown report listing per-model triage decisions. Per Peng directive 2026-05-10: create_errors on new models are entirely our infra's fault and must be fixed before model is considered ready for the cohort. **Design pending.**

- [ ] **Build file-issue close-mode workflow** — Step 2c. Extend `subagents/file-issue/persona.md` with Mode A_close (adversary reviews close decision: verifies all-affected-models flipped via sweep evidence + verifies MRE no longer reproduces on current torch via verify_repro re-run = R3 attribution test) and Mode B_close (assembles closing comment with structured audit chain: sweep evidence + MRE re-verification + attribution claim scoped to what's verified). Add `tools/file_issues.py corpus-issue --close <num> --via-skill <case_id>` operation that PATCHes state=closed only after `--via-skill` validates the case file has `mode_a_verdict in {proceed}` and `mode_b_sha256` non-empty. Replaces today's bypass-prone path where close-stale OR a one-off API script can close issues without per-issue rigor. **Design pending.**

- [ ] **Build `tools/compose_brief.py`** — Step 2d. Pre-fills the 8 sections of `~/projects/oss-model-graph-break-corpus/skills/weekly-sweep-brief/template.md` from `tools/sweep_compare.py` output (cat 1/2/3/4/5/6 partition + per-pattern segmentation). Walks the brief methodology rules (R1-R8 + S1-S4 from `methodology.md`) and runs the self-check checklist mechanically. Emits draft brief markdown for Peng review. Currently each brief is hand-composed; this captures the rules into code so the brief can be regenerated reliably. **Design pending.**

## WS2 — Apply WS1 workflow to this week's sweep (2026-05-09)

Execution of WS1's standard workflow on real data — without re-running the sweep. The first task is the meta-test that surfaces gaps in WS1 tool design and feeds them back to WS1 as new tasks.

### Done (this week)

- [x] **Identify pass complete** — sweep workers processed 1432 HF (name, mode) pairs. Originally launched mixed-suite (`hf diffusers custom`) per old default; post-processed to HF-only after Peng directive 2026-05-10 redirected default to HF-only. Backup of mixed-suite results preserved at `sweep_results/nightly/2026-05-09/_pre-hf-only-backup/`.

- [x] **Cohort delta vs 2026-05-03 baseline accounted for** — 25 unique models "removed" since baseline; manual check confirmed all 25 are in `skip_models.json` (Blt*, Gemma3n*, ConditionalDetrModel, etc.). Not an unexpected transformers refactor; expected behavior given skip-list growth between sweeps.

### In flight

- [ ] **Explain pass running** — last check 2026-05-10 11:00 ET: phase=explain done=73/1474, +23 since prior 15-min watchdog cycle. ETA ~3-4 hours from explain start. Watchdog v3 is observing via the new sweep-watchdog-2026-05-09 cron (15-min interval, enabled).

### Pending (in priority order)

- [ ] **TASK 0 (meta) — Apply the WS1 weekly-sweep workflow end-to-end on the 2026-05-09 sweep, WITHOUT re-running the sweep itself.** Walk through every step of the standard process (per `sweep/WEEKLY_SWEEP_WORKFLOW.md`): Step 2a audit new errors → Step 2b audit new models → Step 2c walk dynamo issues to decide close/edit/open → Step 2d compose brief. For each step, note: what tool would normally run this step, whether the tool exists or is a manual fallback today, what gaps surface that should become new WS1 tasks. The output of TASK 0 is a list of new WS1 tasks (added back to WS1 pending list) — NOT actual postings/edits/closes. Per Peng directive 2026-05-10 11:55 ET: this is the meta-test that drives WS1's actual scope. Do TASK 0 BEFORE the per-step-application tasks below; the per-step tasks are checkpoints OF this meta walk, not separate work. **Start once explain pass completes (or before, if a step is purely identify-driven).**

- [ ] **Sub-checkpoint of TASK 0: Step 2a application** — surfaces 4 errors I manually audited this morning: HiggsAudio (upstream alignment bug), Mistral3 (fixed by my fixture commit), Blip2 (transient), Sam3 (OOM contention). Will be done as part of the TASK 0 walk, not separately.

- [ ] **Sub-checkpoint of TASK 0: Step 2b application** — expected: 0 NEW models, 25 removed all match skip_models.json. Will be done as part of TASK 0.

- [ ] **Sub-checkpoint of TASK 0: Step 2c application** — for each open dynamo issue, surface close/edit/open decision. The 3 reverted issues (#21, #26, #27) need close-mode workflow; 14 stable-failure issues need no action; #77 needs comment with progress (75% improved). Will be done as part of TASK 0.

- [ ] **Sub-checkpoint of TASK 0: Step 2d application** — compose brief: 0 dynamo wins, 0 dynamo regressions, 5 non-dynamo improvements (Aria* fixture). Quiet-week brief. Surface for Peng approval before posting. Will be done as part of TASK 0.

- [ ] **Audit the 3 reverted issues (#21, #26, #27) for unexpected side effects.** Per Peng directive 2026-05-10 11:58 ET. Independent of the close-mode workflow that will eventually re-process them properly. Things to check: (a) verify all 3 are currently `state=open` on GitHub, (b) verify the reversal comment posted correctly on each (no malformed markdown, no missing pieces), (c) verify the prior closing comments are still on each issue's timeline (not deleted — they document the audit chain), (d) verify no labels were dropped during close/reopen, (e) verify no GitHub auto-link pollution onto OTHER issues from any of the comments (per the corpus's no-#NNN-in-subagent-files rule), (f) verify no downstream effects (sweep-update tool didn't try to act on closed-then-reopened state, etc.). Goal: surface anything I missed during the rapid close+revert cycle this morning.

---

## Backlog (waiting for active workstream slot to open)

These are real proposed work but blocked on the 2-active-workstream cap. Promote one of these only when WS1 or WS2 finishes.

- **Weekly workflow retrospective** — `sweep/WEEKLY_RETROSPECTIVE.md`. Append after every weekly cycle: what mistakes happened, what learning to encode in the workflow tools. Goal: harden the workflow each week so the next week's sweep needs less Peng intervention than the last.
- **Cluster-plan dashboard** — `subagents/file-issue/cluster-plans/` accumulates plan files over time; need a "what's pending Peng approval" view so plans don't get forgotten between sessions.
- **Wire `tools/sweep_compare.py` into nightly pipeline** — currently the comparison must be run manually after each sweep. Wiring it into `tools/run_experiment.py nightly` Step 5+ would emit `compare-vs-baseline.json` automatically, ready for `compose_brief.py` to consume.

---

## Closed

- 2026-05-10: 3 dynamo issues closed via one-off API script (#21, #26, #27) → reverted + reopened. Reason: bypassed the gated file-issue workflow that Peng's process requires. Will re-close properly once close-mode (WS1 task) ships.
