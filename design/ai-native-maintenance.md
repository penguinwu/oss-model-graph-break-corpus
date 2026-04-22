# OSS Model Compiler Quality Corpus — AI-Native Maintenance Design

**Owner:** Otter (sole maintainer as of 2026-04-22)
**Original author:** Rocky (Rev 1, 2026-04-02 — design phase)
**Revision:** 2
**Date:** 2026-04-22
**Parent project:** [Design Doc](https://docs.google.com/document/d/1paCL1R8xoN6OajND8c4M5WgA68Uw1iEij-katYFqneM)

## Revision History

- **Rev 1** (Rocky, 2026-04-02) — Original design with separated agent roles (Otter proposes, Rocky validates, Peng approves), GChat-only feedback intake, Meta Tasks for tracking.
- **Rev 2** (Otter, 2026-04-22) — Single-owner refactor (Rocky off project), GitHub Issues as source of truth, web proxy unblocks GitHub access, fifth feedback classification for consumer integration requests, consumer-SLA section.

## 1. Problem

The corpus is a public OSS dataset of 734 open-source models tested for `torch.compile` quality (graph breaks, correctness, errors). It has tooling for sweeping, querying, reproducing, comparing, and issue management. As consumers adopt it, two pressures arise:

- **Version drift.** New PyTorch and transformers releases change which models break. The corpus must stay current or it becomes stale.
- **Consumer feedback.** Internal users find bugs, request features, report data corrections. As of 2026-04-22 we also have **downstream consumers** (e.g., Arsh's `debug-graph-breaks` skill) that depend on the `corpus.json` schema — feedback from them is contractual, not casual.

Goal: a maintenance system where Otter handles most upkeep — triage, fixes, validation, corpus updates, consumer-contract management — with Peng reviewing before anything ships externally.

## 2. Design Principles

- **Tiered autonomy by reversibility.** Agent actions classified into three tiers based on reversibility. Replaces a blanket "conservative" label with specific rules per action type.
- **Validation-first.** Every change runs through automated validation (`tools/validate.py`, smoke test, golden set) before shipping.
- **Internal-first, GitHub-mirror.** Feedback flows in via GChat; canonical tracking lives in GitHub Issues (web proxy unblocks GitHub from devvm).
- **Single owner.** Otter is sole code modifier and validator. Peng reviews tier-3. No concurrent agent editors.
- **Structured intake.** Templates upfront so agents triage without back-and-forth.

### 2.1 Autonomy Tiers

- **Tier 1 — Fully autonomous (no approval).** Answering user questions in the feedback space. Classifying and triaging feedback. Reading code, running analysis, reproducing bugs. Logging and monitoring. Routine sweep ops.
- **Tier 2 — Auto-apply, notify after.** Regenerating summary blocks. Fixing metadata normalization. Formatting fixes in tool output. Updating CLAUDE.md recipes. Filing well-classified GitHub issues with established categories.
- **Tier 3 — Peng reviews design and logic.** All corpus data changes (status, error fields, model additions). Schema changes. Code changes to tools. Golden set updates. Anything pushed to GitHub for the first time in a category. Cross-team external outreach.

Peng's tier-3 review is at the design and reasoning level — she validates that the right problem is being solved the right way. Otter owns code correctness. When surfacing tier-3 items, Otter must provide enough conceptual context for Peng to catch logical errors without reading code.

Each tier can be upgraded independently as trust accrues.

### 2.2 Single-Owner Operation

Rev 1 specified separated agent roles (Otter proposes, Rocky validates, Peng approves). Rev 2 collapses to single-owner: Otter both proposes and validates, with Peng as the human-in-the-loop for tier-3.

Why the change: low feedback volume + Rocky off the project. Rocky's validator role added overhead without proportional safety gain. The `tools/validate.py` and golden-set checks provide the structural safety net that the human validator was meant to provide.

Risk mitigation in single-owner mode:
- Otter must run `tools/validate.py` and `tools/smoke_test.py` after every code change before reporting "done."
- Golden set regression must pass before any corpus update.
- Tier-3 surfaces to Peng with explicit invariant claims ("this should not change row counts; it changes 3 rows").

### 2.3 Assumptions (Updated 2026-04-22)

Status: V = validated, ? = needs verification, X = invalidated.

**GChat capabilities** — all V (no change from Rev 1).

**GitHub access** — was X in Rev 1, now V. Web proxy at `localhost:7824` (under pengwu's identity) bypasses BPF jailer for HTTP/HTTPS. POST/PATCH/DELETE supported. Otter has direct GitHub Issues access via `tools/file_issues.py`.

**Repo and tooling** — all V. Plus new since Rev 1:
- [V] PR status verification works via `tools/check_pr_status.py` (handles ghstack/Phabricator).
- [V] Issue reconcile tool (`tools/file_issues.py reconcile`) — full lifecycle management.
- [V] Otter has direct git push permission to corpus repo (sudo + proxy + token, scope-limited per spaces/CLAUDE.md).

**Agent coordination** — Rev 1's separated-roles assumptions are obsolete. Rocky-can-read-Otter-files no longer a constraint. Single-owner mode eliminates inter-agent coordination from the maintenance loop.

**Version management** — all V (no change).

## 3. Architecture Overview

Four components:

1. **Feedback Space** — GChat group space (`spaces/AAQABmB_3Is`) where users post feedback with structured intake.
2. **Agent Monitor** — `tools/feedback_monitor.py`, run via `feedback-monitor` cron (5x/day weekdays).
3. **Validation Layer** — `tools/validate.py` + `tools/smoke_test.py` + golden set regression.
4. **Self-Healing Loop** — versioned corpus updates when new PyTorch/transformers releases land.

## 4. Component 1: Feedback Space

### 4.1 What It Is

GChat group space "Graph Break Corpus Feedback" (`spaces/AAQABmB_3Is`). Members: Peng, Otter (agent), and internal consumers (William, Animesh, Arsh, others ad hoc).

### 4.2 GChat for Intake, GitHub for Tracking

- **GChat** is natural for internal users — low friction.
- **GitHub Issues** are the source of truth for tracked work. Web proxy unblocks GitHub from devvm.
- Otter responds in GChat AND files GitHub Issues with `<!-- filed-by: otter/file_issues.py -->` markers.
- Tasks integration (Rev 1's plan) was removed — single tracking system, no double-bookkeeping.

### 4.3 Structured Feedback Intake

Reporting template (pinned in space):

> "When reporting an issue, please include:
> - **What you were doing:** which tool, what command
> - **What happened:** error message, unexpected output
> - **What you expected:** what should have happened
> - **Model name(s):** if model-specific
> - **Versions:** `python -c "import torch; print(torch.__version__)"`
> - **Category:** Bug / Data correction / Feature request / Question / Skill integration"

Agent-assisted intake: when a user posts without the template, Otter (1) acknowledges immediately, (2) asks targeted follow-up for missing critical fields, (3) attempts reproduction with whatever was provided, (4) fills in determinable info (current versions, corpus lookups).

### 4.4 Message Classification (5 categories — new in Rev 2)

| Category | What it means | Action |
|---|---|---|
| **Bug** | Tool crash, wrong output, data inconsistency | Reproduce → fix → file GitHub Issue → respond in thread |
| **Feature request** | New CLI flag, query mode, output format | Discuss in thread → if approved, implement → file Issue |
| **Data correction** | Specific model's result is wrong | Re-run model to verify → if confirmed, fix corpus + file Issue |
| **Question** | How to use tools, interpret data | Answer directly in thread; file doc-gap Issue only if pattern emerges |
| **Skill integration request** *(new)* | Downstream consumer (skill maintainer) reports a need: schema instability, missing field, broken reproducer | Treat as **consumer-driven contract change** — see §6 |

The fifth category, added in Rev 2, exists because skill consumers (e.g., Arsh's `debug-graph-breaks`) depend on `corpus.json` schema fields. Their feedback affects contract stability, not just feature surface.

## 5. Component 2: Agent Monitor

### 5.1 Cron

`feedback-monitor` MyClaw cron job (5x/day weekdays) runs in Otter's daemon, calling `tools/feedback_monitor.py`.

### 5.2 Triage Flow

For each new message:
1. Classify into one of the 5 categories.
2. Check for duplicates against open GitHub Issues with the same category label.
3. If duplicate: respond in-thread with link to existing Issue.
4. If new: create GitHub Issue with appropriate `for:*` label, link to GChat message in body.
5. Respond in-thread: acknowledge, share Issue number, explain next steps.

### 5.3 Audit Logging

Every triage decision logged: timestamp, message ID, classification, action taken, reasoning. MyClaw's `raw_response` field on `job_runs` captures full agent output for review.

### 5.4 Working an Item

For bugs and data corrections:
1. Reproduce (run the tool or model).
2. Draft a fix in the local repo.
3. Run `tools/validate.py` + `tools/smoke_test.py`.
4. Report in the GChat thread: what was found, what changed, validation results.
5. If tier-2: commit + push directly. If tier-3: surface to Peng for review first.
6. Close the Issue and respond in-thread: "Fixed in commit XYZ".

For feature requests: draft plan → discuss in thread → after approval, implement → same review and push flow.

## 6. Consumer Contracts and SLAs (new in Rev 2)

Downstream consumers (skill maintainers, benchmarking tools) depend on the corpus's API surface. Maintenance for them is qualitatively different from end-user bug reports.

### 6.1 What Counts as Consumer Surface

- **`corpus.json` schema:** field names, types, semantic meanings of `root_cause`, `break_reasons`, `graph_break_count`, `status` enum values.
- **Reproducer command interface:** `python tools/reproduce.py <ModelName>` invocation form and output format.
- **Pattern classification taxonomy:** the categories `tools/analyze_explain.py` produces.
- **Sweep results JSON shape:** per-model fields the rest of the world reads.

These are documented in USE_CASES.md under "Stable signals consumers can rely on." Changes to any of these need a deprecation path.

### 6.2 SLA Stance (current)

We don't make formal SLA promises. Practically:
- **Field renames or removals:** require advance notice to known consumers + a 1-version overlap (deprecated field stays present alongside new one for one release).
- **New fields:** additive, no notice needed.
- **Semantic changes** (same field, different meaning): treat as breaking — notice + overlap.
- **Pattern taxonomy changes:** notify in feedback space when categories merge/split.

### 6.3 Change Management Process for Skill Integration Requests

When a "Skill integration request" lands (per §4.4):
1. Otter acknowledges in thread, files a tracking Issue with `for:consumer-contract` label.
2. Otter assesses: is this a contract gap (missing field they need), a contract violation (we broke something they relied on), or a coordination request (their roadmap intersects ours)?
3. Tier-3 surface to Peng with the assessment + proposed response.
4. If we commit to a contract change, document it in USE_CASES.md "Stable signals" and notify all known consumers.

Known consumers as of 2026-04-22: Arsh (`debug-graph-breaks` skill, D99943226).

## 7. Component 3: Validation Layer

### 7.1 validate.py

Standalone integrity checker. Every change must pass before shipping.

### 7.2 Golden Set (Semantic Regression Tests)

~20 models with manually verified expected results. Coverage: ≥3 full_graph, ≥5 graph_break, ≥2 create_error, ≥2 eager_error, ≥1 timeout, multiple sources.

Stored as `golden_set.json` alongside `corpus.json`. `validate.py` checks every golden set model against the corpus and fails on unacknowledged changes.

If a version upgrade legitimately changes a golden set model's behavior, Peng updates `golden_set.json` (tier-3 — never the agent unilaterally).

### 7.3 Structural Checks

Corpus integrity: summary block matches actual model counts; `has_graph_break` flag consistent with statuses; no duplicates; required fields present; metadata versions valid; total counts match.

Tool output: `query.py` returns valid JSON; `compare.py --corpus-dynamic` runs without error; `tools/smoke_test.py` 3-model run completes.

Schema validation: error field consistency; dynamic results present where expected.

### 7.4 Usage

```bash
python3 tools/validate.py            # check
python3 tools/validate.py --fix      # auto-fix what can be fixed
python3 tools/smoke_test.py          # 3-model infra smoke
python3 tools/run_experiment.py selftest  # equivalent
```

## 8. Component 4: Self-Healing Loop

### 8.1 Triggers

- **New PyTorch release** — primary trigger.
- **New HF transformers release** — secondary trigger.
- **Consumer-reported staleness** — feedback in GChat that results look out of date.

### 8.2 Versioned Storage

Sweep results live in `sweep_results/<label>/` (e.g., `sweep_results/pt2.11/`, `sweep_results/nightly/2026-04-19/`). Per-sweep `sweep_metadata.json` captures full environment.

`corpus.json` reflects the latest sweep. Historical results preserved for diff. The corpus is **not** a multi-version database — it answers "what breaks now," with history available via comparison.

### 8.3 Version Upgrade Flow (single-owner)

When a new PyTorch or transformers version lands:
1. Otter runs the sweep with the new version → results in `sweep_results/<version-label>/`.
2. Otter runs `tools/compare.py` against the prior baseline → categorized changelog.
3. Otter runs `tools/file_issues.py sweep-report` → reviewable issue plan.
4. Otter runs `tools/validate.py` + golden set check on the proposed corpus update.
5. Otter surfaces tier-3 to Peng: changelog + invariant claims (e.g., "X regressions, Y fixes; eager-failure count should be unchanged at N — verified") + proposed actions.
6. Peng approves; Otter commits + pushes.
7. If golden set models changed, Peng updates `golden_set.json`.

### 8.4 What Exists

- `tools/update_corpus.py` — merges sweep results into `corpus.json`.
- `tools/file_issues.py` — sweep-report, sweep-update, reconcile.
- `tools/check_pr_status.py` — verify PR landed status.
- `sweep/run_sweep.py --check-env` — version logging at sweep start.

## 9. CLAUDE.md Expansion

Repo CLAUDE.md already includes:
- Script Map (sweep + tools layout)
- Common Workflows (sweep, post-sweep, experiments, reproduction)
- Conventions (batch size, backend, default sources, artifact location)
- Validation invariants (in Closure Discipline + Validate Invariants sections)

Continue expanding agent recipes for: adding CLI flags, fixing data inconsistencies, updating corpus from sweep results, adding model-specific fixes (already documented in Model-Specific Fixes section).

## 10. Implementation Status (as of 2026-04-22)

### Phase 1: Foundation — COMPLETE
- `tools/validate.py` ✓
- CLAUDE.md recipes ✓ (ongoing expansion)
- Version logging in sweep ✓ (`--check-env`)
- Versioned `sweep_results/` ✓

### Phase 2: Feedback Loop — COMPLETE (refactored from Rev 1)
- GChat feedback space ✓
- `feedback_monitor.py` cron ✓ (5x/day weekdays)
- ~~Meta Tasks integration~~ — REMOVED. GitHub Issues are source of truth.
- Agent-assisted intake ✓

### Phase 3: Self-Healing — COMPLETE
- `tools/update_corpus.py` ✓
- `tools/file_issues.py` (sweep-report, sweep-update, reconcile) ✓
- `tools/check_pr_status.py` ✓
- End-to-end flow ✓ (single-owner: Otter sweeps + validates, Peng approves tier-3)

### Phase 4: Consumer Contract Management — IN PROGRESS (new in Rev 2)
- Fifth message classification — designed (this doc), not yet wired into `feedback_monitor.py`
- Consumer SLA documentation — designed (§6), pending USE_CASES.md update
- `for:consumer-contract` label — to be added on first use

## 11. Success Criteria

- A user posts feedback in GChat → triaged + Issue filed within 2 hours.
- A bug report → reproduced, fixed, validated within 24 hours.
- A new PyTorch sweep → corpus updated with changelog within 1 day of sweep completion.
- `tools/validate.py` catches the classes of errors found in cold testing.
- Zero manual `corpus.json` editing for routine updates.
- A skill integration request → contract assessment surfaced to Peng within 24 hours; commitment communicated to consumer within 1 week.

## 12. Open Questions

- Should `tools/validate.py` run as a GitHub Action? (Web proxy unblocks GitHub access; CI feasibility worth re-examining.)
- How do external OSS users (non-Meta) file feedback? Currently GitHub Issues only — no GChat bridge.
- What's the right monitoring frequency — current 5x/day weekdays vs event-driven?
- How long do we keep historical `sweep_results/` directories before archiving?
- Should consumer-contract changes get formal versioning (e.g., schema version field in `corpus.json`)?

## 13. References

- Repo: `~/projects/oss-model-graph-break-corpus/`
- Design doc: `design/design-doc.md` (Rev 35; also Google Doc `1paCL1R8xoN6OajND8c4M5WgA68Uw1iEij-katYFqneM`)
- Charter delta: `design/charter-delta-2026-04-22.md`
- USE_CASES.md (consumer catalog)
- Repo CLAUDE.md (operating rules)
