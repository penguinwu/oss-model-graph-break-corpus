# file-issue close-mode — design (rev 2)

**Author:** Otter
**Date:** 2026-05-10 (rev 2: 15:15 ET — simplified per Peng directives 15:09 ET)
**Status:** Implementation in flight
**WS1 task:** "Build file-issue close-mode workflow"

---

## What changed from rev 1

Rev 1 designed close-mode around 3 attribution tests via `tools/verify_repro.py` (current torch / prior torch + current transformers / prior torch + prior transformers), modeled on `methodology.md` R3 brief-composition rule. **Rev 2 corrects two errors in rev 1:**

1. **Wrong evidence basis.** Rev 1 used MRE re-runs as ground truth. Per Peng directive 15:09 ET, the MRE is a developer investigation tool — it may not completely represent the original failure, and relying on it for close decisions is a known failure mode. The ORIGINAL FAILED MODELS in the latest sweep are the ground truth.
2. **Wrong evidence depth.** Rev 1 demanded attribution (torch vs transformers vs vacuous). Per Peng directive 15:09 ET, the close semantics is "fixed on trunk" — we rarely fix on prior releases. Attribution semantics belong to brief composition (R3 / Section 2 "Pure Dynamo wins"), NOT to issue closure.

Net: close-mode shrinks from ~600 LOC to ~150 LOC. No verify_repro. No prior-torch venvs needed. Most of the check ALREADY exists as `tools/file_issues.py::classify_close_candidate`.

## What it does

Wraps the EXISTING close-stale check (`classify_close_candidate` ⇒ `auto-close` requires ALL originally-affected models to be `fullgraph` in current sweep) behind the file-issue audit chain:

- Mode A_close (adversary) verifies the disposition + staleness; emits one of 4 verdicts
- Mode B_close (assembler) writes the closing comment with per-model evidence table
- `tools/file_issues.py corpus-issue --close --via-skill <case_id>` enforces case-file presence + verdict + body sha
- `<sweep_dir>/.audit-rerun-required` marker is honored (refuses if present, per audit_new_errors gap #5 deferred)

## Inputs

1. `<sweep_dir>/sweep-report.json` — the close_candidates plan (already produced by `tools/file_issues.py sweep-report`)
2. `<sweep_dir>/sweep_state.json` → `versions.torch` + `started` (for staleness gate)
3. `<sweep_dir>/.audit-rerun-required` (optional) — marker honored
4. The case file at `subagents/file-issue/invocations/<case_id>.md`

No verify_repro. No model re-runs. No GitHub API beyond the close+comment posts.

## Mode A_close (adversary)

Verdict space (4):
- **`close`** — `classify_close_candidate(candidate) == "auto-close"` AND sweep age ≤ 10 days AND no `.audit-rerun-required` marker
- **`reject-keep-open`** — `classify_close_candidate` returned `review-needed:*` (partial flip, still breaking, reclassified, etc.). Surface to human; do not close.
- **`reframe`** — sweep age > 10 days. Action: re-run nightly + sweep_compare; re-invoke close-mode against fresh data.
- **`block-stale-rerun`** — `.audit-rerun-required` marker present. Action: re-run affected models OR delete the marker with documented reason (`--ack-stale-rows-noop`); then re-invoke.

The verdict is mechanical given the inputs — Mode A_close is largely a check, not a judgment call.

## Mode B_close (assembler)

Emits the closing comment body with sha-pinned audit chain:

```markdown
## Auto-closed by Step 2c on YYYY-MM-DD

This issue tracked <N> originally-affected (model, mode) pairs. On the latest
nightly sweep (`<sweep_label>`, torch <ver>, sweep age <D> days), **all <N>
pairs now compile fullgraph** — the underlying gap appears to be fixed on
trunk.

**Per-model status (current sweep):**
| Model | Mode | Baseline status | Current status |
|---|---|---|---|
| ModelA | eval | eager_error    | fullgraph |
| ModelA | train | graph_break   | fullgraph |
| ...

**Closing as fixed on trunk.** Per the corpus close-mode policy, attribution
(torch vs transformers vs vacuous) is NOT investigated at close time — we
rely on "original models now pass on current trunk" as the close criterion.
Attribution-level claims (e.g., "Dynamo win") live in the weekly brief.

If this is incorrect (e.g., the gap moved to a different model class, or the
auto-detection missed a related symptom), reopen and add a `do-not-auto-close`
label.

<sub>via subagents/file-issue close-mode case_id=<case_id> sweep=<sweep_dir>
sweep_age_days=<D> torch=<ver></sub>
```

Mode B_close failure markers (carry over from Mode B for NEW):
- `OVERSCOPE` — comment >4 paragraphs after self-revision (rare for close — body is templated)
- `VALIDATION_FAILED` — pre-submission gate flag (PII, missing fields, etc.)

## Tooling — `tools/file_issues.py corpus-issue --close`

Argparse signature:

```
python3 tools/file_issues.py corpus-issue \
    --close <issue_num> \
    --via-skill <case_id>           # required=True
    --plan <sweep-report.json>      # required=True (close_candidates source)
    --body <body_path>              # required=True (Mode B output)
    --close-reason completed|not_planned   # required=True
```

Validation BEFORE any GitHub API call:
1. `--via-skill` case file exists; `mode_a_verdict == "close"`; `mode_b_sha256` non-empty
2. `body_sha256` matches the actual body file
3. `--plan` exists; close_candidates contains the issue number; `classify_close_candidate(candidate) == "auto-close"`
4. Plan metadata `timestamp` is ≤ 10 days old (staleness gate)
5. `<sweep_dir>/.audit-rerun-required` does NOT exist (where sweep_dir is derived from the plan path's parent)
6. `--close-reason` ∈ {`completed`, `not_planned`}

If any check fails, exit non-zero with a specific error. NO close performed. NO comment posted.

Tested by `tools/test_file_issues.py::test_close_*` (5+ tests).

## Persona.md amendment

Add a new section to `subagents/file-issue/persona.md` after Mode A / Mode B:

```markdown
## Mode A_close — adversary for close-mode

[verdict space (close / reject-keep-open / reframe / block-stale-rerun),
 input: candidate from sweep-report.json plan + sweep_state.json,
 mechanical check: classify_close_candidate + staleness + marker]

## Mode B_close — assembler for close-mode

[body template, sha-pinned audit footer, failure markers]
```

Persona changes go through adversary-review per local CLAUDE.md trigger.

## What this REPLACES

Today's close paths:
1. `tools/file_issues.py close-stale --apply` — bulk auto-close, no `--via-skill` gate. **Phase A:** keep close-stale alive but ONLY invoked by the cron prompt's wake step (no ad-hoc agent invocation). **Phase B (separate task):** retrofit close-stale --apply to invoke close-mode per-candidate.
2. One-off `urllib` / `gh` / Python scripts — FORBIDDEN per workflow doc § Step 2c "No bypass paths"; close-mode is the only sanctioned path.

## Test plan (rev 2)

7 tests for `tools/test_file_issues.py`:
1. `test_close_requires_via_skill` — argparse refuses without `--via-skill`
2. `test_close_requires_plan_and_body` — argparse refuses without those
3. `test_close_refuses_non_close_verdict` — case file says `close`, but `classify_close_candidate` returns `review-needed:*` → tool refuses
4. `test_close_staleness_gate_refuses_old_plan` — plan timestamp > 10 days → tool refuses
5. `test_close_marker_present_blocks_close` — `.audit-rerun-required` exists → tool refuses
6. `test_close_body_sha256_round_trip` — body modified after Mode B → tool refuses
7. `test_close_reason_validation` — invalid `--close-reason` rejected at argparse

## Manual gates

- Mode A_close verdict review — implicit (verdict drives the action)
- Mode B_close body review — surface proposed body to Peng's space; wait for explicit approval; THEN run `--close`
- External Engagement Approval — corpus repo is below the gate threshold (own repo) but a posting block is still surfaced for visibility

## Implementation scope (rev 2)

- `subagents/file-issue/persona.md`: ~50 lines added (Mode A_close + Mode B_close)
- `tools/file_issues.py`: ~120 lines added (`--close` subcommand + 6-step validation + `.audit-rerun-required` honor)
- `tools/test_file_issues.py`: ~150 lines added (7 tests)
- Small edit to `subagents/file-issue/SKILL.md` (Step 5 authority gate row for close)

Estimated total: ~320 lines. Vs rev 1's ~600+ lines.

## Adversary-review case

`adv-2026-05-10-152000-close-mode-design` — to be created with this implementation commit.
