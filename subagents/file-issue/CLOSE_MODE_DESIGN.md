# file-issue close-mode — design

**Author:** Otter
**Date:** 2026-05-10
**Status:** DRAFT — awaiting Peng review
**WS1 task:** "Build file-issue close-mode workflow"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2c (close-mode attribution tests A/B/C)
**Related skill:** `subagents/file-issue/SKILL.md` (Mode A / Mode B for NEW + EDIT; close-mode extends both)

---

## What it does

Adds **close-mode** to the file-issue subagent so closing an issue goes through the same per-case rigor as filing one: Mode A_close (adversary reviews the close decision) + Mode B_close (assembles the closing comment with structured audit chain) + `tools/file_issues.py corpus-issue --close <num> --via-skill <case_id>` enforcement.

This **replaces** today's bypass-prone paths:
- `tools/file_issues.py close-stale --apply` (no `--via-skill` gate; can close in bulk without per-issue evidence)
- One-off `urllib` / `gh` / Python scripts (the path that produced the 2026-05-10 morning incident — closed #21, #26, #27 without surfacing evidence to Peng)

Both bypass paths are the failure mode this design exists to mechanically prevent.

## The 3 attribution tests (workflow doc § Step 2c)

A sweep-data flip alone is NOT sufficient evidence. Mode A_close runs three tests against the candidate issue's MRE. The verdict drives the close framing:

| Test | What it runs | Outcome → Framing |
|---|---|---|
| **A** | MRE on CURRENT torch | Still reproduces → NOT a close candidate (sweep flip is misleading; the gap is real). Verdict: `reframe-or-reject`. |
| **B** | MRE on PRIOR torch + CURRENT transformers | No longer reproduces here either → root cause is transformers code drift. Framing: `closed-as-not-applicable`. |
| **C** | MRE on PRIOR torch + PRIOR transformers (original repro env) | Still reproduces → torch attribution confirmed; framing: `closed-as-completed (torch fixed)`. No longer reproduces → model removed/rewritten; framing: `closed-as-vacuous`. |

**Only Test C confirming "still reproduces in original env, no longer reproduces on current torch" justifies attribution to a torch fix.**

The tool reuses `tools/verify_repro.py` (already exists — used by NEW path Step 2.5) for each test cell. Three test cells per close-candidate × 1 venv each = 3 verify_repro JSON files per case, written to `/tmp/file-issue-<case_id>-close-test-{a,b,c}.json`.

## Mode A_close (adversary)

Runs in the same shape as Mode A for NEW (Step 3 in `subagents/file-issue/SKILL.md`), but the verdict space is different:

```
VERDICT: close-completed | close-not-applicable | close-vacuous | reject-keep-open | reframe
GAPS_FOUND: <list>
ATTRIBUTION_TEST_RESULTS:
  test_a (current torch):       reproduces | does-not-reproduce
  test_b (prior torch + cur tx): reproduces | does-not-reproduce
  test_c (prior torch + prior tx): reproduces | does-not-reproduce
  attribution: torch | transformers | vacuous | unknown
REJECT_REASON: <only if reject-keep-open — paragraph>
```

Verdict mapping:
- All 3 tests run + Test A does-not-reproduce + Test C reproduces → `close-completed` (torch attribution confirmed)
- Test A does-not-reproduce + Test B does-not-reproduce → `close-not-applicable` (transformers drift)
- Test A does-not-reproduce + Test C does-not-reproduce → `close-vacuous` (model removed/rewritten)
- Test A reproduces → `reject-keep-open` (gap is real; sweep flip is misleading)
- Tests can't be run (MRE corrupted, prior venvs unavailable) → `reframe`

The persona update is anchored on the same 4 criteria as NEW + EDIT — closing is a body-emitting operation, so calibration applies. Closing comment must:
1. Self-contained (audit chain readable without external context)
2. Concise (1-2 paragraph close justification + table; no story)
3. Trustworthy (cites the 3 verify_repro JSONs by sha256)
4. Actionable-as-reproducible (a maintainer who disputes the close can reproduce by re-running the 3 tests with the cited venvs)

## Mode B_close (assembler)

Same shape as Mode B for NEW + EDIT; emits a closing comment body (not a new issue body) with the structured audit chain:

```
TITLE: (unused — close operation, no title change; field included for schema parity but ignored at posting time)
LABELS: (unused; same)
BODY:
## Auto-closed by Step 2c on YYYY-MM-DD

This issue tracked <N> model(s) breaking on `<pattern>`. Per file-issue close-mode (case_id=<case_id>):

**Sweep evidence (current week):**
- Affected models <flipped|partial flipped|removed>: <list>
- Source sweep dir: `<sweep_dir>`

**MRE re-verification (3 attribution tests):**
| Test | Env | Outcome |
|---|---|---|
| A | torch <current_ver> + transformers <current_ver>     | does-not-reproduce |
| B | torch <prior_ver>   + transformers <current_ver>     | does-not-reproduce |
| C | torch <prior_ver>   + transformers <prior_ver>       | reproduces         |

**Attribution:** torch (test C reproduces in original env; test A does not on current torch).
**Close framing:** closed-as-completed.

If this is incorrect (e.g., the gap moved to a different model class), reopen and add `do-not-auto-close`.

<sub>verify_repro JSONs: test_a sha=…, test_b sha=…, test_c sha=…</sub>
```

The footer references the verify_repro JSONs by sha256 — the case file's `body_sha256` covers the full body so any tampering between Mode B output + post is caught at the CLI gate.

Failure markers (`OVERSCOPE`, `MRE_TOO_LARGE`, `VALIDATION_FAILED`, `MODE_NOT_SPECIFIED`) carry over from Mode B for NEW with the same disposition rules (`subagents/file-issue/SKILL.md` Step 4.5).

## Tooling — `tools/file_issues.py corpus-issue --close`

New subcommand mode. Argparse signature:

```
python3 tools/file_issues.py corpus-issue \
    --close <issue_num> \
    --via-skill <case_id>           # required=True
    --close-test-a <json_path>      # required=True
    --close-test-b <json_path>      # required=True
    --close-test-c <json_path>      # required=True
    --body <body_path>              # required=True
    --close-reason completed|not_planned   # required=True (GitHub state_reason field)
```

Validation BEFORE any GitHub API call:
1. `--via-skill` case file exists; `mode_a_verdict in {close-completed, close-not-applicable, close-vacuous}`; `mode_b_sha256` non-empty
2. `body_sha256` matches the actual body file (catches tampering)
3. All 3 verify_repro JSONs exist + are valid + each cell's classification matches what Mode A_close recorded
4. `--close-reason` is one of GitHub's accepted values (`completed` for `close-completed`, `not_planned` for the other two close framings)

If any check fails, exit non-zero with a specific error. NO close performed.

Tested by `tools/test_file_issues.py::test_close_requires_via_skill_and_three_tests` and friends.

## Migration of close-stale

Today: `tools/file_issues.py close-stale --apply` walks plan candidates and closes in bulk via comment + PATCH state=closed (`tools/file_issues.py:1310-1352`). It has no `--via-skill` gate.

Two-phase migration:
1. **Phase A (this design):** add the `--close` subcommand with full close-mode gating. Update local CLAUDE.md trigger so any close operation (single or batch) MUST go through `--close --via-skill <case_id>`. Direct `urllib` / `gh` / one-off scripts FORBIDDEN (already in workflow doc § Step 2c § "No bypass paths").
2. **Phase B (separate task):** rewrite `close-stale --apply` to invoke close-mode per-candidate (one case_id per issue), so the bulk operation respects per-issue rigor. UNTIL Phase B ships: `close-stale --apply` continues to be invoked ONLY by the cron prompt's wake step, and even then only with the dry-run output reviewed first. NO ad-hoc invocation by an agent.

Phase B is a follow-up task; this design covers Phase A.

## Persona.md amendment

Add a new section after Mode A / Mode B:

```
## Mode A_close — close-mode adversary

[verdict space, attribution test outcomes, calibration anchored on 4 criteria]

## Mode B_close — close-mode assembler

[closing comment body template, sha256 footer convention, failure markers]
```

Persona changes go through adversary-review per local CLAUDE.md trigger (subagents/* persona = scoring logic).

## Manual gates

| Gate | What requires Peng approval |
|---|---|
| Cluster plan covering close candidates | Step 0c (existing — same as NEW path); the cluster plan's `affected_cases` includes the close case_ids |
| Mode A_close verdict review | Implicit — Mode A's verdict is the gate; reject-keep-open requires no further action |
| Mode B_close body review | Same as Mode B for EDIT — surface proposed closing body to Peng's space; wait for explicit approval; only then run `--close` |
| External Engagement Approval | Per local CLAUDE.md — closing a corpus issue is below the threshold (own repo) but a posting block is still surfaced for visibility |

## Test plan

- Unit: `tools/test_file_issues.py::test_close_requires_via_skill_and_three_tests` — argparse refuses any close without `--via-skill` + 3 test JSONs
- Unit: `test_close_refuses_on_test_a_reproduces` — case file says `mode_a_verdict=close-completed` but `close-test-a.json` shows `reproduces`; tool refuses (catches Mode A / actual divergence)
- Unit: `test_close_body_sha256_round_trip` — body modified after Mode B output; tool refuses
- Unit: `test_close_reason_validation` — invalid `--close-reason` rejected at argparse
- Integration (against corpus repo, dry-run mode): walk an existing closed-then-reopened issue (#21, #26, or #27) through close-mode; verify the proposed close produces the right verdict + body without actually re-closing

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Prior venv unavailable for Test B/C | Mode A returns `reframe`; surfaces to Peng with the missing-venv evidence; close DOES NOT proceed (no escape hatch) |
| Bulk close-stale path remains the bypass | Phase B follow-up; Phase A makes the safe path obvious AND adds the trigger so agents reach for `--close --via-skill` first |
| Close-mode post-validation is heavier than NEW post-validation (3 tests vs 4 cells; comparable cost) | Acceptable; close is a per-issue decision, not a batch — the per-case cost is amortized over the value of irreversible close decisions |
| Close-completed claim later disputed by maintainer | Audit chain (3 verify_repro JSONs sha-pinned in body) lets a maintainer reproduce the test; `do-not-auto-close` reopen path documented in body |

## Implementation scope

- `subagents/file-issue/persona.md`: ~150 lines added (Mode A_close + Mode B_close blocks)
- `tools/file_issues.py`: ~200 lines added (`--close` subcommand + validation gates)
- `tools/test_file_issues.py`: ~100 lines added (4 new tests)
- `subagents/file-issue/SKILL.md`: small edits (Step 5 authority gate row for close, +failure markers section pointer)
- `sweep/WEEKLY_SWEEP_WORKFLOW.md`: small edit — Step 2c "(NOT YET BUILT)" → "shipped, see CLOSE_MODE_DESIGN.md"
- Adversary-review on persona.md changes + on file_issues.py changes (both = scoring logic)

Estimated effort: 6-8 focused hours including tests + adversary-review iterations.
