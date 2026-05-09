# mre — Subagent for constructing Minimal Reproducers (V1)

This is a STANDALONE subagent. It constructs a verified Minimal Reproducer for a torch.compile / Dynamo / Inductor bug, given sweep evidence and a target torch venv.

V1 is intentionally minimal. The ledger collects rows but no analyzer is shipped. Augment when patterns emerge from real data.

Call from:
- **file-issue Mode B** when assembling an issue body that wants an MRE section
- **Issue-tackling workflows** when an existing issue needs a verified repro for triage / fix work

## When to invoke

- Sweep evidence describes a real failure (not a synthetic claim).
- Downstream consumer benefits from a standalone reproducer.
- You can afford up to 25 min wall-clock.

Skip if the original sweep command is already a one-liner the maintainer can run, OR the downstream gate accepts the symptom + sweep evidence alone (file-issue's stamped-absent path covers this).

## How to invoke

Spawn a local Agent with `subagents/mre/persona.md` as the system prompt. Pass input contract fields in the prompt. Read the structured output back.

Required input fields (else `INPUT_MISSING`):
- `case_id`
- `sweep_evidence_path`
- `time_budget_minutes` (typically 15 default; 25 for hard classes)
- `target_torch_venv`

Optional: `error_class`, `provenance_anchor`, `affected_models`.

Output is one of: `SUCCESS`, `VERIFICATION_FAILED`, `PROVENANCE_UNKNOWN`, `TIME_BUDGET_EXHAUSTED`, `STRATEGY_UNKNOWN`, `INPUT_MISSING`. The 6 RESULT outcomes line up 1:1 with ledger outcomes. See `persona.md` for the full output contract.

## What you do with the output

### `SUCCESS`
- file-issue Mode B: insert `mre_bytes` into the body's MRE section as a single ` ```python repro=true ` fence; insert `expected_signal` JSON into the matching `<details>` block.
- tackle-issue: hand `mre_bytes` to the maintainer / fix-author.
- Confirm `verify_repro_json.classification == "reproduces"` before relying on the MRE.

### `VERIFICATION_FAILED`
- The Agent produced bytes but verify_repro disagreed. Bytes preserved in ledger + output for forensic.
- file-issue Mode B: ship body with stamped-absent MRE section. Note in case file: `mre_status: "attempted_verification_failed"`. Do NOT post the un-verified bytes as if they reproduced.
- tackle-issue: report partial repro to caller WITH negative classification clearly tagged.

### `PROVENANCE_UNKNOWN`
- file-issue Mode B: ship body with stamped-absent MRE section. Note `mre_status: "no_attempt_provenance_unknown"`.
- tackle-issue: report to caller; ask for upstream traceback.

### `TIME_BUDGET_EXHAUSTED`
- file-issue Mode B: ship body with stamped-absent MRE section. Note `mre_status: "attempted_time_budget_exhausted"`, `last_attempt_status: <from output>`.
- tackle-issue: report partial; caller decides.

### `STRATEGY_UNKNOWN`
- Either pick the closest existing strategy and re-invoke with `error_class` set + a note, OR stamp absent and let the ledger entry motivate adding a new strategy via deep-dive.

### `INPUT_MISSING`
- Fix the missing input, re-invoke. Not a real outcome.

## Self-learning (V1: just collect, no analyzer yet)

Each invocation appends one row to `subagents/mre/ledger.jsonl`. Schema in `persona.md`. No analyzer is shipped in V1.

Once we have ~20+ rows, decide what's worth analyzing based on what patterns emerge. Possibilities (don't pre-build):
- Per-error-class success rate
- Median time-spent
- Common failure_modes
- Per-strategy effectiveness

When a pattern is real and the analysis is worth automating, write the analyzer then. Strategy additions to `persona.md` go through adversary-review (per the `subagents/` edit trigger in CLAUDE.md).

## Files

- `persona.md` — system prompt for the spawned Agent
- `SKILL.md` — this file (caller guide)
- `ledger.jsonl` — append-only per-invocation log

## Cautionary tale (2026-05-09)

Otter drafted MRE bodies for 3 dynamo issues (#99, #98, #92) by HYPOTHESIS — reading break_reason text and constructing code that "looked like" what should trigger it. The #99 hypothesis MRE failed verify_repro immediately: empty stderr, exit 0, expected fragment absent on torch 2.9.0+cu128.

This subagent's PROVENANCE rule + hill-climbing-with-fragment-as-oracle technique exists to prevent that failure mode.

## Versioning

V1 (2026-05-09): minimal subagent. Strategies A-E. 15/25 min budget. Ledger collects rows; no analyzer. Adversary-reviewed (case adv-2026-05-09-151200-mre-subagent-design); 6 of 11 gaps addressed in V1, 5 deferred to V2 with reason "premature optimization; defer until ledger has data."
