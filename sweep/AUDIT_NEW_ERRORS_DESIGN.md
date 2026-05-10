# tools/audit_new_errors.py — design (rev 2)

**Author:** Otter
**Date:** 2026-05-10 (rev 2: 14:35 ET — adversary-driven changes per case adv-2026-05-10-145000)
**Status:** Implementation in flight
**WS1 task:** "Build `tools/audit_new_errors.py`"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2a
**Adversary case file:** `subagents/adversary-review/invocations/adv-2026-05-10-145000-audit-new-errors-design.md` (12 gaps; 11 addressed in rev 2, 1 deferred to a new tool)

---

## What it does

For each ERROR row in current sweep that's NEW this week (not in baseline at the same status, not covered by `known_errors.json`, not in `skip_models.json`), classify via heuristic and emit a markdown report + JSON sidecar with proposed triage.

Surfaces only — never auto-writes config files. Reviewer (Peng) approves each action.

## Per Peng directive

- Bias INFRA-FIX over add-to-list. Default action: fix root cause.
- create_errors are entirely infra's fault and MUST be fixed at root.
- Tool surfaces; does NOT auto-write known_errors.json or skip_models.json.

## Inputs (rev 2 — pinned to actual file shapes)

1. `<sweep_dir>/identify_results.json` (jsonl with metadata header). Loaded via `sweep/results_loader.py::load_effective_results` — returns dict `{(name, mode): row}` of effective post-auto-retry rows. **Authoritative source for classification** per disposition #10.
2. `<sweep_dir>/identify_streaming.jsonl` — diagnostic context only (first-seen error before retry). NOT used for classification.
3. `<sweep_dir>/compare-vs-baseline.json` — when present, gives cat 1-6 partition for filtering NEW vs stable. Optional; degraded mode if absent.
4. `sweep/known_errors.json` — per-version applicability filter.
5. `sweep/skip_models.json` — out-of-scope models.
6. `<sweep_dir>/sweep_state.json` → `versions.torch` — for known_errors version filter.

## Real-data field reference (verified against `sweep_results/nightly/2026-05-09`)

Effective rows have these load-bearing fields:
- `name`, `mode`, `status` (one of `full_graph`, `graph_break`, `eager_error`, `create_error`, `worker_error`, `timeout`)
- `error` — error message text (string, possibly empty)
- `error_type` — Python exception class name when present (e.g., `RuntimeError`, `AttributeError`, `OutOfMemoryError`, `AcceleratorError`)
- `phase` — execution phase reached (`create`, `eager`, `done`)
- `phase_at_timeout` — present on timeout rows (`create`, `eager`)
- `returncode` — subprocess exit code on worker-died rows (POSIX signal -N for kernel kills)
- `retry_note` — present when auto-retry pass touched the row (`confirmed_error` = stable; `flaky: was X, now Y` = transient resolved)
- `wall_time_s`, `gpu_mem_mb` — for tier classification (consumed by audit_new_models.py)

## Candidate selection (rev 2)

A row `(name, mode)` is a candidate if ALL hold:

1. Status ∈ `{eager_error, create_error, worker_error, timeout}` (full ERROR_STATUSES set per disposition #3 — workflow doc lists all 4)
2. NOT in `skip_models.json` (model name set)
3. NOT covered by `known_errors.json` for active torch version (see "Version filter" below)
4. If `compare-vs-baseline.json` present: NOT in cat 6 stable (status unchanged from baseline). Cat 6 is reported by sweep_compare directly.
5. If `compare-vs-baseline.json` absent: ALL error rows are candidates (degraded mode; tag report as "PARTIAL — no baseline available").

## Triage classes (rev 2 — structured-field-first per disposition #9)

Rules are evaluated **top-to-bottom; first match wins** (per disposition #11). Each rule has a `case_id` for traceability.

| Order | Class | Rule (structured fields) | Refinement (substring) | Action |
|---|---|---|---|---|
| 1 | `venv-bootstrap-broken` | `status==worker_error` | `error` contains `Unable to load`, `Cannot load symbol`, `libcudnn`, `403 Forbidden`, `has not been allowlisted` | **STOP triage — fix venv first.** Reference `~/.myclaw-shared/recipes/python-venv-bpf.md`. |
| 2 | `gpu-contention` | `error_type==OutOfMemoryError` OR (`status==worker_error` AND `returncode==-9`) | n/a | Wait for auto-retry serial pass; if still OOMs, propose tier upgrade. |
| 3 | `cuda-context-pollution` | `error_type==AcceleratorError` OR `error` contains `device-side assert` | n/a | Wait for auto-retry; if still fails serially, file as upstream bug. |
| 4 | `subprocess-crash` | `status==worker_error` AND `returncode in {-6, -11}` (SIGABRT, SIGSEGV) | n/a | Wait for auto-retry. If `retry_note==confirmed_error`, file as upstream bug. |
| 5 | `fixture-bug` | `error_type==RuntimeError` AND `phase==eager` | `error` contains `Audio must be mono`, `Image features and image tokens do not match`, `expected sequence of length`, `Sizes of tensors must match` | Fix `sweep/worker.py` input synthesis. PR-blocker — must land before brief. |
| 6 | `tier-upgrade` | `status==timeout` AND `phase_at_timeout in {create, eager}` | n/a | Propose `large_models.json` entry (very_large tier per `phase_at_timeout==create` else large). Defers actual proposal to audit_new_models.py. |
| 7 | `upstream-bug` | `error_type==AttributeError` OR `error` contains `transformers` / `torch.` API | n/a | Propose new GitHub issue via file-issue subagent (Step 2c). |
| 8 | `unknown` | (no other match) | n/a | Surface to human reviewer. |

**case_id attribution per rule:** the heuristic table is hardcoded in the script with each entry tagged by `adv-2026-05-10-145000-audit-new-errors-design` (initial case). Future additions tag with the case_id that introduced them. Coverage test: every case_id in the table MUST have at least one fixture row exercising it.

## Re-run scope MARKER (rev 2 — mechanical gate per disposition #5)

When ANY `fixture-bug` candidate is surfaced, audit emits `<sweep_dir>/.audit-rerun-required`:

```
# Models needing re-run after fixture-bug fix; one (model, mode) per line.
HiggsAudioV2TokenizerModel|eval
HiggsAudioV2TokenizerModel|train
```

Step 2c tools (`tools/file_issues.py corpus-issue`, close-mode when built) read this file at startup and refuse to run if it exists. The reviewer either re-runs the affected models (which removes the marker) OR explicitly deletes it with a documented `--ack-stale-rows-noop` flag.

**Consumer-side honor:** the marker is EMITTED by audit. The HONOR side (file_issues.py refusing to start) is a SEPARATE WS1 task added to PLAN.md (it's a behavior change in file_issues.py and needs adversary-review on its own). For this audit's MVP, the marker is emitted and the workflow doc Step 2c gains language that says "human reviewer must check the marker." The full mechanical block lands in close-mode's adversary-review pass.

## Version filter — `applies_to_versions` (rev 2 — fail-loud per disposition #6)

`known_errors.json` entries gate by `applies_to_versions: [...]` (major.minor list).

**Missing field → audit fails LOUD** (exit 2, error message names the entry). Forces explicit decision per workflow doc's "discouraged but allowed for legacy entries" — we make it un-allowed.

**Patch-release granularity:** match is on major.minor. If a fix lands in 2.13.1, the entry must be narrowed manually (e.g., to `["2.10", "2.11"]`). Documented limitation; revisit if patch-release fixes become common.

## Cat 6 known_errors mismatch (rev 2 — per disposition #7)

A separate report subsection: for each cat 6 row covered by `known_errors.json`, verify the entry's `error_pattern` substring is actually present in the row's effective `error`. Mismatches → surface as candidates (the entry might be hiding a different root cause).

## Output format

### Markdown (`<sweep_dir>/audit-new-errors.md`)

```
# New errors triage — sweep YYYY-MM-DD vs baseline YYYY-MM-DD
torch: <ver>  transformers: <ver>  diffusers: <ver>

## Summary
- N total ERROR rows in current sweep (incl. worker_error + timeout)
- N covered by known_errors.json
- N stable failures (cat 6; reported by sweep_compare, not here)
- N candidates surfaced below

## Candidates (in priority order)
### venv-bootstrap-broken (count) — STOP, fix venv
### gpu-contention (count)
### cuda-context-pollution (count)
### subprocess-crash (count)
### fixture-bug (count) — PR-blocker; re-run-required marker emitted
### tier-upgrade (count)
### upstream-bug (count)
### unknown (count) — manual triage

## Cat 6 known_errors mismatch (count) — entries may be hiding different root causes

## Re-run-required marker
`<sweep_dir>/.audit-rerun-required` was emitted (or NOT emitted if no fixture-bug candidates).

## Action checklist for Peng review
- [ ] STOP if any venv-bootstrap-broken — fix venv before re-running anything
- [ ] Approve fixture-bug fixes; re-run affected models BEFORE Step 2c
- [ ] Approve upstream-bug filings → file-issue subagent
- [ ] Triage unknown candidates manually
```

### JSON sidecar (`<sweep_dir>/audit-new-errors.json`)

```json
{
  "sweep_date": "...",
  "baseline_date": "...|null",
  "torch_version": "...",
  "summary": {"total_errors": N, "in_known_errors": N, "stable_cat6": N, "candidates": N},
  "candidates": [
    {
      "name": "...", "mode": "...", "status": "...",
      "error_type": "...", "phase": "...", "returncode": "...|null",
      "retry_note": "...|null",
      "error_first_line": "...",
      "triage_class": "...", "matched_rule": "<case_id>",
      "suggested_action": "...",
      "first_seen_error": "..."   // from streaming for diagnostic context
    }
  ],
  "cat6_mismatches": [...],
  "rerun_marker_emitted": true|false
}
```

## Manual gates

- `unknown` triage — human reads
- `upstream-bug` filing — file-issue Step 5 authority
- known_errors.json adds — DISCOURAGED for create_error per Peng directive; mechanical enforcement deferred to a NEW PLAN.md task `tools/check_known_errors.py` (separate adversary-review)

## CLI

```
python3 tools/audit_new_errors.py <sweep_dir>
```

Exit codes:
- 0 — report written; candidates surfaced (or zero candidates)
- 1 — input parse error or no `identify_results.json`
- 2 — known_errors.json entry missing `applies_to_versions` (fail-loud per rev 2)
- 3 — `compare-vs-baseline.json` absent → degraded mode (report tagged PARTIAL; still exits non-zero so caller knows)

## Test plan (rev 2)

7 critical tests pinned to real-data fixtures from 2026-05-09:

1. `test_field_names_match_real_data` — 4 verbatim rows from 2026-05-09 effective; every row classifies to a non-`unknown` class. Catches: gap #1 (wrong field name).
2. `test_classifies_real_2026_05_09_candidates` — the 8 effective error rows (HiggsAudio×2, PI0Model×2, Rwkv*×4) each get the expected class.
3. `test_venv_bootstrap_broken_class` — synthetic worker_error row with cuDNN load failure → `venv-bootstrap-broken`. Catches: gap #2.
4. `test_uses_effective_row_not_streaming` — fixture where streaming has "Audio must be mono" but effective has "Sizes of tensors must match" → effective wins. Catches: gap #10.
5. `test_applies_to_versions_missing_fails_loud` — known_errors entry without field → exit 2. Catches: gap #6.
6. `test_branch_precedence_first_match_wins` — row matching multiple heuristics → expected class per top-to-bottom order. Catches: gap #11.
7. `test_rerun_marker_emitted_on_fixture_bug` — fixture with a fixture-bug candidate → `.audit-rerun-required` exists with `(name, mode)` lines. Catches: gap #5 (emission side).

Deferred tests (separate tasks):
- bias_infra_fix_blocks_create_error → covered by future `tools/check_known_errors.py`
- cat6_known_errors_pattern_mismatch → covered by `cat6_mismatches` schema in JSON; test added when consumer surfaces drift
- schema_compose_brief_consumer → moot (compose_brief dropped)

## Implementation scope

- `tools/audit_new_errors.py`: ~250 lines
- `tools/test_audit_new_errors.py`: ~250 lines (7 tests + real-row fixtures)
- Two new PLAN.md tasks:
  1. `tools/check_known_errors.py` — bias-INFRA-FIX mechanical gate (deferred from gap #4)
  2. file_issues.py + close-mode marker honor (deferred from gap #5 consumer side)

Adversary-review: case file `adv-2026-05-10-145000` covers this design + impl. Commit message references it.
