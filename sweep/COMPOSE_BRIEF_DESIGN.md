# tools/compose_brief.py — design

**Author:** Otter
**Date:** 2026-05-10
**Status:** DRAFT — awaiting Peng review
**WS1 task:** "Build `tools/compose_brief.py`"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2d
**Source-of-truth for content rules:** `skills/weekly-sweep-brief/{template.md, methodology.md, SKILL.md}`
**Depends on:** `compare-vs-baseline.json` (sweep_compare wiring) + `audit-new-errors.json` + `audit-new-models.json`

---

## What it does

Pre-fills the 8 required sections of the weekly sweep brief from `tools/sweep_compare.py` output (cat 1-6 partition + per-pattern segmentation), runs the methodology.md self-check checklist mechanically, and emits a draft brief markdown for Peng review.

**Today** the brief is hand-composed: Otter walks the methodology rules manually each week. Hand-composition has surfaced repeated R2 violations (mixing cat 3 with cat 1/cat 4) and S1 reconciliation failures (headline numbers don't match section sub-totals). Code = mechanical guard for those rules.

## Per Peng directive

- The brief is for the EXTERNAL PT2 dynamo team audience (workplace group `1251481819891072`). All R-rules + S-rules in `methodology.md` apply mechanically — failures BLOCK the post.
- The composed brief is a DRAFT for Peng review BEFORE post (per workflow doc § Manual gates "Approve brief before post"). The tool emits markdown; posting is a separate step the human reviewer triggers.

## Inputs

1. `<sweep_dir>/compare-vs-baseline.json` (cat 1-6 partition; R1 source of truth)
2. `<sweep_dir>/audit-new-errors.json` (Step 2a output — for Section 4 "Compile regressions" + Section 7 "NEW break-reason types")
3. `<sweep_dir>/audit-new-models.json` (Step 2b output — for Section 6 "Newly compile-testable models")
4. `<sweep_dir>/sweep_state.json` → `versions.{torch,transformers,diffusers}` (header)
5. `<baseline_dir>/sweep_state.json` → same (header)
6. Optional: `<sweep_dir>/issues_actions.json` (Step 2c output — for Section 5; if missing, Section 5 emits "_No issue cleanup this week — see `subagents/file-issue/invocations/` for any in-flight cases._")
7. `skills/weekly-sweep-brief/template.md` (8-section structure; tool reads it as the layout, doesn't duplicate)
8. `skills/weekly-sweep-brief/methodology.md` (self-check checklist; tool walks each item programmatically)

No model re-runs. No GitHub API. No `--pattern` queries auto-issued (those require human judgment about WHICH pattern to highlight; tool surfaces top-N reasons so reviewer can decide).

## Section-by-section composition (matches `skills/weekly-sweep-brief/template.md` 8 sections)

### Section 1 — Headline

Auto-fill:
- Net GB delta on cat 3 (sum of `gb_delta` across all cat 3 entries — apple-to-apple)
- N flipped graph_break → full_graph (count from cat 1 with status flip eager_error|graph_break → full_graph)
- Attribution status: `VERIFIED on K models | UNVERIFIED — testing pending`. Read from a NEW input file `<sweep_dir>/attribution_test_results.json` (if present); absent → mark UNVERIFIED.
- Real upstream compile regressions count: cat 2 minus cat-2-rows-covered-by-known_errors (from audit-new-errors.json)

Hard-fail if any number derives from anywhere other than the 3 source JSONs. The tool refuses to compose if the source files are missing — no fallback to ad-hoc query.

### Section 2 — Pure Dynamo wins

Walk cat 1 entries with `current_status='full_graph'` AND `baseline_status in {graph_break, eager_error}`. Group by model cluster (e.g., all `Qwen3*` rows together). Emit:
- Table of clusters / models with baseline_status → current_status
- Table of break-reason types eliminated (counts across baseline `break_reasons` of the flipped models)
- "### Attribution test (verified on K models)" subsection — fill from `attribution_test_results.json` if present; else "### Attribution status: UNVERIFIED" + propose the `sweep/worker.py` re-run command pre-populated with the cluster's first 2 models

### Section 3 — Compile-success → compile-success with reduced GBs

Walk cat 3 entries with `gb_delta < 0`. Per-model table. Total GB reduction = sum.

### Section 4 — Compile regressions

Walk cat 2 entries. For each, cross-reference `audit-new-errors.json` to get the triage class (= "real" vs "known PT bug" vs "deterministic-mode trigger").
- If 0 real: state "0 real" + per-cat-2 row + reason
- If >0 real: per-regression detail table; for each, propose `subagents/file-issue/SKILL.md` invocation (with case_id stub) → human reviewer routes through file-issue Step 0.

### Section 5 — Issues — actions taken (umbrella-split policy)

Read from `<sweep_dir>/issues_actions.json` if it exists (this file is OPTIONAL; not yet built — it would come from the close-mode + EDIT-mode workflows once they ship). For now, emit a stub:

```
### Closed as superseded
_No issue cleanup this week_

### New issues filed
_No new issues filed this week_

### Updated existing issues with current data
_No edits this week_

### Net effect on tracked issues
_See `subagents/file-issue/invocations/` for any in-flight cases._
```

When close-mode + EDIT-mode workflows produce `issues_actions.json` (subsequent task), this section auto-populates.

### Section 6 — Newly compile-testable models

Per R6: cat 1 SUCCESSES + cat 4 SUCCESSES combined. Decomposition:
- Cat 4 (truly new): model count + work-item count + % full_graph
- Cat 1 (was-error-now-success): per prior status: eager_error → success: N, timeout → success: N, worker_error → success: N
- From `audit-new-models.json`: count of new models proposed for tier upgrade + count proposed for skip

### Section 7 — NEW break-reason types surfaced

Walk cat 4 explain rows. Compute per-(file, op) signature. Compare against baseline cat 3+ baseline cat 6 sigs. Emit table of patterns NOT seen in baseline. Highlight ops new to corpus (e.g., `aten.bincount`).

This is the existing `tools/sweep_compare.py` "NEW break reasons in new models" section verbatim — copy from `compare-vs-baseline.md` if present; recompute if not.

### Section 8 — Actionable for Animesh / PT2 team

Walk cat 3 GB regressions + cat 4 NEW patterns. For each, compute leverage:
- "If you fix `<pattern>`, `<N>` breaks across `<M>` models clear"
- Sort by leverage descending
- Cap at 5 ± 2 items (per template.md)
- Each item must name an issue # (read from `issues_actions.json` if present; else "to-be-filed via Step 2c")

If leverage can't be quantified for a candidate, the tool DROPS it (per S4 — "if you can't quantify leverage, the item shouldn't be in the actionable list").

## Self-check checklist (mechanical)

The tool walks each item in `methodology.md` § "Self-check checklist" programmatically:

| Check | How tool enforces |
|---|---|
| `sweep_compare.py --check-only` exit 0 | Tool subprocesses sweep_compare with `--check-only` against the same inputs; refuses to compose on non-zero |
| R1: every number from sweep_compare | All numbers in tool's output trace to one of the 3 source JSONs; no `subprocess.run("python -c ...")` for aggregates |
| R2: cat-3-explicit phrasing | Tool emits "regression on cat 3 common pairs" / "exposure from cat 1 / cat 4"; refuses to write the un-qualified version |
| R3: attribution VERIFIED ≥2 OR explicit UNVERIFIED | If `attribution_test_results.json` missing OR has < 2 verified models, tool tags Section 1 + Section 2 with UNVERIFIED |
| R4: umbrella-split policy applied | Tool reads `<sweep_dir>/issues_actions.json` `umbrella_splits` field; if any unsplit umbrella appears in candidates, refuses to compose |
| R5: search-existing-issues for new filings | The tool DOES NOT file issues — but emits proposed file-issue invocations with the `dup_search` field pre-populated for human reviewer |
| R6: cat 1 + cat 4 combined for "newly testable" | Section 6 hardcoded to combine; tool refuses Section 6 to use cat 4 alone |
| R7: signal-boost via `post_to_feedback.py` | Tool's emitted "next steps" section names the wrapper command verbatim; raw `gchat send` not present |
| R8: destination group name verified | Tool emits a `meta workplace.group details` snippet to be run before posting; the brief's destination is a {{GROUP_NAME}} placeholder filled at post-time |
| S1: headline reconciles with body | Tool computes the cross-section reconciliation programmatically and embeds the result in a `<!-- self-check: S1 reconciliation pass/fail -->` comment; if fail, refuses |
| S2: forbidden vague language | Linter sweep on the composed body for `likely`, `substantially`, `some models`, `a few models`; refuses if found |
| S3: audience-adapted detail | Linter sweep for bare `#NNN`, internal tool names without context, jargon ("cat 3" / "cat 4") used without intro definition; refuses if found |
| S4: actionable items name leverage | Section 8 entries without `breaks: N, models: M` are dropped (per logic above) |

Every check has a `--skip-check <name>` escape hatch with a `--reason <text>` mandatory companion. Skip lines land in the brief's footer comment for audit. Per Peng directive (file-issue precedent): skip = visible, never silent.

## Output format

`<sweep_dir>/brief-draft.md` — full markdown body matching `skills/weekly-sweep-brief/template.md` layout, with the 12-item self-check passes as `<!-- -->` comments at the bottom.

Plus `<sweep_dir>/brief-draft-meta.json` with:
- `self_check_results`: per-rule pass/fail
- `attribution_status`: VERIFIED / UNVERIFIED + count
- `pending_filings`: list of `{cluster_id, dup_search_query, suggested_target}` for Section 5
- `pre_post_commands`: the verbatim `meta workplace.group details` + `meta workplace.post create` + `post_to_feedback.py send` commands the reviewer runs

## Manual gates

| Gate | What requires reviewer approval |
|---|---|
| Approve brief before post | Reviewer reads `brief-draft.md` + `brief-draft-meta.json`; approves explicitly before any post step |
| Skip-check use | Each `--skip-check X --reason Y` flag becomes a visible footer line; reviewer challenges if reason isn't strong |
| External Engagement Approval (per local CLAUDE.md) | Posting to workplace group + post_to_feedback.py is gated by the External Engagement rule; tool emits the proposal block contents but does NOT post |

## CLI

```
python3 tools/compose_brief.py <sweep_dir> [--baseline <baseline_dir>] [--out-dir <dir>]
                               [--skip-check <name> --reason <text>]...
```

Defaults match `audit_new_errors.py`. Exit codes:
- 0: brief draft written, all self-checks pass (or skipped with reasons)
- 1: a self-check failed AND no `--skip-check` for it; refusal message names the failed check + how to skip if intentional
- 2: input parse error (missing source JSONs)

## Test plan

- Unit per-section: each of the 8 sections has a unit test with crafted inputs — e.g., Section 1 reconciliation between cat 3 sums + headline, Section 6 cat 1 + cat 4 combination
- Self-check mechanical: each of the 12 checklist items has a positive + negative fixture
- Linter sweep tests: forbidden language fixtures (S2/S3) + audit-trail comment generation
- Integration: run against `sweep_results/nightly/2026-05-09` (live data) — verify the composed brief is a "quiet week" brief (per WS2 sub-checkpoint Step 2d note: 0 dynamo wins, 0 dynamo regressions, 5 non-dynamo improvements). Compare against the methodology.md hand-composed structure.
- Regression: snapshot test on a known-good prior week (2026-05-03 brief) — re-compose from frozen inputs, diff against the human-composed final.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Mechanical self-check is too rigid; legitimate exceptions blocked | `--skip-check X --reason Y` escape hatch + visible footer line |
| Tool's auto-grouping in Section 2 (cluster by name prefix) misclassifies | Section 2's grouping is a suggestion; Otter / Peng can re-group manually before post; tool emits both raw + grouped views |
| Inputs (audit-new-errors.json + audit-new-models.json) absent because Steps 2a/2b haven't shipped yet | Tool degrades gracefully — emits "Step 2a output not present; Section 4 + Section 7 emit minimal fallback" instead of refusing; tags brief draft as "PARTIAL — Steps 2a/2b not yet automated" |
| Section 5 stub looks empty until close-mode + EDIT-mode + issues_actions.json land | Acceptable for v1; Section 5 fills in once close-mode workflow ships AND emits `issues_actions.json` (this is the explicit dependency on the close-mode design above) |
| Self-check linter false-positives on legitimate use of "likely" | The linter rule is "no `likely` in load-bearing claims" — it only flags `likely` in Section 1 / Section 2 / Section 8 (load-bearing); Section 7's prose is exempt |

## What this UNBLOCKS in WS1

Once compose_brief.py ships:
- Step 2d goes from hand-composed (S1 reconciliation failures, occasional R2 violations) to mechanically-checked draft + reviewer approval
- The 12-item self-check is enforced not just by methodology.md prose but by the tool's exit-code semantics
- The brief-composition skill at `skills/weekly-sweep-brief/SKILL.md` updates to point at compose_brief.py as the entry point (Step 5 in that skill's workflow becomes "run `tools/compose_brief.py`" instead of "follow template.md by hand")

## Implementation scope

- `tools/compose_brief.py`: ~600-800 lines (8 section composers + self-check linters + reconciliation logic)
- `tools/test_compose_brief.py`: ~400-500 lines (per-section + per-rule fixtures + snapshot test)
- Small edit to `skills/weekly-sweep-brief/SKILL.md`: Step 5 pointer to the new tool
- One-line addition to `tools/run_experiment.py nightly` Step 5f (chain after audit_new_models): `compose-brief --sweep-dir <dir>` (allow_fail=True)
- Adversary-review on initial commit (composition logic = scoring / reasoning over sweep evidence)

Estimated effort: 8-12 focused hours (the largest of the 5 designs). Recommended to ship in 3 increments:
1. Sections 1-3 + S1 reconciliation (the highest-error-rate sections in hand-composition)
2. Sections 4-7 + R-rule mechanical checks
3. Section 8 leverage computation + S2/S3/S4 linters
