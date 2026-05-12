# close-mode rev 4 v2: relaxed criterion + cascade-to-file-new

**Author:** Otter
**Date:** 2026-05-11 21:15 ET (rev 2; supersedes `CLOSE_MODE_REV4_DESIGN.md` rev 1 which was REJECTED by adversary 2026-05-10)
**Status:** Architecture resolved — all 4 questions answered by Peng 2026-05-11 20:36-21:02 ET. Ready for adversary review + impl.
**WS1 task:** "close-mode rev 4 v2: relaxed criterion + cascade-to-file-new"

---

## Architecture decisions (Peng 2026-05-11 20:36-21:02 ET)

| # | Question | Decision |
|---|---|---|
| Q1 | Source-of-truth for "shift" classification (own signature system OR reuse `sweep_compare`) | **Reuse `sweep_compare.classify_shift`.** Extend its output schema with a per-issue rollup sidecar. No parallel signature system. |
| Q2 | Routing for "moved-GB" cases (build comment-mode OR file-new-only) | **No comment-mode build.** The corpus's primary GB-issue routings are NEW + EDIT + CLOSE. A moved-GB case = CLOSE the old issue (the GB it tracks IS fixed) + cascade-to-FILE-NEW for any next-GB patterns the originally-affected models now hit. Model-list shifts on the SAME GB pattern = EDIT body's affected list. |
| Q3 | Verdict space (extend rev 3's set OR restructure into tuple) | **Extend** with one new verdict `close-with-cascade` + structured cascade payload (list of next-GB patterns + their proposed file-new framings). Rev 5 verify-JSON layer (downstream) is unchanged. |
| Q4 | Data-missing handling (block / neutral / positive evidence) | **Fetch via per-pair re-run within 7-day nightly window.** Don't block, don't speculate — go get the data. |

---

## Cohort definition reminder (carries from rev 3)

The "compile-testable cohort" = model × mode pairs that are NOT in `known_errors.json` (eager-side bugs we deliberately exclude) AND NOT in `skip_models.json` (intentional skips). When `known_errors` removes an entry (e.g., upstream eager-side bug closes), the pairs re-enter the cohort.

The "originally-affected pairs" of an issue = the set of (model, mode) pairs cited in the issue body's `## Affected Models` table at filing time. close-mode reads this set from the issue body.

---

## What changes from rev 3 (strict criterion)

**Rev 3 close criterion (strict):** all originally-affected pairs are `full_graph` with zero `graph_break_count` in current sweep.

**Rev 4 v2 close criterion (relaxed):** all originally-affected pairs no longer hit THIS issue's tracked GB pattern in current sweep. Pairs may still graph_break — just on DIFFERENT patterns. The next-GB patterns observed cascade into separate file-new actions.

The strict criterion becomes a special case (cascade payload empty: zero next-GB patterns to file).

## Why

Same models, same Dynamo behavior, but the model-side (transformers) or torch-side source moved → original break_reason at a different file:line, OR original break_reason no longer fires but a different one does. Rev 3 sees this as "models still graph_break → no close"; the GB the issue tracks IS fixed, but the issue stays open forever.

Concrete scenario: GraniteMoeHybrid on tx 5.6.2 broke on `_local_scalar_dense` from `.tolist()` at `modeling_granitemoehybrid.py:924` (#55 territory). After today's tx 5.8.0 modellibs upgrade, the same model breaks on `torch.all(attention_mask == 1)` at `modeling_granitemoehybrid.py:1219` (#78 territory). If an issue tracked the `:924` break and rev 4 v2 close-mode ran:
- Originally-affected pairs no longer hit `:924` GB pattern → relaxed criterion satisfied → CLOSE
- New `:1219` GB pattern observed on the same pairs → cascade payload: file-new (or EDIT #78 if existing-issue dup-search hits)

---

## Mechanism

### Step 0 — Per-issue break_reason extraction (unchanged from rev 1)

For each issue Mode A_close evaluates, extract the ORIGINAL tracked break_reason:
- **Source 1 (preferred):** issue body's structured `## Break Reason` block (per the corpus's NEW-issue template). Parse the file:line + reason-text.
- **Source 2 (fallback):** sweep ref + cat-2 evidence at filing time, recorded in body's `## Source` block. Re-derive from the cited sweep's `explain_results.json`.

Output: a normalized `tracked_pattern` tuple — `(file_path, line_or_range, reason_excerpt_normalized)`. The `line_or_range` allows for ±2 line tolerance to absorb minor source-version drift.

### Step 0.5 — Per-pair data-fetch (NEW per Q4)

Pre-flight: for each originally-affected pair, check whether current sweep has observable data:
- **Observable:** sweep contains an identify+explain row for the pair → use it.
- **Missing:** pair is in `skip_models.json`, in `known_errors.json` as eager_error, or wholly absent (e.g., not in current sweep's cohort due to version drift), OR the sweep has `.audit-rerun-required` marker for the pair.

For each MISSING pair, trigger a per-pair re-run on the latest nightly torch (within 7-day venv staleness window):
- **Pre-step:** check `~/envs/torch-nightly-cu126/`'s `torch.version.git_version` against the latest published nightly. If venv is >7 days old, run `tools/run_experiment.py refresh-nightly` first.
- **Re-run command:** `~/envs/torch-nightly-cu126/bin/python sweep/worker.py --model-json '{"name":"<name>","source":"hf"}' --pass-num 1 --device cuda --mode <mode>` (identify pass), then `--pass-num 2` (explain pass) if the identify result is graph_break.
- **Result handling:** the re-run produces an identify_results row + (conditionally) an explain_results row. These rows are the data point for the close criterion evaluation. Re-runs are recorded in the close-mode case file's `data_fetch.re_runs[]` field (provenance).
- **Bypass policies:**
  - For `skip_models.json` entry: re-run with skip-list bypass (the entry was set for cohort-management reasons; close-mode evaluation needs the actual data).
  - For `known_errors.json` eager_error entry: re-run with the entry honored (we don't want to silently bypass known eager-side bugs; the re-run will still produce eager_error which counts as "still missing" for close-criterion purposes — see "Detail nuances for adversary review" below).

### Step 1 — Mode A_close (extended verdict space per Q3)

Mode A_close walks the close criterion against the (now-complete) data and emits one of:

| Verdict | When | Cascade payload? |
|---|---|---|
| `close` | Strict criterion satisfied: all pairs `full_graph`, zero `graph_break_count`. | Empty (no next-GB patterns observed). |
| `close-with-cascade` (NEW per Q3) | Relaxed criterion satisfied: all pairs no longer hit THIS issue's `tracked_pattern`. At least one pair hits a DIFFERENT GB pattern. | Non-empty: list of `next_gb_patterns` (each with `(file_path, line, reason_excerpt)` + affected pairs + dedup_search proposal for existing-issue match). |
| `reject-keep-open` | At least one pair STILL hits this issue's `tracked_pattern`. | N/A. |
| `reframe` | Sweep age > 10 days (data is too stale for any close decision). | N/A. |
| `block-stale-rerun` | `.audit-rerun-required` marker present AND Step 0.5 re-runs failed (e.g., timeout, OOM, infrastructure). | N/A. |
| `not-a-candidate` | Issue # not in plan's `close_candidates`. | N/A. |

The `close-with-cascade` payload schema (proposed):

```yaml
cascade:
  next_gb_patterns:
    - pattern_id: <stable hash of (file:line + reason_excerpt_normalized)>
      file: transformers/models/granitemoehybrid/modeling_granitemoehybrid.py
      line: 1219
      reason_excerpt: "Data-dependent branching ... torch.all(attention_mask == 1)"
      affected_pairs:
        - {name: GraniteMoeHybridForCausalLM, mode: eval, gb_count: 9}
        - {name: GraniteMoeHybridForCausalLM, mode: train, gb_count: 9}
        - {name: GraniteMoeHybridModel, mode: eval, gb_count: 8}
        - {name: GraniteMoeHybridModel, mode: train, gb_count: 8}
      dedup_search:
        query: "torch.all(attention_mask == 1) modeling_granitemoehybrid.py"
        existing_match: 78  # or null if no match
        proposed_routing: edit  # 'edit' if existing_match is non-null; 'file-new' otherwise
```

### Step 2 — `sweep_compare.classify_shift` integration (per Q1)

`sweep_compare.classify_shift` already classifies per-pair GB-pattern shifts between baseline and current. Extension needed: a per-issue rollup sidecar at `<sweep_dir>/issue_close_evaluation.json` that, for each tracked-issue's `(tracked_pattern, originally_affected_pairs)`, summarizes:
- Pairs where `tracked_pattern` STILL fires
- Pairs where `tracked_pattern` no longer fires + what next-GB they hit
- Pairs that are missing data (queue for Step 0.5 re-run)

`tools/sweep_compare.py` gets a new `--issue-close-evaluation <plan-path>` flag that consumes a Step 0 cluster plan + writes the rollup. close-mode Mode A_close reads this rollup as primary input.

### Step 3 — Cascade execution (per Q2)

When Mode A_close emits `close-with-cascade`:
1. Mode B_close emits the closing comment body (rev 3 template, observational, no attribution claim).
2. close-mode posts the close + closing comment via `tools/file_issues.py corpus-issue --close <num>` (rev 3 path).
3. For each `next_gb_pattern` in the cascade payload:
   - If `dedup_search.existing_match` is non-null AND `proposed_routing: edit`: surface the EDIT proposal to Peng (existing tooling: `tools/file_issues.py corpus-issue --edit <existing#>`).
   - If null AND `proposed_routing: file-new`: surface a NEW filing proposal to Peng (existing tooling: `tools/file_issues.py corpus-issue` NEW path with full file-issue subagent walk).
4. Cascade actions are NOT auto-executed; they're surfaced for Peng's per-pattern approval. Each cascade action is its own file-issue case_id chained via `parent_case_id` to the close-mode case_id.

---

## Detail nuances for adversary review (NOT Peng decisions — for design adversary)

These are intentionally surfaced for adversary challenge before impl:

1. **Re-run cost cap (Q4 follow-up):** for an umbrella issue with N originally-affected pairs all missing, what's the upper-bound N before close-mode falls back to `block-stale-rerun` instead of running all of them? Proposed: cap at 20 pairs per close-mode invocation; >20 → fall back to `block-stale-rerun` with a "too many missing pairs" reason. Alternative: time-based cap (e.g., re-runs collectively >30 min wall → fall back).

2. **`known_errors.json` entry on missing pair:** Step 0.5 honors the entry (re-run still produces eager_error). For the close criterion, eager_error counts as "still missing" (does not satisfy "no longer hits THIS GB"). Alternative interpretations: (a) treat eager_error as positive (the model can't reach Dynamo; the GB this issue tracks definitionally isn't firing), or (b) treat it as `reject-keep-open` (we can't verify either way). Proposed: option (b) — conservative.

3. **`skip_models.json` entry on missing pair:** Step 0.5 BYPASSES the entry to get data. Concern: skip entries exist for a reason (often "this model OOMs") — bypassing risks resource exhaustion. Mitigation: per-pair re-run respects the per-model timeout tier; a pair that historically OOMed will OOM again deterministically and surface as `block-stale-rerun`.

4. **Re-run produces a NEW eager_error or new failure mode (not present in original sweep):** treat as "still missing" for close criterion; record in case file's `data_fetch.re_runs[]` for forensic visibility.

5. **Cascade payload's `dedup_search` accuracy:** the proposed search uses `tools/dedup_source_lines.py` (exists since 2026-05-11 commit `c1a7b20`). Tool finds source-line citation overlaps in existing issue bodies. Concern: a transformers source-line shift (line 1219 → 1218 between versions) breaks loose-path matching. Mitigation: loose-path match strips `:NN` and matches just the path; reduces this risk to "right pattern, wrong line" → which `dedup_source_lines.py` already handles.

6. **`tracked_pattern` line-tolerance (±2 lines):** balances source-version drift (small line shifts in unchanged code) vs spurious matches (different code at adjacent line). Risk: a refactor that moves code by >2 lines reads as "no longer firing" when it actually still fires at a new location. Mitigation: combine line-tolerance with reason_excerpt_normalized comparison — both must match for "still hitting" classification.

7. **Stale-venv detection (Q4 sub-detail):** how does close-mode decide the venv is "fresh enough"? Check `python -c 'import torch; print(torch.version.git_version)'` against `torch.version.published_nightlies()` API (or a manifest file). If git_version is >7 days behind latest nightly's git_version, refresh first. Concern: pip's `--pre torch` install may hit BPF jail; recipe at `~/.myclaw-shared/recipes/python-venv-bpf.md`.

---

## File inventory (what ships with rev 4 v2)

```
subagents/file-issue/
├── CLOSE_MODE_REV4_V2_DESIGN.md   # This file (replaces rev 1 design)
├── persona.md                      # MODIFIED: extend Mode A_close verdict space + add Mode B_cascade-narrative
├── invocations/                    # New case_id format: close-rev4-YYYY-MM-DD-...

tools/
├── sweep_compare.py                # MODIFIED: add --issue-close-evaluation flag + per-issue rollup sidecar emit
├── file_issues.py                  # MODIFIED: extend close-mode (Step 7) to emit cascade payload to /tmp/<case_id>-cascade.json + verify cascade payload's dedup_search via dedup_source_lines integration
├── close_mode_re_run.py            # NEW: per-pair re-run helper (wraps sweep/worker.py invocation + 7-day venv staleness check)
├── test_close_mode_re_run.py       # NEW: unit tests
├── test_file_issues.py             # MODIFIED: extend with rev 4 v2 cases (close-with-cascade, cascade payload validation, re-run integration)

sweep/
├── sweep_compare.py is in tools/, not sweep/  # actually correct
```

## Test plan

| Test | Pins |
|---|---|
| `test_relaxed_criterion_satisfied_strict_close` | All pairs full_graph → `close` verdict, empty cascade |
| `test_relaxed_criterion_satisfied_with_cascade` | All pairs no longer hit tracked_pattern but hit different patterns → `close-with-cascade` verdict, populated cascade |
| `test_relaxed_criterion_failed_one_pair_still_hits` | One pair still hits tracked_pattern → `reject-keep-open` |
| `test_data_missing_triggers_re_run` | Pair in skip_models.json → Step 0.5 re-run → result becomes data point |
| `test_data_missing_re_run_eager_error_still_missing` | Re-run produces eager_error → counts as still missing (per nuance #2 (b)) |
| `test_data_missing_re_run_too_many_pairs_blocks` | >20 missing pairs → `block-stale-rerun` (per nuance #1) |
| `test_stale_venv_triggers_refresh` | Venv git_version >7 days old → refresh-nightly invoked first |
| `test_cascade_payload_dedup_search_existing_match` | Next-GB pattern matches existing issue → `proposed_routing: edit` |
| `test_cascade_payload_dedup_search_no_match` | Next-GB pattern is novel → `proposed_routing: file-new` |
| `test_tracked_pattern_line_tolerance_in_range` | Pair hits at line 1218; tracked_pattern line=1219 → "still hitting" (per nuance #6) |
| `test_tracked_pattern_line_tolerance_out_of_range` | Pair hits at line 1280; tracked_pattern line=1219 → "no longer hitting" |

11 tests minimum.

## Sequencing

1. **Adversary review on this design** (~1 session) — the 7 detail nuances above are the primary attack surface.
2. **Address adversary gaps** (~0.5 session).
3. **Implementation:**
   - `tools/close_mode_re_run.py` + tests (~1 session).
   - `sweep_compare.py` per-issue rollup sidecar + integration tests (~0.5 session).
   - close-mode Mode A_close extension + Mode B_cascade narrative + cascade payload schema + tests (~1 session).
4. **Live validation** on 3-5 close candidates from this week's sweep + cascade dry-runs (~0.5 session).

**Total estimate: 3-4 agent sessions.** Lands before the 2026-05-16 weekend sweep with margin for iteration.
