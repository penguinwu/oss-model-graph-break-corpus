# close-mode rev 4: relaxed criterion + new-issue rebucketing for moved GBs

**Author:** Otter
**Date:** 2026-05-10 21:10 ET (rev 1; REJECTED 21:30 ET by adversary case adv-2026-05-10-211000)
**Status:** REJECTED — needs redesign before implementation. See `subagents/adversary-review/invocations/adv-2026-05-10-211000-close-mode-rev4-design.md` for the 10 gaps + dispositions. Key architecture-level issues: (1) Source-1 body parser dead on real bodies; (2) op-extraction asymmetry vs sweep_compare; (3) parallel signature system instead of reusing sweep_compare.classify_shift; (4) verdict-space conflation with rev 5; (5) add-to-existing proposals require comment-mode that doesn't exist. Awaiting Peng review of the architectural decisions before rev 2 redesign.
**WS1 task:** "close-mode rev 4: relaxed criterion + new-issue rebucketing"
**Related directive:** Peng 2026-05-10 15:27 ET — "If all models reported in an issue no longer graph break at the original reported GB reasons, but hit the next GB. We can close this issue but create new issues for the new GB sites. This part should follow new issue pattern — one needs to check if existing issues already report this problem; if so, add the new models to existing issues; otherwise, open a new issue."

---

## What changes from rev 3 (strict criterion)

Rev 3 close-mode requires ALL originally-affected (model, mode) pairs to be `full_graph` with zero graph_break_count. This is the **strict criterion**: same models, same Dynamo behavior verdict (no breaks at all).

Rev 4 introduces the **relaxed criterion**: an issue is closeable if all originally-affected pairs no longer break at the ORIGINAL reported GB reasons, even if they now hit DIFFERENT GB sites. The moved-GB sites get triaged via the new-issue pattern (search-existing-first, then file).

The strict criterion becomes a special case of the relaxed criterion (zero new GBs at any site).

## Why

Same model, same Dynamo behavior, but transformers source changed → break_reason at a different file:line. Strict close-mode (rev 3) sees "GB count unchanged, status unchanged → no close." But the Dynamo team DID address the original break; the new break is a different bug at a different site.

Concrete scenario from this morning's report — the +4 Granite GB delta:
- Last week: break at `modeling_granitemoehybrid.py:1219`
- This week: break at `:1218` (1-line offset, same code) PLUS new break at `:924` (`.tolist()` call — different bug)

If an issue tracked the line-1219 break and we ran rev-4 close-mode:
- Original break at `:1219` → cleared in current sweep
- New break at `:924` → moved to a different site
- Action: close THIS issue + file new issue for the `.tolist()` break (after dup-search)

Without rev 4, every weekly walk produces "0 closeable" because moved-GB cases are common — defeating the close-mode infrastructure's purpose.

## Mechanism

### Step 0: Per-issue break_reason extraction

For each issue Mode A_close considers, extract the ORIGINAL break_reason:
- Source 1 (preferred): the issue body contains a structured `## Break Reason` block (per the corpus's NEW-issue template). Parse it.
- Source 2 (fallback): from sweep_compare's cat-2 evidence at the time the issue was filed (recorded in the issue body's `## Source` block + sweep ref). Re-derive.
- Source 3 (worst case): if neither extractable, treat as legacy issue requiring human triage; verdict = `reframe`.

Store as: `original_break_signature = (file, line_normalized, op_or_explanation)` where `line_normalized` strips trailing-line-number drift (the `:1219 → :1218` shift) by considering line ranges or NORMALIZING line numbers across known transformers source paths.

### Step 1: Per-pair current break analysis

For each originally-affected pair, look up current sweep's explain row:
- Status `full_graph` → no current breaks
- Status `graph_break` → current break_reasons list

Compute per-pair `current_break_signatures = [(file, line, op), ...]`.

### Step 2: Compare original vs current

For each originally-affected pair:
- **Original site cleared** if NO current break_signature matches the original (file + line within tolerance + op).
- **Original site still present** if any current break_signature matches.
- **Moved GB sites** = current_break_signatures NOT matching original (i.e., NEW break sites).

### Step 3: Verdict + actions

Per-pair classification:
- `cleared-no-new` — original cleared, no new GB sites → contributes to close
- `cleared-with-new` — original cleared, but new GB site(s) appeared → contributes to close + emits new-issue proposal
- `original-persists` — original site still present → BLOCKS close

Whole-issue verdict (extends rev 3 verdict space):
- `close-strict` (existing, rev 3): all pairs `cleared-no-new`. All pairs are full_graph.
- `close-relaxed-with-new-issues` (NEW): all pairs `cleared-no-new` OR `cleared-with-new`; emits a list of new-issue proposals for the moved GB sites.
- `reject-keep-open` (existing, rev 3): any pair `original-persists`.
- Other rev 3 verdicts (`reframe`, `block-stale-rerun`, `not-a-candidate`, `verify-failed` from rev 5) carry over.

### Step 4: New-issue proposal generation (only if `close-relaxed-with-new-issues`)

For each moved GB site (collected across all pairs):
1. Group pairs by `(file, line_normalized, op)` signature
2. For each signature group, run `gh issue list --search` against existing open + closed issues using the file path + op as query terms (per methodology R5)
3. For each signature, emit one of:
   - **`add-to-existing`** — search found a matching issue (signature + affected models). Action: comment on existing issue with the new-pairs list.
   - **`file-new`** — no matching issue. Action: invoke file-issue subagent (Step 0 cluster plan) for a NEW issue.
4. New-issue proposals are SURFACED to the human reviewer in the verify_close JSON's `new_issue_proposals[]` field. They're NOT auto-filed by close-mode itself — file-issue subagent + Peng's cluster-plan-approval gate still applies.

Output JSON structure (extends rev 5's verify_close JSON):
```json
{
  ... // rev 5 fields
  "verdict": "close-relaxed-with-new-issues",
  "verdict_reason": "All 6 pairs cleared original; 4 new GB sites surface 1 new-issue proposal + 1 add-to-existing proposal",
  "moved_gb_analysis": {
    "originally_affected_pairs_classified": [
      {"name": "MusicgenForCausalLM", "mode": "train", "classification": "cleared-with-new",
       "original_signature": ["modeling_musicgen.py", 320, "BUILTIN(lt)"],
       "new_signatures": [["modeling_musicgen.py", 412, ".tolist()"]]},
      ...
    ],
    "new_issue_proposals": [
      {"signature": ["modeling_musicgen.py", 412, ".tolist()"],
       "affected_pairs": [["MusicgenForCausalLM", "train"], ["MusicgenModel", "train"]],
       "action": "file-new",
       "dup_search_results": [],
       "proposed_title": "[dynamo] .tolist() data-dependent break in Musicgen family (train mode)"},
      {"signature": ["modeling_musicgen.py", 600, "data-dep branch"],
       "affected_pairs": [["MusicgenMelodyModel", "train"]],
       "action": "add-to-existing",
       "existing_issue_num": 122,
       "comment_body": "Add MusicgenMelodyModel/train (1 break) to issue tracking ..."},
    ]
  }
}
```

## Operator workflow

1. Run rev-5 verify_close_candidate (with rev-4 logic baked in) → emits JSON
2. Review verdict + new_issue_proposals
3. If verdict == `close-relaxed-with-new-issues`:
   - For each `add-to-existing` proposal: invoke file-issue subagent comment-mode (when shipped) to add models
   - For each `file-new` proposal: invoke file-issue subagent NEW-mode (Step 0 cluster plan + Mode A + Mode B)
4. THEN invoke close-mode `--close` with the verify JSON → posts close + comment
5. file-issue NEW invocations require Peng's cluster-plan approval per existing rules

## Scope of "moved-GB" vs "regression"

A new GB site is moved-GB if it's at a DIFFERENT (file, line, op) signature than the original. If the new GB has the SAME signature as the original (just at a different line within the same file due to source drift), that's NOT moved — it's the original break still present (line-tolerance match). The line-normalization function handles known transformers source-path drift (modellibs vs pip-installed).

## Implementation scope

- `tools/verify_close_candidate.py` (rev 5): EXTEND with break-signature extraction + comparison + new-issue proposal generation. ~200 lines added on top of the rev-5 base.
- `tools/file_issues.py`: extend `_do_close_op` to accept verdict `close-relaxed-with-new-issues` (currently only accepts `all-clear`). Refuse if any new-issue proposals haven't been disposed yet (a `--accept-new-issue-proposals` flag or similar gate).
- `tests/test_verify_close_candidate.py`: ~150 lines added (5 tests covering moved-GB scenarios)
- `subagents/file-issue/persona.md`: small Mode A_close addendum for the new verdict

Estimated 8-12h on top of rev 5. Adversary-review required.

## Test plan

5 new tests for `tools/test_verify_close_candidate.py`:
1. `test_relaxed_criterion_close_with_no_new_breaks` — original cleared, no new GBs → `close-strict` (rev 3 backward compat)
2. `test_relaxed_criterion_close_with_moved_gbs` — original cleared, new GB at different site → `close-relaxed-with-new-issues` + new-issue proposals emitted
3. `test_line_tolerance_handles_source_drift` — original at line 1219, current at line 1218 with same op → MATCHES (no moved GB)
4. `test_dup_search_emits_add_to_existing` — moved GB matches an existing issue → `add-to-existing` proposal with issue_num
5. `test_relaxed_blocks_when_original_persists` — original site still present → `reject-keep-open` even if other sites moved

## Open design questions (surface to Peng if adversary doesn't catch)

1. **Line tolerance threshold:** how much line drift counts as "same site"? Currently proposed: ±10 lines + same op signature. Should this be tunable per file or globally? Per Peng's R3 attribution discipline — strictly, ANY line offset means transformers source changed, so they're NOT byte-identical signatures. But operationally, treating ±10 lines as same-site avoids huge new-issue noise on every transformers patch release.

2. **dup-search granularity:** R5 in methodology.md requires searching existing issues before filing. For the rev 4 new-issue proposals, the dup-search needs to be MORE aggressive than NEW issues (since we're surfacing many proposals at once). Should we use the file path AND the op as separate queries? Currently proposed: combine both terms. May need refinement after first real use.

3. **Auto-file vs surface-only:** the design says "surface to human reviewer; not auto-file." Should there be an `--auto-file-new-issues` mode that invokes file-issue subagent in batch? Or always surface? Currently proposed: always surface (matches the close-mode rev-3 dispositional posture).

## Adversary-review case

`adv-2026-05-10-211000-close-mode-rev4-design` — to be created.
