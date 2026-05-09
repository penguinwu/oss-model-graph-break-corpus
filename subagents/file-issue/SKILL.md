---
name: file-issue
description: Use BEFORE filing ANY GitHub issue (corpus repo OR pytorch/pytorch upstream). Spawns a stateless local sub-agent twice — first as ADVERSARY to challenge framing (Mode A), then as ASSEMBLER to write the body (Mode B). Anchored on pytorch/pytorch's contribution standards (see persona.md). Discipline: per-specific-fix scope, validated-live evidence, calibrated terseness. Skipping requires Peng's written approval.
---

# file-issue

> **Phase 1 ship (2026-05-08):** This SKILL + persona.md + MIGRATION.md + the adversary-review migration to `subagents/adversary-review/` (KEEPING `SKILL.md` filename for harness compatibility) + the `--via-skill` enforcement at the CLI level + 2 doc-consistency rules. **Phase 2** (templates, validation-recipes, Mode B's full calibration files) ships AFTER 1–2 real invocations have surfaced what the design actually needs. Per adversary-review case adv-2026-05-08-153427-file-issue-design META observation #1. See `subagents/MIGRATION.md` for the migration plan.

## Pre-requisites (verify before first use)

The skill requires all of the following. Run `subagents/file-issue/PREREQ-CHECK.sh` to verify in one shot:

- **Claude Code's `Agent` tool** available (built-in; verify by running `Agent(...)` once)
- **Local CLAUDE.md trigger** added: `grep -q "subagents/file-issue/SKILL.md" ~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md`
- **`tools/file_issues.py` `--via-skill` flag** implemented as `argparse required=True` on every posting subcommand: `python3 tools/file_issues.py corpus-issue --help | grep -q "via-skill.*required"`
- **GitHub auth** for posting: `gh auth status` exits 0
- **Web proxy** (only for pytorch upstream): `curl -sf -x http://localhost:7824 https://api.github.com/zen` exits 0
- **Validation venvs** documented in `subagents/file-issue/validation-recipes/` (Phase 2)

Missing any prerequisite → the skill stops at Step 0 with a specific error message naming the missing item.

## When to use

**Trigger conditions (any one):**
- About to file a NEW issue in the corpus repo (penguinwu/oss-model-graph-break-corpus)
- About to file a NEW issue in pytorch/pytorch upstream
- About to add a follow-up COMMENT with a repro / numeric evidence to an existing pytorch/pytorch issue

**Do NOT use for:**
- Re-running an existing tool that posts comments without new repro evidence (e.g. weekly sweep updates via `tools/file_issues.py sweep-update`)
- PR comments, code-review threads
- Internal Workplace / GChat posts

## Why this exists

Tests Otter writes reflect Otter's mental model. Same for issues — Otter naturally lumps related-looking failures into umbrella issues, cites numbers from memory, leaves the repro implicit. PyTorch's CONTRIBUTING.md is explicit that low-quality or verbose AI-assisted contributions are rejected (see `persona.md` Mode B preamble for the exact quote). The four criteria Peng established 2026-05-08:

1. **Self-contained** — standalone reproduction instructions, no external artifacts
2. **Concise** — preserve maintainer attention; PyTorch maintainers read 50 issues a day
3. **Trustworthy** — validate symptoms LIVE at filing time, not from memory or prior runs
4. **Actionable (= reproducible)** — the maintainer can reproduce the symptom in their own environment from the body alone. **The body does NOT propose a fix.** The maintainer decides the fix. (Redefined 2026-05-08T20:23 ET — see "What this skill does NOT do" below.)

The sub-agent enforces all four mechanically. Mode A blocks bad framing; Mode B blocks bad bodies.

## What this skill does NOT do

**This skill does NOT help Otter suggest fixes in issues.** Per Peng directive 2026-05-08T20:23 ET, Otter does not have the dynamo-internals expertise to suggest fixes credibly, and Otter-suggested fixes have been refuted by maintainers in the past (Alban refuted a fix on a pytorch/pytorch issue 2026-05-06; Mode A's first invocation 2026-05-08 reinforced the anti-pattern by recommending an existing-issue rewrite include "Pick ONE direction"). The lesson: Otter's expertise is finding + reproducing breaks; the maintainer decides the fix.

Mechanically, this means:

- **Forbidden section headers in any issue body:** `Proposed fix`, `Suggested fix`, `Possible directions`, `Possible fixes`, `Recommendations`, `What you should do`, `How to fix`, `Triage options` (when those options ARE fix proposals)
- **Forbidden inline phrases:** `Consider X`, `Maybe try Y`, `We could Z`, `One approach is`, `The dynamo team should`, `A reasonable fix would be`
- **Mode A** rejects (verdict `reframe`) any draft containing these patterns
- **Mode B** refuses (returns `VALIDATION_FAILED`) any body containing these patterns
- **Single carve-out:** if the draft has a `regression_evidence` field with a specific PR + bisect/measurement evidence, the body MAY name the offending PR. Speculation forbidden — proof required.

This is encoded in `subagents/file-issue/persona.md` (criterion #4 + Mode A check #8 + failure-modes table) and pinned by `tools/test_file_issues.py::test_mode_b_refuses_fix_suggestion_content` and `tools/check_doc_consistency.py::rule_no_fix_suggestions_in_templates`.

## Case ID convention

Case IDs use a **sub-agent prefix** to avoid collision with `adversary-review` invocations:

- `file-YYYY-MM-DD-HHMMSS-<short-slug>` — file-issue invocations
- `adv-YYYY-MM-DD-HHMMSS-<short-slug>` — adversary-review invocations (post-migration)

Example: `file-2026-05-08-191500-wav2vec2-ngb`. The prefix appears in the case_id everywhere — directory paths, log entries, posted-issue footer markers, `--via-skill` lookups.

## The procedure (strict order)

### Step 0 — Pre-requisites check

Run the prereq script (or manual checks above). If anything fails, STOP and surface the missing prerequisite to Peng.

### Step 1 — Otter triage

Write a draft framing to `/tmp/file-issue-<case_id>-draft.md` containing:
- **target_repo**: `penguinwu/oss-model-graph-break-corpus` OR `pytorch/pytorch`
- **issue_type**: `bug` | `feature` | `comment` (last for follow-ups on existing issues)
- **proposed_title**: ≤12 words, specific, names the API/symbol + symptom
- **symptom_one_liner**: what fails, observed where
- **evidence_pointers**: paths to results files, sweep dirs, commit shas — anything Mode B will cite
- **proposed_repro**: pseudo-script or sketch of the MRE (≤30 lines target)
- **repro_strategy**: in one sentence, the CONCRETE command/script the maintainer runs to see the symptom in their own environment. Example: "Run `python repro.py` and observe `RuntimeError: ...`." NOT a fix proposal. (Replaced the prior `single_fix_claim` field 2026-05-08T20:23 ET — see criterion #4 redefinition. If you find yourself typing "fix this by..." or "add support for...", that's a fix-suggestion and Mode A will `reframe`.)
- **regression_evidence** (optional): IF you have proof that a specific recent PR caused the symptom (commit sha + bisect/measurement), put it here. This is the ONLY path that lets the body name a fix-direction. Without this field, ANY fix-suggestion content in the body is a hard reject.
- **dup_search**: GitHub search query you ran + result ("searched github.com/<target>/issues?q=<QUERY> — N results, none matching"). Mode A checks the query contains a specific symbol from `proposed_repro` (model class name OR API symbol).

### Step 2 — Live validation (criterion #3)

Re-execute the repro RIGHT NOW. Capture:
- Actual stdout / measured numbers
- The exact venv used (path)
- Wall-clock timestamp (UTC)
- `python -m torch.utils.collect_env` output (mandatory for pytorch/pytorch issues; recommended for corpus)

Write to `/tmp/file-issue-<case_id>-validation.md`. Mode B refuses to assemble without this file. Mode A refuses any `proceed` if a number in the draft does not literally appear in the validation file.

For symptom types we file repeatedly (numeric divergence, graph-break, eager-error), recipe scripts will live at `subagents/file-issue/validation-recipes/<symptom-type>.md` (Phase 2). For Phase 1, write the validation file by hand.

### Step 3 — Mode A: adversary review (framing)

Spawn local Agent. System prompt = `subagents/file-issue/persona.md` Mode-A block. Context = the draft + validation file from Steps 1–2. Output (machine-parseable):

```
VERDICT: proceed | proceed-with-fixes | reframe | split | reject
GAPS_FOUND: <list, each with severity + concrete fix>
FIXES: <only if proceed-with-fixes — list of small fixes Otter applies in-place before Step 4>
SPLITS: <only if split — list of N candidate issues with one-line frame each>
REJECT_REASON: <only if reject — paragraph referencing at least one of the 4 criteria>
```

Five verdicts (NOT four — adversary case 2026-05-08-153427 gap #2):

- **proceed** → all clear; go to Step 4 directly. Should be RARE on first pass.
- **proceed-with-fixes** → small in-place fixes Otter applies (title rewording, label add, dup-search query refinement) without re-invoking Mode A. Each fix is logged to the case's invocation file before Step 4. **Cap: 3 fixes; if Mode A wants to enumerate >3, that's actually `reframe`.**
- **reframe** → Otter revises draft, re-runs Step 3. Used when the framing has structural defects (vague single_fix_claim, wrong template, missing key evidence).
- **split** → Otter loops Steps 1–3 once per split candidate. Each gets its own case_id (chained `parent_case_id` field). **Precedence: if both `split` and `reframe` could apply, choose `split` (the framing problem is the larger gap).**
- **reject** → drop the candidate; log the reject reason; surface to Peng if symptom is real but not file-able.

### Step 4 — Mode B: assembler

Spawn local Agent (FRESH invocation; Mode A's session does not carry over). System prompt = `subagents/file-issue/persona.md` Mode-B block. Context = the stable framing from Step 3 + the validation evidence + the matching template from `subagents/file-issue/templates/<target>-<type>.md` (Phase 2 — for Phase 1, the persona embeds a minimal default template inline). Output:

```
TITLE: <title text>
LABELS: <comma-separated>
BODY:
<the full markdown body, ending with the footer marker:
<!-- via subagents/file-issue case_id=<case_id> -->>

SELF_CHECK:
- [x] criterion 1 ... explanation
- [x] criterion 2 ... word count: N
- [x] criterion 3 ... cites validation file
- [x] criterion 4 ... single_fix_claim preserved
```

Mode B applies the PDF's checklists mechanically (MRE Part 5 / calibration Part 7 / validation Part 8). The PII rules (no `/home/<user>/`, no `/usr/local/fbcode/`, no `[a-z]+@meta.com`, no employee unixnames from USER.md roster) apply to BOTH corpus and pytorch targets — the corpus repo is public OSS per shared CLAUDE.md "Visibility Control - External Repos" rule.

### Step 4.5 — Mode B failure-path dispositions

Mode B may return one of four failure markers instead of a body. Otter handles each:

| Marker | Disposition |
|---|---|
| `OVERSCOPE` | Symptom paragraph >4 sentences after one self-revision. Loop to Step 1 — likely a `split` candidate. Log row: `posted_url=blocked at Mode B (OVERSCOPE)`. |
| `MRE_TOO_LARGE` | MRE >30 lines after one self-revision. Loop to Step 2 — reduce shape, re-validate with smaller inputs. Log row: `posted_url=blocked at Mode B (MRE_TOO_LARGE)`. |
| `VALIDATION_FAILED` | Pre-submission gate flagged ≥1 item (PII, missing env block, wrong template, etc.). Surface failure list to Peng; either fix the framing AND re-validate (Step 2) OR drop. Log row: `posted_url=blocked at Mode B (VALIDATION_FAILED)`. **Otter is FORBIDDEN from "softening" the calibration in the prompt to bypass this** — that defeats the gate. |
| `MODE_NOT_SPECIFIED` | Invocation prompt malformed. Bug in the calling code. Log row: `posted_url=harness error`; fix the prompt and re-invoke. |

Self-revision protocol (aligned with persona.md): Mode B attempts ONE self-revision before emitting a failure marker. The persona's "attempt one self-revision; on second failure, output `<MARKER>`" is the source of truth.

### Step 5 — Authority gate

| Target | Authority | Posting command |
|---|---|---|
| **Corpus repo** (bug or feature) | Otter posts | `python3 tools/file_issues.py corpus-issue --via-skill <case_id> --body /tmp/file-issue-<case_id>-body.md`, then one-line FYI to Peng's space |
| **pytorch/pytorch** (any) | **Peng must approve** | "🚨 EXTERNAL ENGAGEMENT PROPOSED" block in Peng's space with Mode B's verbatim body; wait for token (`approved`/`go`/`(a)`); flip `PYTORCH_UPSTREAM_POSTING_ENABLED = True` in `tools/file_issues.py`; THEN `python3 tools/file_issues.py pytorch-upstream --via-skill <case_id> --post`; flip the constant back to `False` and commit. |
| **Comment on existing pytorch/pytorch issue** | **Peng must approve** | Same proposal block + same flip-constant flow + `python3 tools/file_issues.py pytorch-upstream --via-skill <case_id> --comment <issue#> --post` |

**Code-level guard on pytorch-upstream posting** (per Peng directive 2026-05-08 19:56 ET): the constant `PYTORCH_UPSTREAM_POSTING_ENABLED` in `tools/file_issues.py` defaults to `False`. When `False`, `--post` exits non-zero with an explanation. The ONLY way to actually post is a deliberate source-code edit (constant → `True`) plus the External Engagement Approval token. Test `test_pytorch_upstream_posting_disabled_by_default` asserts the committed default stays `False`. This is mechanical defense-in-depth: a future Otter that forgets the External Engagement rule still cannot post by accident.

**`--via-skill` is REQUIRED on every posting subcommand**, including `corpus-issue` (adversary-review case 2026-05-08-153427 gap #4). Implemented as argparse `required=True` at the CLI level — the tool refuses to even start without it. No tier of light enforcement.

### Step 6 — Log (per-file-per-case schema)

Each invocation writes to its OWN file (eliminates concurrent-append corruption — adversary case 2026-05-08-153427 gap #9):

```
subagents/file-issue/invocations/file-2026-05-08-191500-wav2vec2-ngb.md
```

The file's required schema:

```markdown
---
case_id: file-2026-05-08-191500-wav2vec2-ngb
parent_case_id: <if split, the parent's case_id; otherwise null>
date_utc: 2026-05-08T23:15:00Z
target_repo: penguinwu/oss-model-graph-break-corpus
issue_type: bug
persona_sha: <git rev of subagents/file-issue/persona.md at invocation>
draft_path: /tmp/file-issue-<case_id>-draft.md
draft_sha256: <hash>
validation_file_path: /tmp/file-issue-<case_id>-validation.md
validation_sha256: <hash>
mode_a_verdict: proceed | proceed-with-fixes | reframe | split | reject
mode_a_sha256: <hash of Mode A raw output>
mode_a_fixes_applied: <if proceed-with-fixes, multi-line list of fixes Otter applied in-place; one per line, prefixed with "- ". Empty for other verdicts. Per adversary impl-review gap #8.>
mode_b_sha256: <hash of Mode B raw output, blank if Mode A blocked>
body_sha256: <hash of JUST the BODY: section that gets posted; this is what --via-skill compares>
footer_marker: "<!-- via subagents/file-issue case_id=<case_id> -->"
posted_url: <github URL | blocked at Mode A | blocked at Mode B (REASON) | rejected by Peng | pending>
---

## Mode A raw output

\`\`\`
<verbatim>
\`\`\`

## Mode B raw output (if reached)

\`\`\`
<verbatim>
\`\`\`

## Disposition notes

<freeform — Otter's notes on what happened, esp. for non-`proceed` paths>
```

Companion file `subagents/file-issue/invocations_log.md` is a GENERATED index (one row per case file, with the YAML header fields). Built by `tools/build_invocations_log.py subagents/file-issue/` (Phase 1 ships the script). The aggregator runs nightly via cron + on demand. Reading the log is the audit entry point; reading individual case files is the deep-dive.

`tools/file_issues.py --via-skill <case_id>` reads the matching case file directly (NOT the aggregated log), so no concurrency hazard at posting time.

### Step 7 — Iteration cadence (per Peng req #4)

After every **3 issues filed** (per sub-agent), spend 5 minutes on a retrospective:
- Walk the last 3 case files; what gaps did Mode A surface most often?
- Which calibration items did Mode B fail most often?
- Which template field needs adding?

Append a row to `subagents/file-issue/RETROSPECTIVE.md` with date + observations + persona/template revisions proposed. The forcing function: every 4th `Step 6` invocation must include "retrospective check: <date of last entry>" in the disposition-notes — if >3 issues since last retrospective, the next Step 5 (post) is BLOCKED until the retrospective is logged.

(This same cadence retroactively applies to `subagents/adversary-review/` post-migration. Per adversary case 2026-05-08-153427 NOTE #3.)

## What enforces this

Three layers:

1. **File artifacts** — `/tmp/file-issue-<case_id>-{draft,validation}.md` + `subagents/file-issue/invocations/<case_id>.md` + posted issue footer marker. Five-hop audit chain.

2. **Mechanical refusal at the CLI** — `tools/file_issues.py corpus-issue --via-skill <case_id>` and `pytorch-upstream --via-skill <case_id>` enforce `required=True` at argparse level. The tool exits non-zero before doing anything if `--via-skill` is missing. The check then validates: case file exists, `mode_a_verdict in {proceed, proceed-with-fixes}`, `mode_b_sha256` non-empty, `body_sha256` matches the actual body being posted. **Tested via `tools/test_file_issues.py::test_via_skill_required` and `test_body_sha256_round_trip` (Phase 1 ships both).**

3. **Sampling** — Peng can:
   - `cat subagents/file-issue/invocations/<case_id>.md` for any case to see verbatim sub-agent output
   - Hash any posted issue's body and compare against `body_sha256` in the case file
   - Run a nightly `tools/audit_issue_footers.py` script that scans GitHub for corpus-repo issues lacking the `<!-- via subagents/file-issue case_id=... -->` footer marker and surfaces orphans in the daily brief (Phase 2)

Plus discipline: local CLAUDE.md trigger (added in Phase 1 commit). "Before filing ANY issue → invoke `subagents/file-issue/SKILL.md`. Skipping requires Peng's written approval."

## Phase 1 file inventory (what ships now)

```
subagents/
├── README.md                                # Convention + audit chain explainer
├── MIGRATION.md                             # Migration plan from prior skills/ location to here
├── _common/                                 # Empty (start)
├── adversary-review/                        # MIGRATED from prior skills/ location (see MIGRATION.md)
│   ├── SKILL.md                             # KEEPING SKILL.md filename (harness convention; case 2026-05-08-153427 gap #12)
│   ├── persona.md                           # Unchanged from pre-migration content
│   ├── escalation_template.md
│   ├── invocations/                         # NEW per-file-per-case dir
│   │   ├── adv-2026-05-07-093400-smoke.md          # Migrated entry 1
│   │   ├── adv-2026-05-07-124100-cohort-regen-fix.md  # Migrated entry 2
│   │   ├── adv-2026-05-07-190947-doc-vs-impl.md       # Migrated entry 3
│   │   └── adv-2026-05-08-153427-file-issue-design.md # This review
│   ├── invocations_log.md                   # Generated aggregate (replaces old reviews_log.md)
│   ├── V2_PROMOTION.md
│   └── RETROSPECTIVE.md                     # NEW — first iteration cadence
└── file-issue/
    ├── SKILL.md                             # This file
    ├── persona.md                           # Mode A + Mode B
    ├── PREREQ-CHECK.sh                      # Smoke-test all dependencies
    ├── invocations/                         # Empty until first invocation
    └── invocations_log.md                   # Empty header (generated index)

tools/
├── file_issues.py                           # Modified: --via-skill required=True on all post subcommands
├── build_invocations_log.py                 # NEW — aggregator
├── test_file_issues.py                      # NEW — pins --via-skill enforcement + body_sha256 round-trip
└── check_doc_consistency.py                 # Extended with subagent_required_fields + subagent_paths_migrated rules

design/
└── sub-agent-architecture.md                # NEW — design rationale (3-page version)

docs/
└── sub-agents.md                            # NEW — user-facing 1-pager: what they are, how to read a case file

~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md       # Updated trigger paths to point at subagents/adversary-review/
                                             # Added trigger: subagents/file-issue/SKILL.md
```

**NOT shipped in Phase 1** (deferred to Phase 2 after 1–2 real invocations):
- `subagents/file-issue/templates/` — issue body templates (5 files: corpus-bug, corpus-feature, pytorch-bug, pytorch-pt2-bug, pytorch-feature). Phase 1 uses inline minimal defaults from persona.md.
- `subagents/file-issue/{mre,calibration,validation}-checklist.md` — Phase 1 inlines the checklist content into persona.md directly.
- `subagents/file-issue/validation-recipes/` — Phase 1 writes validation files by hand per case.
- `tools/audit_issue_footers.py` — orphan-marker scanner.

This staged approach matches Peng req #4 (iterate via use), per adversary case 2026-05-08-153427 META observation #1.
