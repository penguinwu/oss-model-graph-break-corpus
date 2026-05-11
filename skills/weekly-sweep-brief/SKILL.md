---
name: weekly-sweep-brief
description: Compose the weekly nightly-sweep brief for the PT2 dynamo team / Compile Q&A audience. Use when generating a sweep comparison report intended for external broadcast. Walks through data collection (sweep_compare.py), validation gates (attribution must be verified; segmentation must be enforced), template (8 fixed sections), and posting workflow (Workplace group + signal boost via post_to_feedback.py). Trigger phrases: "weekly sweep brief", "nightly sweep summary for the team", "post the sweep report".
---

# Weekly Sweep Brief — workflow + template

## Audience & scope

- **Audience:** PT2 dynamo team, model owners, skill maintainers (technical readers)
- **Scope:** Dynamo-relevant changes only. Eager-side timeouts, harness gaps, and create-error infra are tracked separately and excluded.
- **Cadence:** Weekly. Post to Workplace group `1251481819891072` (**PyTorch Compiler AI-native Working Group** — always verify name via `meta workplace.group details --group-id=1251481819891072` before referring by name; do NOT trust memory). Signal-boost to user group via `post_to_feedback.py`.

## Workflow (run in order, no shortcuts)

### Step 1 — Gather data (single source of truth)

```bash
python3 tools/sweep_compare.py \
    --baseline sweep_results/nightly/<prior-week-date> \
    --current  sweep_results/nightly/<this-week-date> \
    --out experiments/<this-week-date>-nightly-report.md \
    --json experiments/<this-week-date>-nightly-report.json \
    --verbose
```

If invariant check fails (exit 2 = explain coverage gap): run `tools/amend_sweep.py` to backfill explain on amended `graph_break` rows. NEVER write the brief from a sweep with failing invariants — the numbers will be wrong.

### Step 2 — Verify attribution before any "Dynamo improvement" claim

If the report shows N models flipping `graph_break → full_graph`, **test attribution on at least 2 of them**:

```bash
# Re-run a flipped model with current harness + OLDER torch + baseline transformers:
$OLDER_TORCH_VENV/bin/python3 sweep/worker.py \
    --model-json '{"name":"<flipped-model>","source":"hf"}' \
    --pass-num 1 --device cuda --mode eval
```

Outcomes:
- Reproduces the original break → torch attribution confirmed (Dynamo win).
- Compiles cleanly → attribution is harness or transformers, NOT torch — investigate before claiming a Dynamo win.

Verify the call site code is byte-identical between baseline and current transformers (`diff` the relevant files). If the code changed, attribution is mixed.

### Step 3 — Segment all per-pattern deltas (NEVER aggregate across categories)

For any break-reason pattern you want to discuss in the brief:

```bash
python3 tools/sweep_compare.py --baseline X --current Y \
    --pattern "aten._local_scalar_dense.default"
```

The output gives `cat3_delta` (apple-to-apple regression/improvement), `cat1_current` (newly compile-testable exposure), `cat4_current` (truly new model exposure). **Only `cat3_delta` means "regression" or "improvement."** `cat1` and `cat4` are EXPOSURE — patterns that became newly observable, not Dynamo getting worse.

A single corpus-wide scalar like "+41 net" mixes apple-to-apple with exposure and is meaningless. The tool refuses to print it; the brief must too.

### Step 4 — Apply umbrella-split policy when proposing/discussing issues

Before referring to any tracked issue:
1. Check if it's an umbrella (multiple distinct root causes bundled). If so, split first per the corpus CLAUDE.md "Umbrella-split policy" section.
2. Search existing issues for the break-reason substring BEFORE filing new ones (per "Search existing issues BEFORE filing" rule). Filed-then-closed-as-duplicate creates noise.

### Step 5 — Compose brief from template

Use `template.md` (in this skill dir). 8 fixed sections, all required. See `methodology.md` for what each section must contain + forbidden patterns.

### Step 5.5 — Pre-publish gates (Peng directive 2026-05-11 13:05 ET)

Before Step 6 (self-check), walk this gate. ANY failure blocks publication.

**Gate A — Zero pending follow-ups.** Grep the composed brief for "pending follow-up", "TODO", "TBD", "needs follow-up", "needs investigation", "to be filed", "will file". If ANY match: STOP. Either execute the implied action (file the issue, add the comment, run the check) and update the brief with the resulting #/result, OR explicitly remove the speculative wording. The brief is for COMPLETED actions. A user reading the brief should not be left with action items waiting on us.

**Gate B — Tracking-issue citation for known_errors changes.** For each `known_errors.json` entry added or removed since baseline (surfaced in §6), verify the body cites the pytorch/pytorch tracking issue # by URL. Then run `python3 tools/check_known_errors.py --check-tracking-status` to verify each cited tracking issue's CURRENT upstream state (closed/open). If a tracking issue's current state contradicts the brief's framing (e.g., brief says "fix landed and we removed entry" but the tracking issue is still open), fix the brief OR re-investigate.

**Gate C — Definitive answer per new break-reason in §7.** Each break-reason listed in §7 must have one of: existing-issue # / newly-filed-this-sweep # / executed-TODO-with-#-now-cited. If any break-reason lacks a definitive answer, either execute the lookup/filing now and add the #, or remove the break-reason from §7 (it's not new-this-week) — do not ship with "covered by ?" wording.

**Gate D — No internal corpus-side glitches in user-facing prose.** Scan §5 + §8 for mentions of corpus-side process corrections (e.g., "we accidentally closed via a buggy script", "reverted via revert script", "close-mode rev N caught this"). Such mentions belong in PLAN.md, not in the brief — Dynamo team isn't the audience for our internal infra hygiene. If found, delete or move to PLAN.md.

If all four gates pass, proceed to Step 6.

### Step 6 — Self-check before posting

Walk through `methodology.md` § "Self-check checklist" — every item must pass. If anything fails, fix before Step 7. Do NOT post a brief that fails the checklist.

### Step 7 — Post

```bash
# 7a: Workplace post
meta workplace.post create \
    --message="$(cat /tmp/<this-week>-nightly-brief.md)" \
    --group-id=1251481819891072 \
    --formatting=MARKDOWN \
    --title="PT2 Nightly Sweep Brief — <date>"

# Capture the returned permalink URL.

# 7b: Verify the destination group name BEFORE composing the signal boost
#     (do NOT trust memory — USER.md lists similar-sounding groups with different IDs)
GROUP_NAME=$(meta workplace.group details --group-id=1251481819891072 2>&1 | grep "^  name:" | sed 's/  name: //')
echo "Posting to group: $GROUP_NAME"

# 7c: Signal boost via gated wrapper (NEVER raw `gchat send` — see local CLAUDE.md § Feedback space)
python3 ~/.myclaw/spaces/AAQANraxXE4/tools/post_to_feedback.py send \
    "[🦦 Otter]: Weekly nightly-sweep brief for <date> is now posted to the $GROUP_NAME: <permalink>

TL;DR: <2-3 line headline>"
```

## Files in this skill

- `SKILL.md` (this file) — workflow
- `template.md` — the 8-section structure
- `methodology.md` — what each section requires, forbidden patterns, self-check checklist

## When NOT to use

- Daily standups → use `tools/daily_summary.py` + `daily-briefing` skill instead.
- One-off graph-break investigations → write directly, no template needed.
- Internal team sync (own GChat space) → no template needed.
