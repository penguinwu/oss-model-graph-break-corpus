---
name: adversary-review
description: Use BEFORE committing changes to validator code (sweep/explain.py) OR additions of new sweep cases (sweep/models.py model entries, new corpus entries). Spawns an independent adversarial reviewer (V1: local Agent; V2: Rocky) to find gaps in the work. Records the reviewer's full output verbatim + every gap's disposition (addressed/deferred/disputed) in reviews_log.md as Peng-in-the-loop evidence. Skipping requires Peng's written approval.
---

# adversary-review

## When to use

**Trigger conditions (any one):**
- About to commit changes to `sweep/explain.py` (validator/scoring logic — semantic correctness)
- About to commit new model entries added to `sweep/models.py` (new sweep cases)
- About to commit new entries to any corpus model registry / catalog file

**Do NOT use for:**
- Sweep harness mechanical changes (those go through `test-sweep-changes` — different bug class)
- Renames, formatting, comment-only changes
- Tooling/analysis scripts (`tools/analyze_*.py`, etc.)

## Why this exists

Tests I write reflect MY mental model of what should be tested. Blind spots in my mental model produce blind spots in tests — and bugs that should be caught at commit-time slip into sweep results, where they cost wall-clock and erode trust in headline numbers.

An independent reviewer with a different framing catches what I miss. V1 uses a local Agent (in-session subagent, fresh context per invocation, blind to my prior reasoning). V2 will swap to a separate stateful peer (Rocky) — but the persona, log, and disposition discipline carry forward unchanged.

**Peng-in-the-loop guardrail:** every invocation produces a `reviews_log.md` row with the reviewer's full output verbatim + a SHA256 hash + the disposition of every gap raised. This makes it impossible to "skip" or "sanitize" the review without leaving a trail.

## The procedure (strict order)

Skipping a step requires Peng's explicit written approval in the conversation.

### Step 1 — Prepare the review request

Gather:
- **change_summary** (1-3 sentences): what you're committing and why
- **files_under_review** (paths): the concrete files the reviewer should read
- **context** (links/notes): any prior plan.md, related issues, design constraints the reviewer needs

Write these into a temporary review-request markdown file at `/tmp/adversary-review-<case_id>.md` where `case_id` is `YYYY-MM-DD-HHMMSS-<short-slug>`.

### Step 2 — Invoke the adversary

Spawn a local Agent with the persona as the system prompt:

```
Use the Agent tool with:
  description: "Adversary review of <case_id>"
  subagent_type: general-purpose
  prompt: <contents of skills/adversary-review/persona.md> + "---" +
          <contents of /tmp/adversary-review-<case_id>.md>
```

The reviewer must produce its output in the structured format defined in `persona.md` (verdict, gaps_found list, suggested_tests, confidence).

### Step 3 — Record the review (verbatim + hash)

Append a row to `skills/adversary-review/reviews_log.md`:

```
### <case_id>

| field | value |
|-------|-------|
| date_utc | <ISO8601> |
| trigger | <validator-code | new-sweep-case | other> |
| files | <files_under_review> |
| persona_sha | <git rev of persona.md at invocation time> |
| verdict | <reviewer's top-level verdict> |
| output_sha256 | <hash of the raw output block below> |

**Reviewer raw output:**
```
<the Agent's full reply, verbatim — no edits>
```

**My disposition:**
- gap 1: <short description> → <addressed | deferred (reason) | disputed (escalation_id)>
- gap 2: ...
```

Compute the SHA256 of the raw output block:

```bash
echo -n "<raw output>" | sha256sum
```

The hash protects against silent post-hoc edits.

### Step 4 — Respond to each gap

For every gap the reviewer raised, choose one disposition:

- **Addressed** — make the fix in the same commit (preferred). Add a follow-up commit only if the fix is large enough to warrant separate review.
- **Deferred** — file an OPEN-LOOP entry with reason ("not in scope of this commit", "blocked on X"). Quote the open-loop ID in the disposition.
- **Disputed** — you believe the gap isn't real. Use `skills/adversary-review/escalation_template.md` to escalate to Peng. Do NOT commit until Peng resolves.

### Step 5 — Commit with evidence

The commit message must include:

```
adversary-review: <case_id>
- reviewer verdict: <verdict>
- gaps: <N>; addressed: <A>, deferred: <D>, disputed: <X>
- log entry: skills/adversary-review/reviews_log.md#<case_id>
```

If verdict is "approve with no gaps," the line still appears (proves the review happened).

## After commit — surface to Peng

Send a one-line note to Peng's space (your own space, `--as-bot`):

```
adversary-review fired for <case_id>: <N> gaps, <A> addressed / <D> deferred / <X> disputed. Log: <link or path>.
```

If any disputes are pending Peng review, prefix with `🚨` and include the escalation summary.

## When skipping the trigger (explicit acknowledgment required)

If you commit a substantial script change that does NOT invoke adversary-review (because it falls outside the trigger conditions above), the commit message MUST include:

```
adversary-review-skipped: <reason>
```

This rule applies to any commit touching files under `sweep/`, `corpus/` (if it exists), or any new top-level Python module in the repo. Pure docs, config, or `tools/analyze_*.py` commits do NOT need the line.

Examples of valid skip reasons:
- `tooling-only — modifies tools/check_X.py, no validator semantics affected`
- `harness-mechanical — covered by test-sweep-changes 5 gates instead`
- `rename only, no semantic change`
- `comment/docstring updates only, no logic touched`
- `skill-artifact — modifies skills/adversary-review/* meta-files, no validator/sweep code`

If you cannot articulate a clean skip reason, that is itself a signal to invoke adversary-review.

**Rationale:** without this rule, "I forgot" is indistinguishable from "I deliberately didn't think it needed it." Forcing the explicit line makes the choice visible in commit history and Peng can challenge skips she disagrees with.

## Iteration cadence

After every 3 reviews, spend 5 minutes on retrospective:

1. Did the reviewer catch real things? (true positives)
2. Did the reviewer flag noise? (false positives — adjust persona to reduce)
3. Did I skip / over-apply the trigger? (calibrate trigger conditions)
4. Update `persona.md` if patterns emerge. Commit the persona diff with rationale.

Track running tallies in a `## Stats` section at the bottom of `reviews_log.md`.

## V2 promotion (Peng's call)

V2 = swap local Agent for Rocky as the reviewer. **Peng calls the V2 timing.** This skill does not auto-promote on any threshold.

When proposing V2 to Peng, include these inputs (not gates — just data for her decision):

- Total invocations in `reviews_log.md`
- True-positive rate (gaps that turned out to be real)
- Persona stability (edits in recent reviews)
- Whether calibration friction has been felt (Otter wishing Peng could tune the reviewer directly without going through Otter)

See `V2_PROMOTION.md` for what changes mechanically when she says go.

## Failure modes

- **"I forgot to invoke"** — caught by reviewing recent commits. If a qualifying commit has no `adversary-review:` line, the trigger was skipped. File a mistake reflection at `~/.myclaw/spaces/AAQANraxXE4/reflections/mistakes/`.
- **"The reviewer was useless"** — single noisy review is not evidence; track over 3-5 reviews. If pattern persists, persona needs work.
- **"I disputed everything"** — high dispute rate means either the persona is wrong or I'm being defensive. Either way, Peng needs to see the pattern. Surface in the next weekly brief.
