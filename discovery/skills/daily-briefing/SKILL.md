---
name: daily-briefing
disable-model-invocation: true
description: Compose Otter's morning brief for the OSS Model Graph Break Corpus project from structured signals, and post to GChat. Adapts content to what's interesting today; suppresses to a single line if literally nothing changed.

---

# daily-briefing

## What this skill does

Compose Peng's morning brief for the OSS Model Graph Break Corpus project. Run a data-gathering script, read the JSON, write a brief that highlights what's interesting today, and post it to her GChat space.

**Composition is yours; data is fixed.** Never invent a commit, an issue number, a closure, a plan name, or a metric. Use only fields the data script emitted.

## When this skill is invoked

You'll be invoked from cron at 7:30 AM ET daily (system cron in PT timezone). The prompt will be a one-liner: "Compose today's brief and post per the daily-briefing skill." That's it. Do everything yourself.

## Step 1 — Gather the signals

Run the data emitter:

```bash
python3 /home/pengwu/projects/oss-model-graph-break-corpus/tools/brief_data.py
```

Parse the JSON. Top-level keys:

- `today`, `yesterday`, `since_iso` — date scaffolding
- `commits` — list of `{sha, msg}` from the corpus repo since yesterday 00:00 local
- `closed_issues` — issues closed on the corpus repo in the last 24h (excludes PRs)
- `closed_loops` — bullets from `OPEN-LOOPS.md` Recently-Closed sections matching yesterday or today
- `plans` — discovery plan files + staleness flags
- `experiments` — discovery-experiment convention drift
- `backlog_aged` — board cards with `**Queued at:**` older than 7 days
- `open_loops` — section counts + `needs_input` items awaiting Peng
- `handoff` — current state snapshot from `~/.myclaw/spaces/AAQANraxXE4/HANDOFF.md`

## Step 2 — Decide whether to suppress

**If all four of these are true, emit only the suppression line and stop:**

- `commits == []`
- `closed_issues == []`
- `closed_loops == []`
- `handoff.exists` is False OR `handoff.mtime` is older than 24h from `today`

Suppression line:

```
[🦦 Otter] Daily brief — <today>

No new activity since yesterday.
```

Post that and exit. Do NOT include plans / open-loops / Backlog sections in suppression mode — Peng explicitly asked not to repeat the same state on quiet days.

## Step 3 — Compose the full brief (if not suppressing)

Pick what's interesting today. You decide order and emphasis. **Never invent.** Fields with empty values can simply be omitted from the output — no need to write "(none)" everywhere.

Suggested skeleton (rearrange / drop sections based on what's notable):

```
[🦦 Otter] Daily brief — <today>

*Closed since yesterday*

(commits — show up to ~10 by hash + first line; if more, "...and N more")
(closed issues — `#N — title`)
(closed_loops — first ~80 chars per bullet, drop `**bold**` markers)

*HANDOFF*  (only if mtime within 24h)

(first ~15 lines of handoff.first_lines, trim aggressively if long; or just 2-3 sentences summarizing what's in flight)

*Plans*  (only if any STALE; if all green, just say "all green")

(plan name + age days; STALE marker if stale)

*Experiment convention drift*  (only if drift > 0)

(per-drift item with key problems)

*Backlog aged*  (only if non-empty)

(`#N (Nd) — title`, top 5)

*Awaiting your input*  (only if non-empty)

(`<section> — <task> (Nd ago)` with ← if started >= 2 days ago)

—
Source: tools/brief_data.py + check_plan.py + check_experiments.py + OPEN-LOOPS.md + project board #1
```

## Step 4 — Format for GChat

GChat formatting rules — non-negotiable:

- *No markdown tables* (don't render on mobile). Use bullets or plain lines.
- *No `**bold**`.* Only `*single-asterisk*` italics render.
- *Blank lines* are the separator between sections, not headers like `###`.
- *Section titles* use `*single asterisks*`.
- *Total length* must be under 4000 chars. If you're over, trim sections (drop low-signal ones first).

## Step 5 — Post

Write the composed brief to a temp file, then:

```bash
gchat send AAQANraxXE4 --as-bot --quiet --text-file <tempfile>
```

Use `--text-file` (not inline `gchat send "..."`) — long inline messages hang on a stdout PIPE deadlock.

## What NOT to do

- *Don't invent data.* If `closed_issues` is empty, don't say "I closed several issues" — there's no record.
- *Don't repeat yesterday's brief.* The suppression rule exists so quiet days don't generate noise.
- *Don't chain to other tools.* You only have Read + Bash. No Edit, no internet, no MCP.
- *Don't pad.* If a section has nothing notable, drop it.
- *Don't add commentary about your own process.* Peng wants the signal, not "I noticed that..."

## Failure modes

- *brief_data.py crashes:* post a one-liner: `[🦦 Otter] Daily brief — <today> failed: <error one line>`. Don't try to recover.
- *gchat send fails:* write the brief to `/tmp/daily_brief_<today>.md` and exit nonzero. Cron log will pick it up.
