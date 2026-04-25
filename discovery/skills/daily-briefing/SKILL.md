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

Pick what's interesting today. You decide order and emphasis. **Never invent.** Fields with empty values are simply omitted from the output — no need to write "(none)" everywhere.

### Visual rules (apply to every section)

- *Blank line between every section.* Sections are visually separated by whitespace, not headers.
- *Number every list item.* Use `1. 2. 3.` so Peng can reference items by number ("close item 3, defer 5").
- *Section titles in `*single asterisks*`.* Title is its own line, blank line above + below.
- *Sub-items / context* under a numbered item: indent with two spaces, no leading marker.
- *Keep titles concrete.* Not "Plans" — say "Active workstream plans". Not "Backlog" — say "Aged Backlog (>7 days)".

### Skeleton

```
[🦦 Otter] Daily brief — <today>

*What shipped since yesterday*

(One narrative paragraph — 1-3 sentences. Synthesize from commits + closed issues + closed_loops. Do NOT enumerate commit hashes; describe the WORK done. Example: "Built the experiment scaffold + drift detector + runner skill arm; closed #64 (Mistral3 relaunch absorbed into #59); closed 3 WS1 pilots from this morning's session." Always cite issue numbers when referring to closures, since those are linkable.)

*HANDOFF state*  (only if mtime within 24h)

(2-3 sentences summarizing what's in flight per HANDOFF.md. Highlight any blockers explicitly.)

*Active workstream plans*

1. WS<N> — <plan name>: ok / STALE (<N>d since last_check)
2. WS<N> — <plan name>: ok / STALE (<N>d since last_check)
...

(If all are ok with recent last_check, write "All N plans current." instead of enumerating.)

*Experiment convention drift*  (only if drift > 0)

1. <slug>: <key problem>
2. ...

*Aged Backlog (>7 days)*  (only if non-empty)

1. #<N> (<age>d) — <title>
2. ...

*Awaiting your input*  (only if non-empty — these are blockers)

1. <WS> — <task> (<age>d) ← (← marker if started >= 2 days ago)
2. ...

—
Source: tools/brief_data.py + project board #1
```

### Composition guidance

- *"What shipped since yesterday" is the headline.* Lead with it. Synthesize, don't enumerate. Issue numbers are fine (they link); commit hashes are noise.
- *"Active workstream plans" replaces the old "Plans"* — the previous label was unclear. Show plan name + workstream tag + staleness.
- *Trim aggressively.* If a section has only 1-2 items, fine; if it has 10+, summarize the top and say "+N more".
- *Numbering is across-section, not global.* Each section starts at 1.

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
