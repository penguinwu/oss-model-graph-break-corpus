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
- *Bullets, not paragraphs.* Even narrative content gets broken into 1-line bullets. No 3-sentence prose blocks.
- *Continuous numbering.* Items are numbered `1. 2. 3. ...` across the WHOLE brief, never restarting at 1 per section. Makes "close item 7, defer 9" trivial to write.
- *HANDOFF section is a synthesis, not a dump.* Don't paste the file verbatim — pull out 2-4 bullets of what Peng needs to know to pick up where things left off. Highlight blockers explicitly.
- *Section titles in `*single asterisks*`.* Title is its own line, blank line above + below.
- *Keep titles concrete.* Not "Plans" — say "Active workstream plans". Not "Backlog" — say "Aged Backlog (>7 days)".

### Skeleton

```
[🦦 Otter] Daily brief — <today>

*What shipped since yesterday*

1. <One-line bullet summarizing one thread of work, with issue refs if applicable.>
2. <Next bullet.>
3. <Next bullet.>

(Synthesize from commits + closed issues + closed_loops into bullets. Do NOT enumerate commit hashes — describe the WORK done. Issue numbers are fine — they link. Keep each bullet to one line. Aim for 3-7 bullets total.)

*HANDOFF state*  (only if mtime within 24h)

(2-4 bullets of what Peng needs to pick up where things left off. Pull from handoff.first_lines but synthesize — don't paste raw subsections. Highlight blockers explicitly with the word "Blocked:". Keep each bullet to one line.)

*Active workstream plans*

<continuing numbering from above>. WS<N> — <plan name>: ok / STALE (<N>d since last_check)
<next>. WS<N> — <plan name>: ok / STALE (<N>d since last_check)
...

(If all are ok with recent last_check, write "All N plans current — no stale items." as a single numbered bullet instead of enumerating.)

*Experiment convention drift*  (only if drift > 0)

<continuing>. <slug>: <key problem>
<continuing>. ...

*Aged Backlog (>7 days)*  (only if non-empty)

<continuing>. #<N> (<age>d) — <title>
<continuing>. ...

*Awaiting your input*  (only if non-empty — these are blockers)

<continuing>. <WS> — <task> (<age>d) ← (← marker if started >= 2 days ago)
<continuing>. ...

—
Source: tools/brief_data.py + project board #1
```

### Composition guidance

- *"What shipped since yesterday" is the headline.* Bullets, one line each. Issue numbers OK; commit hashes are noise.
- *Continuous numbering.* If "What shipped" has 5 bullets (1-5), and "Active workstream plans" follows with 3 entries, those are 6, 7, 8. Never restart numbering.
- *HANDOFF is synthesis, not dump.* 2-4 bullets max. Pull blockers explicitly. Peng wants the bottom line, not the whole file.
- *Trim aggressively.* If a section has 10+ items, show top N and "+M more (numbered K through K+M)".

## Step 4 — Render as HTML (rich card)

Brief is posted as a GChat **rich card** (HTML) for better visual hierarchy. Compose the brief as a single HTML document. Skeleton:

```html
<h2>🦦 Otter — Daily brief — YYYY-MM-DD</h2>

<h3>What shipped since yesterday</h3>
<ol>
  <li>Bullet copy here. Issue refs like <a href="https://github.com/penguinwu/oss-model-graph-break-corpus/issues/N">#N</a> become real links.</li>
  <li>Next bullet.</li>
</ol>

<h3>HANDOFF state</h3>
<ul>
  <li>2-4 synthesized bullets (NOT verbatim — Peng wants the bottom line).</li>
  <li><b>Blocked:</b> highlight blockers in bold.</li>
</ul>

<h3>Active workstream plans</h3>
<ol start="N">
  <li>WS1 — Skill Discovery Phase 3: ok (Xd since last_check)</li>
  <li>...</li>
</ol>

<h3>Aged Backlog (>7 days)</h3>
<ol start="N">
  <li><a href="...">#N</a> (Xd) — title</li>
</ol>

<h3>Awaiting your input</h3>
<ol start="N">
  <li>WS1 — task description (Xd) — &larr; (use &larr; entity for arrow)</li>
</ol>

<p><i>Source: tools/brief_data.py + project board #1</i></p>
```

HTML rules:

- *Use `<ol start="N">`* to continue numbering across sections (since `<ol>` always starts at 1 unless `start` is set). Track the running number across sections.
- *Use `<a href="...">#N</a>`* for issue references — produces clickable links.
- *Use `<b>...</b>`* sparingly for emphasis (e.g., "Blocked:", "STALE").
- *Use `<i>...</i>`* for subtle de-emphasis (e.g., footer Source line).
- *Keep section headers as `<h3>`.* The top brief title is `<h2>`. Don't go deeper than `<h3>`.
- *Drop empty sections entirely.* No "(none)" placeholders.
- *Total under 4000 chars.* HTML overhead counts.

## Step 5 — Post via card

Write the HTML to a temp file, then:

```bash
gchat send AAQANraxXE4 --as-bot --quiet --card <tempfile>
```

`--card` requires `--as-bot`. Sends a rich card. The HTML renders as formatted content in GChat (links, lists, bold all work).

**Suppression mode (no activity):** still use `--card` with this minimal HTML:

```html
<h2>🦦 Otter — Daily brief — YYYY-MM-DD</h2>
<p>No new activity since yesterday.</p>
```

## What NOT to do

- *Don't invent data.* If `closed_issues` is empty, don't say "I closed several issues" — there's no record.
- *Don't repeat yesterday's brief.* The suppression rule exists so quiet days don't generate noise.
- *Don't chain to other tools.* You only have Read + Bash. No Edit, no internet, no MCP.
- *Don't pad.* If a section has nothing notable, drop it.
- *Don't add commentary about your own process.* Peng wants the signal, not "I noticed that..."

## Failure modes

- *brief_data.py crashes:* post a one-liner: `[🦦 Otter] Daily brief — <today> failed: <error one line>`. Don't try to recover.
- *gchat send fails:* write the brief to `/tmp/daily_brief_<today>.md` and exit nonzero. Cron log will pick it up.
