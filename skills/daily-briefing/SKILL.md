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
- `filed_issue_activity` — `{available, total_count, new_activity_count, issues}` from `tools/check_filed_issues.py --changes-only --no-update`. Tracks NEW activity (comments, PR merged, state change) on issues we own across primary repos (corpus, pt2-skill-discovery) + Otter-filed external issues (pytorch/pytorch, transformers since 2026-03-01)
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

The brief has TWO major buckets, in this order:
1. *🚨 Needs your attention* — top of brief, things requiring Peng's input. **This is the most important section.** If only one thing in the whole brief, this is what Peng should look at first.
2. *📦 Corpus project* — graph break corpus work (sweep, AutoDev, GB analysis, models.py, sweep harness)

Cross-cutting items (HANDOFF state, infra hooks, MyClaw work) go in a 3rd *🛠 Cross-cutting* section ONLY if substantive — otherwise drop entirely.

**Skill-discovery work is briefed by a separate skill** at `~/projects/pt2-skill-discovery/skills/daily-briefing/SKILL.md` (its own cron, its own scope). Do NOT include pt2-skill-discovery items here — they belong in that brief, not this one. (Migration completed 2026-05-02.)

```
[🦦 Otter] Daily brief — <today>

🚨 *Needs your attention*  (only if non-empty — top of brief)

1. <emoji> <project tag> <task one-liner> (<age>d) ← (← if >= 2d old)
2. 💬 <project tag> <repo>#<N> — <event signal e.g. "+2 comments since last check" or "closed by maintainer"> — <title (<60 chars)>
3. ...

(Use these emoji prefixes for quick-glance triage:
  🔴 = blocking (delays downstream work)
  🟠 = aging (>=4d, getting stale)
  🟡 = recent (<4d, soft attention)
  💬 = NEW activity on a tracked filed-issue (comment, state change, PR merged) — sourced from `filed_issue_activity.issues` where new_activity=True
Tag each item with 📦 (corpus) or 🛠 (cross-cutting).
Keep each item to one line. If 0 items in EITHER source, drop the section entirely.)

**Sources for this section:**
- `open_loops.needs_input` — items waiting on Peng (use 🔴/🟠/🟡 by age)
- `filed_issue_activity.issues` (where `new_activity=True`) — issues we filed/own that got new activity since last check; use 💬 prefix + the `signal` field. Each issue's `html_url` becomes a clickable link in the HTML render. **Always include these — Peng wants visibility into responses on filed issues.**

📦 *Corpus project — what shipped*

<continuing>. <emoji> <one-line bullet for one corpus thread, with issue refs if applicable>
<continuing>. ...

(Use emoji to convey signal type:
  ⚡ = code shipped
  📊 = sweep result
  ✅ = issue closed / fix landed
  🐛 = regression discovered
  📝 = doc / report shipped
ONLY include items scoped to: sweep, AutoDev, GB analysis, models.py, sweep harness, corpus tools/. Drop infra/skill-discovery items.)

📦 *Corpus project — open issues / loops*  (only if substantive)

<continuing>. <emoji> #N (Xd) — title
<continuing>. ...

🛠 *Cross-cutting (infra, hooks, MyClaw)*  (only if substantive)

<continuing>. <emoji> ...

📋 *Active workstream plans*

<continuing>. <emoji> WS<N> — <plan name>: 🟢 ok / 🟠 STALE (<N>d since last_check)

(🟢 = current, 🟠 = stale. Tag with 📦 — all corpus-scoped plans.)

—
Source: tools/brief_data.py + project board #1
```

### Composition guidance

- *🚨 "Needs your attention" is the headline section.* Top of brief. Order items by 🔴 → 🟠 → 🟡. If empty, drop entirely (no "(no items)" placeholder).
- *Corpus-only scope.* Drop any item scoped to pt2-skill-discovery (cases, skill catalog, harness, discovery experiments) — those go to the pt2-skill-discovery brief instead. Cross-cutting infra goes in the 🛠 section.
- *Continuous numbering across all sections.* Makes "close item 7" trivial.
- *Trim aggressively.* If a section has 10+ items, show top N and "+M more (numbered K through K+M)".

## Step 4 — Render as HTML (rich card)

Brief is posted as a GChat **rich card** (HTML) for better visual hierarchy. Compose the brief as a single HTML document. Skeleton:

```html
<h2>🦦 Otter — Daily brief — YYYY-MM-DD</h2>

<h3>🚨 Needs your attention</h3>
<ol>
  <li>🔴 📦 WS1 — task description (Xd) &larr;</li>
  <li>🟡 🛠 cross-cutting item (Xd)</li>
</ol>

<h3>📦 Corpus project — what shipped</h3>
<ol start="N">
  <li>⚡ Bullet about a corpus commit. Issue refs like <a href="https://github.com/penguinwu/oss-model-graph-break-corpus/issues/N">#N</a> become real links.</li>
  <li>📊 Sweep result bullet.</li>
  <li>🐛 Regression bullet.</li>
</ol>

<h3>📦 Corpus project — open issues</h3>
<ol start="N">
  <li>🟠 <a href="...">#N</a> (Xd) — title</li>
</ol>

<h3>🛠 Cross-cutting</h3>
<ol start="N">
  <li>HANDOFF / infra / hook items (only if substantive).</li>
</ol>

<h3>📋 Active workstream plans</h3>
<ol start="N">
  <li>🟢 📦 WS1 — plan name: ok (Xd since last_check)</li>
  <li>🟠 📦 WS2 — XYZ plan: STALE (Xd since last_check)</li>
</ol>

<p><i>Source: tools/brief_data.py + project board #1</i></p>
```

**Section ordering rule:** 🚨 always first, then 📦, then 🛠, then 📋. The 🚨 section is the most-important-thing — Peng might only have 5 seconds to glance, and 🚨 is what should fit in those 5 seconds.

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
