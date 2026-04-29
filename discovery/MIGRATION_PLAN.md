# Migration Plan: discovery → pt2-skill-discovery

**Status:** plan signed off by Peng 2026-04-27 21:06 ET. Augmented with review findings 2026-04-29. Execute next.
**Revision:** 2 (see revision log)
**Owner:** Otter
**New repo name:** `pt2-skill-discovery`
**Strategy:** Option A (PYTHONPATH-style imports if/when needed; Option B = pip install -e deferred to future)
**Bi-project Otter** (one agent owns both repos with explicit context-switch markers)
**No git history preservation** (project is days old; clean break is fine)

---

## Why split

The corpus repo and the skill-discovery work serve different audiences and have different cadence/discipline:
- Corpus = team's daily issue-tracking tool (sweep, AutoDev, nightly cron)
- Discovery = bespoke research (depth-per-case investigations)

When mixed in one repo, the rules optimized for one leak into the other. The mix-up risk is real (see V8 case body's stale "Inductor noise floor" reference, retracted in design.md but not propagated to the case prompts because they live in different mental boxes that share a dir).

## Key insight that simplified the plan

*Discovery has zero runtime Python dependencies on corpus.*

Discovery's case files hardcode paths like `VITS_SRC = Path("/home/pengwu/envs/.../modeling_vits.py")` — those are the ONLY references to "where things live." Corpus doesn't expose any code that discovery imports. The `sweep.explain.run_graph_break_analysis` function we use is ~75 LOC and self-contained; we duplicate it into the new repo.

The "discovery uses corpus as input" relationship is *conceptual / data-level*: corpus's tracked-models list informs which cases discovery should author. That's a curation dependency between humans, not a code dependency between repos.

## New repo layout (per Peng 2026-04-27 21:06 ET)

```
pt2-skill-discovery/
├── scripts/                          # all .py files (Python package)
│   ├── __init__.py
│   ├── runner.py                     # CaseSpec, WatchedFile, restore/diff helpers
│   ├── run_config.py                 # single-config end-to-end (parallel runner Stage 1)
│   ├── launch_parallel.py            # parallel orchestrator with lifecycle gate
│   ├── merge_results.py              # post-hoc cross-config aggregator
│   ├── revalidate.py                 # post-hoc revalidation tool
│   ├── validate_runner.py            # canonical check primitive
│   ├── perf.py                       # measure_perf primitive
│   ├── _measure_case.py              # per-case perf subprocess wrapper
│   ├── variants.py                   # V0..V8 catalog + compose_prompt
│   ├── _lifecycle_gate.py            # mandatory pre-launch gate
│   ├── clean_sandboxes.py            # housekeeping
│   ├── smoke_test.py                 # mandatory pre-launch smoke
│   └── explain_helper.py             # DUPLICATED from corpus/sweep/explain.py (~75 LOC)
├── cases/                             # discovery cases — sibling of scripts/
│   ├── vits_model_train.py
│   ├── mistral3_data_dep.py
│   ├── dbrx_moe_data_dep.py
│   ├── jamba_mask_branch.py
│   ├── aria_data_dep.py
│   ├── paddle_ocr_vl_data_dep.py
│   └── _smoke_parallel.py
├── experiments/                       # per-experiment plans + reports
│   ├── README.md
│   ├── 2026-04-cross-case-skill-discovery/
│   ├── 2026-04-parallel-runner-vits-validation/
│   └── 2026-04-phase1-pilot-cases/
├── skills/                            # skill markdown files
│   ├── debug-graph-breaks/
│   └── per-case-analysis/
├── design/                            # all design docs
│   ├── design.md
│   ├── parallel-runner-design.md
│   └── experiment_lifecycle.md
├── README.md                          # short intro pointing at design/
├── CLAUDE.md                          # discovery-only tier table + lifecycle reference
└── .gitignore
```

Imports become:
- Within discovery: `from scripts.runner import CaseSpec, WatchedFile`
- Cases import from scripts: `from scripts.runner import CaseSpec`
- Validator + perf use `from scripts.explain_helper import run_graph_break_analysis` (NOT `from sweep.explain import ...`)
- *No corpus imports anywhere in discovery code*

---

## Phase-by-phase execution (tomorrow morning, ~3 hours)

### Phase 1 — Bootstrap new repo (~15 min)

1. Create GitHub repo `penguinwu/pt2-skill-discovery` (private — your call on visibility)
2. Initial commit:
   - `README.md` (3 lines: "Discovery harness for compile skill investigation. See design/ for architecture, CLAUDE.md for operating rules.")
   - `.gitignore`: Python defaults + `/tmp/runs/` + `/tmp/discovery-runs/`
   - `CLAUDE.md` STUB (will populate in Phase 7)
3. Local clone to `~/projects/pt2-skill-discovery/`
4. Verify push permission works for the new repo (extend the existing sudo+proxy push pattern; update Otter's local CLAUDE.md scope in Phase 7)

### Phase 2 — File moves + restructure (~30 min)

1. Create the directory skeleton: `scripts/`, `cases/`, `experiments/`, `skills/`, `design/`
2. Copy files per layout above:
   - `corpus/discovery/{runner,run_config,launch_parallel,merge_results,revalidate,validate_runner,perf,_measure_case,variants,_lifecycle_gate,clean_sandboxes,smoke_test}.py` → `scripts/`
   - `corpus/discovery/cases/*.py` → `cases/`
   - `corpus/discovery/experiments/*` → `experiments/`
   - `corpus/discovery/skills/*` → `skills/`
   - `corpus/discovery/{design,parallel-runner-design,EXPERIMENT_LIFECYCLE}.md` → `design/` (rename `EXPERIMENT_LIFECYCLE.md` → `experiment_lifecycle.md` per Peng's spec)
3. Add `scripts/__init__.py` (empty)
4. Copy `corpus/sweep/explain.py:run_graph_break_analysis` (the function + its imports + the `_BreakCollector` class it depends on, ~75 LOC total) → `scripts/explain_helper.py`
5. Skip `discovery/runs/` (gitignored anyway; new runs go to new repo's `/tmp/runs/` tree)

### Phase 3 — Update imports (~30 min)

For each file in `scripts/` and `cases/`, replace:
- `from discovery.X import Y` → `from scripts.X import Y`
- `from sweep.explain import run_graph_break_analysis` → `from scripts.explain_helper import run_graph_break_analysis`
- `from discovery.runner import CaseSpec, WatchedFile` (in cases) → `from scripts.runner import CaseSpec, WatchedFile`

Specific files known to have cross-module imports:
- `validate_runner.py` imports sweep.explain (changes per above)
- All cases import `from discovery.runner import CaseSpec, WatchedFile` (changes per above)
- `_lifecycle_gate.py` may have references to `discovery/...` paths in its smoke-test invocation — change to `scripts/...`

### Phase 4 — Update path references (~15 min)

1. `scripts/_lifecycle_gate.py`: smoke-test command becomes `python -m scripts.smoke_test --skip-cases`. Smoke timestamp file `/tmp/.discovery_smoke_last_pass` can stay as-is (cross-repo neutral)
2. `scripts/run_config.py`:
   - `TF_PACKAGE_ROOT` constant unchanged (still `/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers`)
   - Import paths for case discovery: `importlib.import_module(f"cases.{case_id}")` (was `discovery.cases.X`)
3. `scripts/launch_parallel.py`: spawns `python -m scripts.run_config ...` (was `discovery.run_config`)
4. `scripts/merge_results.py`: same kind of changes if it has any spawning
5. `scripts/_measure_case.py`: imports cases similarly
6. `scripts/clean_sandboxes.py`: scan paths `/tmp/runs/` and `/tmp/discovery-runs/` — keep both (legacy + new)

### Phase 3.5 — Absolute-path audit (~15 min) [ADDED rev 2]

The "no corpus imports" claim is true at the Python `import` level, but absolute-path string leaks exist in the source tree and must be fixed before Phase 5.

Confirmed leaks (from 2026-04-29 audit):
- `discovery/filesystem_integrity.py:41` — hardcodes `/home/pengwu/projects/oss-model-graph-break-corpus/` in a docstring/comment.
- `discovery/experiments/2026-04-cross-case-skill-discovery/plan.md` lines 60, 162, 165 — shell-command examples reference `corpus/discovery/skills/...` and `python -m discovery.run_case`.
- `discovery/experiments/2026-04-parallel-runner-vits-validation/plan.md:96` — same shape.

Procedure:
1. `grep -rn "oss-model-graph-break-corpus" scripts/ cases/ experiments/ skills/` in new repo — fix every hit. Either (a) remove the absolute path entirely, (b) rewrite as `~/projects/pt2-skill-discovery/...`, or (c) parametrize via env var.
2. `grep -rn "/discovery/" scripts/ cases/` (excluding intentional comments) — fix.
3. `grep -rn "from discovery\." scripts/ cases/` — should be 0 after Phase 3 import update.
4. `grep -rn "python -m discovery" experiments/ skills/` — replace with `python -m scripts...`.

Failure mode if skipped: Phase 5 smoke test passes (because Python imports are clean) but a future agent copy-pastes a stale shell command from a plan.md and operates against the wrong repo.

### Phase 5 — Validation (~3-4 hours, was 45-60 min) [EXPANDED rev 2]

Original plan was a single smoke + single Gate 2 trial — proves "structurally working", not "fully functioning". Expanded into a tiered acceptance gate. Each tier MUST pass before the next.

**Tier 0 — Static (~5 min):**
- Per-module import probe: for each .py in `scripts/` and `cases/`, run `python -c "import scripts.X"` (or `import cases.Y`). Any ImportError = blocking.
- Absolute-path grep: `grep -rn "oss-model-graph-break-corpus" scripts/ cases/` — must be 0 hits outside design narrative.
- Stale-import grep: `grep -rn "from discovery\." scripts/ cases/` — must be 0.

**Tier 1 — Self-tests (~10 min):**
- `python -m scripts.smoke_test --skip-cases` (Layer 1, 9 tests). Outputs must match corpus baseline.
- `python -m scripts.smoke_test --cases-only` (Layer 2, per-case validate). Match corpus output.

**Tier 2 — Single-trial structural (~45 min):**
- Gate 2 single VITS V8 trial via launch_parallel (original Phase 5 step 3 command, unchanged).
- Compare result.json against corpus's last Gate 2 output: same schema fields populated; similar fix_status / gb_count / eager_self_diff (exact agent text varies); perf_shape_sanity OK on both tiers; flags consistent.

**Tier 3 — Full feature surface (~60 min):**
- Multi-config parallel launch: `python -m scripts.launch_parallel --case vits_model_train --variants V0,V8 --skills none --n 2 --max-parallel 2 ...`. Verifies parallel orchestrator + lifecycle gate end-to-end.
- `merge_results.py` against the multi-config output. Verifies post-hoc aggregator.
- `revalidate.py` against an existing run. Verifies revalidation tool.
- `clean_sandboxes.py --dry-run`. Verifies housekeeping path scanning.

**Tier 4 — Cold-start isolation (~15 min):**
- `mv ~/projects/oss-model-graph-break-corpus/discovery /tmp/discovery_aside_$(date +%s)` (temporarily move corpus discovery dir aside)
- Re-run Tier 1 + Tier 2 + Tier 3 in new repo. ANY failure = there's a hidden runtime dep on corpus's discovery dir.
- Restore: `mv /tmp/discovery_aside_* ~/projects/oss-model-graph-break-corpus/discovery`
- This is the load-bearing test for the "zero corpus runtime deps" claim.

**Tier 5 — Cross-cutting integrations (~30 min):**
- Daily-briefing skill: port wholesale to new repo with its tool dependencies (brief_data.py, check_plan.py, check_experiments.py). New repo gets its own scoped brief; corpus keeps its current brief unchanged. *Do not refactor for multi-repo awareness* — per-project briefs is the right shape; an Otter-level meta-skill that orchestrates them is a separate follow-up (see Open Loops), not migration scope.
- Tools triage (4 tools currently in `corpus/tools/` that touch discovery):
  - `tools/check_experiments.py` — scans `discovery/experiments/`. Port to new repo (called by new repo's brief_data.py).
  - `tools/brief_data.py` — port to new repo, scoped to new repo via `REPO = Path(__file__).resolve().parents[1]` (already self-resolving — no code change needed beyond the move).
  - `tools/check_plan.py` — port to new repo (called by brief_data.py).
  - `tools/new_experiment.py` — port to new repo (it creates discovery experiments).
  - `tools/new_case_issue.py` — files GitHub issues. Decision needed: target pt2-skill-discovery or stay corpus? (See "Open decision" below.)

**Tier 6 — Dual-run period (3-5 calendar days, passive):**
- Both repos exist. corpus/discovery/ stays on disk, marked READ-ONLY (banner only — see Phase 6a).
- All NEW discovery work goes to new repo only.
- Spot-check daily that I haven't fallen back to corpus paths.
- Only after Tier 6 passes do we execute Phase 6b (the actual `git rm`).

If any tier fails: see "Rollback" section below before re-trying.

### Phase 6 — Corpus-side cleanup [SPLIT into 6a + 6b in rev 2]

Per Peng's "don't remove old until new is proven", split into two phases separated by the Tier 6 dual-run period.

#### Phase 6a — Mark corpus discovery READ-ONLY (~15 min, immediately after Tier 5 passes)

1. Add `corpus/discovery/MIGRATED.md` banner: "Authoritative copy at penguinwu/pt2-skill-discovery as of YYYY-MM-DD. DO NOT EDIT HERE — edits will be lost." Include link.
2. Update corpus `CLAUDE.md`:
   - Add a top-level note: "discovery/ has migrated — see corpus/discovery/MIGRATED.md"
   - Remove "## Discovery Experiments" section
   - Remove the discovery row from the tier table
   - Add note pointing at new repo
3. Commit + push corpus banner change. Files stay on disk for fallback.

Result: corpus/discovery/ is still readable, runnable as a fallback if the new repo breaks, but conceptually frozen.

#### Phase 6b — Actual deletion (~20 min, after Tier 6 dual-run period)

Only execute if Tier 6 passed cleanly: ≥1 real workload completed in new repo, no fallbacks observed.

1. `git rm -r discovery/` from corpus repo
2. Replace `corpus/discovery/MIGRATED.md` with `corpus/discovery/MOVED.md` (one-line pointer): "Skill-discovery work has moved to penguinwu/pt2-skill-discovery"
3. `tools/check_experiments.py`: delete from corpus (port already happened in Tier 5)
4. Audit corpus `tools/` for any tool that still references `discovery/`: should be 0 after Tier 5 triage; verify
5. Commit + push corpus cleanup

### Phase 7 — Otter context update (~20 min) [EXPANDED rev 2]

1. Update `~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md`:
   - Recognize TWO primary projects: corpus + pt2-skill-discovery
   - Add *context-switch protocol* (behavioral): "When switching project context, state the project name explicitly in chat."
   - Add *mechanical safeguard* [ADDED rev 2]: at session start, parse the active cwd; if cwd is in either repo, emit a one-line "Project: <name>" banner as the first line of every response that touches files. If cwd is ambiguous (neither/outside both), require explicit project declaration before any file edit.
   - Update push-permission section: extend git-push permission from `penguinwu/oss-model-graph-break-corpus` to ALSO include `penguinwu/pt2-skill-discovery` (same guardrails: only push to main, never force, log every push, etc.)
   - Update CLAUDE.md primary project line to acknowledge both
2. Populate the new pt2-skill-discovery `CLAUDE.md`:
   - Tier table (Tier A only — discovery is the only thing that lives here now)
   - Reference to `design/experiment_lifecycle.md`
   - Universal closure rule (copy from corpus)
   - Push permission acknowledgment

### Phase 8 — Wrap-up + external references audit (~30 min) [EXPANDED rev 2]

1. *Otter memory audit:* grep memory dir for `discovery/`, `corpus/discovery/`, `oss-model-graph-break-corpus/discovery`. Update each hit:
   - `reference_corpus_seeding.md` — update paths
   - `reference_pytorch_skills.md` — verify
   - `reference_sweep_skill.md` — verify
   - Any other entries
2. *External references audit* [ADDED rev 2]:
   - Field report (`sweep_results/experiments/animesh-fullgraph-2026-04-28/FIELD-REPORT-GRAPH-BREAKS.md` + Drive doc 1KpJP00U9lPxe3TzA42Sd8m7oqHd2x468Ju52hUGPQnk) — update Cross-references row pointing at `discovery/experiments/.../findings.md` to point at new-repo path or just reference the Drive doc.
   - VITS report Drive doc 1I3JU8oKajuPzYi6hmwYbaK9px-Jsjw_sm43uJBONON8 — content references corpus paths; update if material.
   - AutoDev kanban: file a tracking task "Discovery issues now go to pt2-skill-discovery" so future agents see the routing change.
   - Workplace posts referencing corpus/discovery: best-effort comment with new location.
3. *Cron audit:* `sqlite3 ~/.myclaw/myclaw.db "select name, command from jobs where command like '%discovery%';"` — confirm 0 hits (current state, but re-verify post-migration).
4. Verify the parallel runner experiment from commit `8a4881a` era is reproducible in the new repo — Tier 2/3 already covers this.

---

## Open decision: AutoDev issue ownership [ADDED rev 2]

`tools/new_case_issue.py` currently files GitHub issues to `penguinwu/oss-model-graph-break-corpus`. After migration, two options:

- **Option A (lean):** Discovery issues → pt2-skill-discovery (own kanban, own labels, own filtering). Corpus issues stay in corpus. Clean separation, mirrors the repo split.
- **Option B:** All issues stay in corpus (single kanban for the team). Discovery just lives in a different repo.

Otter's recommendation: Option A. But this is a Peng decision; default left unspecified until decided.

---

## Rollback plan [ADDED rev 2]

The new repo is created early (Phase 1). If a later phase reveals the layout is wrong, here's what to do:

**Rollback Tier 0/1 failure (in pt2-skill-discovery, before any push to main):**
- Edit locally; re-test; only push when clean.
- No corpus-side cleanup happens until Phase 6a; corpus is untouched.

**Rollback Tier 2/3 failure (single trial / multi-config broken):**
- Diagnose locally. Likely import or path issue surfaced by real workload.
- corpus/discovery/ is still authoritative — fall back is "use corpus, fix new repo, retry."

**Rollback Tier 4 cold-start failure (hidden corpus dep discovered):**
- This is the most informative failure — surfaces a genuine layout assumption that's wrong.
- Restore `mv /tmp/discovery_aside_* ~/projects/oss-model-graph-break-corpus/discovery` immediately.
- Document the missed dep in this plan; revise Phase 2/3 accordingly; retry Tier 4.

**Rollback Tier 5/6 failure (skill / tool / dual-run issue):**
- New repo is functional but ecosystem-incomplete. Fix in place; do NOT proceed to 6b.
- corpus/discovery/ remains as fallback.

**Catastrophic rollback (decision to abandon the v1 layout):**
- Do NOT delete the GitHub repo (history is cheap; abandoning a "v1" branch is fine).
- Mark new repo's main branch as "abandoned-v1" in README.
- All work continues in corpus until v2 plan is drafted.
- corpus/discovery/ is untouched throughout, so this is a no-op rollback.

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Stale path references missed during import update | Phase 5's validation catches them — smoke test + Gate 2 trial would fail |
| Cross-repo state drift (e.g., we update corpus paths and forget to mirror) | Should be rare since discovery is fully self-contained at runtime; document in pt2-skill-discovery CLAUDE.md if dependencies appear |
| Bi-project Otter context mix-ups | Phase 7 explicit context-switch protocol; revisit if mix-ups continue (could split to a second agent later) |
| `discovery/runs/` history left behind in corpus | Acceptable — they're gitignored; future runs go to new repo's `/tmp/runs/` |
| Cron jobs that reference moved files | Phase 8 audit — none expected, verify |
| Smoke test's `/tmp/.discovery_smoke_last_pass` shared across repos | Acceptable — cross-repo neutral; both repos check + write the same file. Or rename to `/tmp/.skill_discovery_smoke_last_pass` to disambiguate |

---

## Option B (deferred — eventual upgrade path)

If/when AutoDev needs to consume discovery code as a library OR a 2nd contributor wants to install discovery into their venv:

1. Add `pyproject.toml` to pt2-skill-discovery declaring it as a Python package (~20 LOC + decide what's public)
2. `pip install -e ~/projects/pt2-skill-discovery` in the consumer's venv
3. Replace any path-based imports (none currently) with normal `from scripts.X import Y`

Cost: ~30-45 min when we get there. No urgency now.

---

## Tomorrow-morning resume index

If you (or another agent) wake up cold and pick this up:
1. Read this doc top-to-bottom
2. Phase 1 Step 1 is the first action: create the GitHub repo
3. Phases are strictly ordered — don't skip Phase 5 validation before Phase 6 cleanup
4. If Phase 5 fails, fix and re-run; do NOT delete the corpus-side discovery dir until validation is clean

## Revision log

- *2026-04-27 21:11 ET* — Plan filed after iteration with Peng. Approved for tomorrow-morning execution.
- *2026-04-29 19:15 ET* — Rev 2.1 (clarification). Per Peng: don't make the daily-briefing skill load-bearing for migration. Right shape is per-project skills (each repo owns its own), with an Otter-level meta-skill orchestrating — but that's a separate follow-up, not migration scope. Tier 5 simplified accordingly: port the skill + its tool deps wholesale, no multi-repo refactor.
- *2026-04-29 18:50 ET* — Rev 2. Augmented after holistic review:
  - Added Phase 3.5 (absolute-path audit) — found leaks in `filesystem_integrity.py` + 2 experiment plan.md files.
  - Expanded Phase 5 from "smoke + single trial" into 6-tier acceptance gate (static / self-test / single-trial / full feature surface / cold-start isolation / dual-run period).
  - Split Phase 6 into 6a (banner only, immediately) and 6b (actual `git rm`, deferred ≥3-5 days after Tier 6 dual-run validates) per Peng's "don't remove old until new is proven."
  - Expanded Phase 7 with mechanical bi-project safeguard (cwd-based banner) — behavioral rule alone is fragile.
  - Expanded Phase 8 with external references audit (field report, Drive docs, AutoDev kanban, cron).
  - Added "Open decision: AutoDev issue ownership" section.
  - Added "Rollback plan" section with per-tier failure paths.
  - Did NOT change weakness #3 (duplicated explain_helper drift) per Peng — design choice from round 1.
