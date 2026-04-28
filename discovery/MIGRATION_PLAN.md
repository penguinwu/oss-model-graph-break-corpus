# Migration Plan: discovery → pt2-skill-discovery

**Status:** plan signed off by Peng 2026-04-27 21:06 ET. Execute tomorrow morning.
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

### Phase 5 — Validation (~45-60 min)

1. *Smoke test:* `python -m scripts.smoke_test --skip-cases` in the new repo. All 9 Layer-1 tests should pass with same outputs as in corpus today.
2. *Per-case smoke (Layer 2):* `python -m scripts.smoke_test --cases-only`. Each case's validate.py shim runs against canonical baseline. Should match corpus output.
3. *Re-run Gate 2 single-VITS V8 trial via launch_parallel in new repo:*
   ```
   python -m scripts.launch_parallel \
       --case vits_model_train --variants V8 --skills none --n 1 \
       --experiment-dir /tmp/runs/migration-validation/ \
       --max-parallel 1 \
       --plan experiments/2026-04-parallel-runner-vits-validation/plan.md
   ```
4. Compare result.json against tonight's Gate 2 output (corpus repo, commit `8a4881a` era):
   - Same schema fields populated
   - Similar fix_status / gb_count / eager_self_diff (exact agent diffs may differ run-to-run; structure should match)
   - perf_shape_sanity OK for both tiers
   - flags consistent
5. *If both pass:* migration validated. Proceed to Phase 6.
6. *If either fails:* diagnose, fix, retry — do not proceed until clean.

### Phase 6 — Corpus-side cleanup (~20 min)

1. `git rm -r discovery/` from corpus repo
2. Add `discovery/MOVED.md` (one-line pointer): "Skill-discovery work has moved to penguinwu/pt2-skill-discovery"
3. Update corpus `CLAUDE.md`:
   - Remove "## Discovery Experiments" section
   - Remove the "Discovery (multi-trial...)" row from the tier table (only Tier B + Tier C remain corpus-side)
   - Update "Universal floor" closure rule wording if it references discovery
   - Add note pointing at new repo
4. Update `tools/check_experiments.py`: it scans `discovery/experiments/`. Either (a) delete the tool, or (b) make it skip if dir doesn't exist (tier-A specific check moves with discovery). Lean: delete from corpus, port to discovery if still needed.
5. Update `discovery/README.md` (corpus-side): file is gone after `git rm -r discovery/` — no action needed
6. Audit corpus `tools/` for any tool that still references `discovery/`: update or delete
7. Commit + push corpus cleanup

### Phase 7 — Otter context update (~15 min)

1. Update `~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md`:
   - Recognize TWO primary projects: corpus + pt2-skill-discovery
   - Add explicit *context-switch protocol*: "When switching project context, state the project name explicitly in chat. Don't apply tier-table rules from one repo to the other."
   - Update push-permission section: extend git-push permission from `penguinwu/oss-model-graph-break-corpus` to ALSO include `penguinwu/pt2-skill-discovery` (same guardrails: only push to main, never force, log every push, etc.)
   - Update CLAUDE.md primary project line to acknowledge both
2. Populate the new pt2-skill-discovery `CLAUDE.md`:
   - Tier table (Tier A only — discovery is the only thing that lives here now)
   - Reference to `design/experiment_lifecycle.md`
   - Universal closure rule (copy from corpus)
   - Push permission acknowledgment

### Phase 8 — Wrap-up (~15 min)

1. Update Otter memory entries that reference `discovery/...` paths:
   - `reference_corpus_seeding.md` — update paths to `pt2-skill-discovery/scripts/...`
   - `reference_pytorch_skills.md` — verify any discovery references
   - `reference_sweep_skill.md` — verify
   - Any other entries with `discovery/` in them
2. Verify no cron job touches the moved files. Spot-check `myclaw.db jobs` table for any reference to `discovery/`
3. Verify the parallel runner experiment we just landed (commit `8a4881a` era) is reproducible in the new repo — that's the validation from Phase 5

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
