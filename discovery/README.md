# discovery/

Per-case discovery agent. Sibling to `sweep/`.

`sweep/` is breadth-first across the corpus — "did each model break, how badly."
`discovery/` is depth-first per case — "what strategies exist for this case, what do they cost."

## Status

Phase 1 closed (Dbrx + Jamba). Phase 3 in flight — see `experiments/2026-04-cross-case-skill-discovery/plan.md`.

See `design.md` for the harness-level design (v0.4).

## Layout

```
discovery/
├── README.md          # this file
├── design.md          # harness design (v0.4)
├── plan.md            # WS1 Phase 3 workstream pointer (see experiments/.../plan.md for methodology)
├── perf.py            # measure_perf primitive
├── runner.py          # config-driven trial runner (skill × variant × N)
├── run_case.py        # CLI wrapper around runner
├── variants.py        # constraint variant catalog (V0/V2/V4/V6)
├── enrich_tier2.py    # tier-2 perf enrichment for completed runs
├── _measure_case.py   # subprocess perf measurement (avoids module-state contamination)
├── cases/             # per-case config files (one .py + one .baseline.json per case)
├── runs/              # per-trial raw artifacts (stream.jsonl gitignored, archived to Drive)
├── reports/           # legacy Phase 1 reports — new reports live under experiments/<exp>/reports/
├── experiments/       # per-experiment plans + reports + synthesis (see experiments/README.md)
└── skills/            # vendored Claude skills used by trial agents
    └── debug-graph-breaks/SKILL.md  # Arsh's skill (canonical: azahed98/pytorch fork)
```

## Running

Trial harness:

```bash
python -m discovery.run_case --case <case_id> \
    --variants V0,V2,V4,V6 \
    --skills none,/home/pengwu/projects/oss-model-graph-break-corpus/discovery/skills/debug-graph-breaks/SKILL.md \
    --n 3 --timeout 1800
```

24 trials sequential per case (8 cells × N=3). ~12 hr wall.

Standalone perf measurement:

```bash
python -m discovery.perf  # smoke test
python -m discovery.cases.<case_id>  # per-case baseline measurement
```

## Conventions

See `experiments/README.md` for the per-experiment directory convention. Use the scaffold tools — never hand-roll experiments or per-case issues:

- `tools/new_experiment.py "<slug>" --title "<Title>"` — lightweight default: dir + plan.md + README row, no GitHub issue
- `tools/new_experiment.py "<slug>" --title "<Title>" --with-umbrella-issue` — opt-in to ALSO file an umbrella issue + add to project board
- `tools/new_case_issue.py <experiment-slug> <case_id> "<Model name>"` — opt-in per-case GitHub issue
- `tools/queue_task.py "<title>"` — only when there is no existing per-case issue

Issue creation is *orthogonal* to scaffolding. Most local experiments don't warrant team-visible GitHub tracking; use `--with-umbrella-issue` only for experiments producing shipped findings or needing cross-team comments.

Drift detected by `tools/check_experiments.py` (run nightly via the daily brief).
