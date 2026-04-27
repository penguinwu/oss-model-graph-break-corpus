# Parallel Discovery Runner — Design

**Status:** design draft, 2026-04-27. Not yet implemented.
**Author:** Otter

## What problem this solves

A discovery experiment runs M independent configurations (variants × skill arms × trial seeds). Today they run sequentially: a 12-config experiment takes ~3 hours. Each config is fundamentally independent — no inter-config dependency — so parallelism is a clear infra win.

Goal: launch M configurations as M independent OS processes, then merge per-config results into one summary.

## The fundamental design choice — what gets replicated per parallel runner

*The entire `transformers` package source directory is replicated per trial (~49 MB), plus the baseline test script. Made visible to the trial via `PYTHONPATH`.*

Why the entire package, not just the one file the agent edits:

Python's import system can't shadow a single file inside a package. When code does `import transformers.models.vits.modeling_vits`, Python finds `transformers/` first, then descends. To control which `modeling_vits.py` Python loads, you have to control which `transformers/` directory it descends through. Simplest way: copy the whole package and put our copy first on the search path with `PYTHONPATH=$SANDBOX:$PYTHONPATH`.

Per-trial directory layout:

```
/tmp/runs/<exp>/<config_id>/
├── sandbox/                              ← per-trial; cleaned up after
│   ├── transformers/                     ← FULL COPY of transformers (~49 MB)
│   │   └── models/vits/modeling_vits.py  ← agent edits THIS file
│   └── baseline_vits.py                  ← test script copy (~few KB)
├── prompt.txt                            ← agent prompt with sandbox paths
├── stream.jsonl, agent_diff.patch
└── result.json
```

Replicated vs watched distinction:
- *Replicated:* the entire transformers package — needed for Python imports to work
- *Watched:* just `modeling_vits.py` + `baseline_vits.py` — the files the agent may edit; mutation check flags any unrelated writes

Cost: 49 MB × 6 parallel trials = ~300 MB disk; `cp -r` is ~2-5 sec per trial.

## Architecture

```
launch_parallel.sh                              # 5-line shell loop
  walk lifecycle gate ONCE (smoke + plan.md)
  for config_id in cfg1..cfgM:
    nohup python -m discovery.run_config \
      --config <config_id> --out /tmp/runs/<exp>/<config_id>/ &
  wait
  python -m discovery.merge_results --in /tmp/runs/<exp>/ --out summary.{md,json}
```

Each `run_config` process owns one configuration end-to-end:

```
run_config.py:
  1. mkdir /tmp/runs/<exp>/<cfg>/sandbox/
  2. cp -r ~/envs/torch211/lib/python3.12/site-packages/transformers $SANDBOX/transformers
  3. cp /tmp/discovery-runs/<case>/baseline_vits.py.original $SANDBOX/baseline_vits.py
  4. compose prompt: substitutes $SANDBOX paths into case body
  5. invoke agent (claude subprocess) with PYTHONPATH=$SANDBOX prepended
  6. capture diff: diff -u <originals> $SANDBOX/...
  7. run validator subprocess (inherits PYTHONPATH from this process)
  8. run perf subprocess (inherits PYTHONPATH from this process)
  9. write result.json
  10. cleanup sandbox (or keep on failure)
```

## Sandbox lifecycle

```
PRE-TRIAL: setup
  mkdir $SANDBOX
  cp -r .../transformers $SANDBOX/
  cp <baseline_test_script>.original $SANDBOX/<baseline_test_script>
  → ~2-5s, ~50 MB

DURING-TRIAL: agent edits
  $SANDBOX/transformers/models/vits/modeling_vits.py  (agent writes)
  $SANDBOX/baseline_vits.py  (agent writes)

POST-AGENT: capture
  diff -u <originals> $SANDBOX/... → agent_diff.patch

POST-TRIAL: validate + perf
  Subprocesses inherit PYTHONPATH=$SANDBOX:... and import from sandbox

CLEANUP:
  Failure → keep dir for inspection
  Success → rm -rf $SANDBOX
```

## What changes vs current sequential design

| Today (sequential) | This design (parallel) |
|---|---|
| Agent edits live files in site-packages | Agent edits files in per-trial sandbox |
| Restore from .original between trials | Throw away sandbox dir between trials |
| Validator/perf import from site-packages (sees agent edits because file is mutated) | Validator/perf inherit `PYTHONPATH`; import from sandbox |
| One trial at a time on the GPU | M trials in parallel |

## Case spec extension

Each case spec adds one method `get_case_spec_sandboxed(sandbox: Path)` that returns a `CaseSpec` with paths pointing into the sandbox. The existing `get_case_spec()` stays for back-compat with the sequential `run_case.py`. CASE_BODY becomes a template with `{BASELINE_SCRIPT}` / `{VITS_SRC}` placeholders.

5 case specs to migrate (vits, mistral3, dbrx, jamba, aria, paddle_ocr_vl). Each is ~30 LOC + careful prose review of the case body to ensure rendered prompt reads naturally.

## Implementation order

1. `discovery/run_config.py` — owns one config end-to-end (refactored from `runner.py:run_trial`)
2. `discovery/launch_parallel.py` — lifecycle gate + spawn loop + wait + merge
3. `discovery/merge_results.py` — glob result.json files, aggregate
4. Per-case `get_case_spec_sandboxed()` + CASE_BODY template (start with VITS, validate end-to-end, then propagate)
5. Smoke test: `test_parallel_runs_isolated` (3 parallel run_config against synthetic case)

Total scope: ~500 LOC + ~5 hrs of careful per-case prompt rewriting.

## Open questions

1. *Default `--max-parallel`?* Lean: 6 for VITS-class cases. Measure per-case via smoke before going higher.
2. *Cleanup-on-failure precise definition?* Lean: failure = (`agent_exit_code != 0` AND `!= 124`) OR `validation.error not None`. Timeout (124) is the agent's deliberate cap; not a "broken trial."
3. *Mutation check semantics now that the boundary is "soft"?* — agent could edit any file under sandbox/transformers/, not just the watched ones. Either (a) extend mutation check to scan all of sandbox/transformers/ for unexpected writes, or (b) accept the soft boundary and rely on prompt discipline.
4. *Per-process Python startup overhead* — ~5-8s per process for torch + transformers import. At 6-way parallelism, this is amortized but visible. Need to measure on this VM.
5. *Inductor + Triton cache contention under concurrent compile* — file-locked, so safe but possibly serializing. Need to measure.

## Risks

- *PYTHONPATH precedence subtlety:* if `run_config.py` itself imports `transformers` (even transitively), its sys.modules caches the site-packages version. Subprocesses spawned with the right `env=` get the sandbox version (subprocess starts fresh). Discipline: `run_config.py` should NOT import transformers in its own process; only in subprocesses.
- *Cleanup discipline:* failed trials leave sandbox dirs (~50 MB each). Need a `discovery/clean_sandboxes.py` housekeeping tool.
- *Per-process startup overhead* (above) — could limit speedup at high parallelism.

## Lifecycle integration

- Launcher walks the lifecycle gate ONCE before spawning configs.
- Per-config processes do NOT re-walk (would waste ~10s × N).
- Bypass at the launcher level (`--lifecycle-bypass --reason "..."`) applies to all spawned configs.
