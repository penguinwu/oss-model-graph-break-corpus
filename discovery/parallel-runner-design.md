# Parallel Discovery Runner — Design

**Status:** design draft (v0.1, 2026-04-27). Not yet implemented.
**Author:** Otter
**Motivation:** The current discovery harness is sequential. A 12-config experiment (e.g., 4 variants × 3 trials × 1 skill arm, or 2 variants × 2 skill arms × 3 trials) takes ~3 hours of wall time at ~15 min/config. Each config is fundamentally independent — no inter-config dependency — so parallelism is the obvious next infra investment.

## Goal

Run the M configurations of a discovery experiment in parallel as N independent processes, each owning its own sandbox of all writable files, then merge the per-config results into one summary post-hoc.

Target speedup: 6-12x at typical experiment sizes (limited by GPU memory, not CPU). Real wall-time reduction: ~3 hours → ~30 minutes for typical 12-config matrices.

## Non-goals

- *Not* a replacement for the lifecycle gates — every parallel launch still goes through smoke_test + plan.md gate-checks (lifecycle gate runs ONCE in the launcher, not per-config).
- *Not* an attempt to share state across configs — configs are 100% independent by design. Cross-config insights are the merge step's job.
- *Not* a distributed system — assumes one machine, one GPU initially. Multi-GPU/multi-machine is a future extension; the architecture is compatible but out-of-scope.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  launch_experiment.sh   (or python -m discovery.launch_parallel)│
│  • Parse experiment definition (configs to run, output dir)     │
│  • Walk the lifecycle gate ONCE (smoke_test + plan.md)          │
│  • For each config: spawn `python -m discovery.run_config ...`  │
│    in background; capture pid                                   │
│  • Wait for all to complete                                     │
│  • Invoke `python -m discovery.merge_results ...`               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (N independent processes)
┌─────────────────────────────────────────────────────────────────┐
│  discovery.run_config --config CFG --out OUT_DIR                │
│  Owns one configuration end-to-end:                             │
│  • mkdir sandbox/ under OUT_DIR                                 │
│  • cp .original watched files → sandbox/                        │
│  • Compose prompt with sandbox-relative paths                   │
│  • Invoke agent (claude subprocess) — agent edits sandbox/ files│
│  • Capture diff: sandbox/ files vs .original                    │
│  • Run validator (subprocess) — sees only sandbox/              │
│  • Run perf (subprocess) — sees only sandbox/                   │
│  • Write result.json to OUT_DIR                                 │
│  • Cleanup sandbox (or keep on failure for inspection)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (after all configs complete)
┌─────────────────────────────────────────────────────────────────┐
│  discovery.merge_results --in EXPERIMENT_DIR --out summary.md   │
│  • Glob result.json from all per-config dirs                    │
│  • Compute cross-config aggregates (fix_status distribution,    │
│    fix_survives_perf distribution, gb_call_sites by type)       │
│  • Re-run cross-trial flags that orchestrator did inline        │
│    (e.g., tier1-tier2-direction-mismatch — now per-config local)│
│  • Write summary.md + summary.json                              │
└─────────────────────────────────────────────────────────────────┘
```

## Sandbox contract

Each config owns a directory `<EXPERIMENT_DIR>/<config_id>/` with:

```
<EXPERIMENT_DIR>/<config_id>/
├── sandbox/                        # Per-config copies of all watched files
│   ├── modeling_vits.py            # writable; agent edits this
│   ├── baseline_vits.py            # writable; agent may edit this
│   └── ... (other watched files)
├── prompt.txt                      # Agent prompt (with sandbox/ paths)
├── stream.jsonl                    # Agent's stream output
├── claude_stderr.log
├── agent_diff.patch                # diff -u .original sandbox/
├── validation_stdout.log
├── validation_stderr.log
├── perf_stdout.log
├── perf_tier2_stdout.log
└── result.json                     # Final per-config result
```

**What gets sandboxed:** every entry in `case.watched_files`. The case spec lists these; the per-config script reads the spec and sets up the sandbox accordingly. No case-level changes needed if we use environment variables (next section).

**What stays shared (read-only or content-addressed):**
- Original `.original` snapshots (read-only)
- HF model cache (`~/.cache/huggingface/`) — read-only post-download
- Inductor cache (`/tmp/torchinductor_*/`) — content-addressed, file-locked
- Triton cache (`~/.triton/cache/`) — content-addressed, file-locked

These do NOT need sandboxing. Concurrent reads or content-addressed writes are safe.

## Per-config path discipline

The agent + validator + perf code all reference watched-file paths. Today those paths are hardcoded in:
- The case spec's `watched_files` list (e.g., `/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py`)
- The case body prompt (e.g., "edit /home/pengwu/.../modeling_vits.py")
- The validate.py shim (imports `transformers.models.vits.modeling_vits`)
- The perf script (imports same)

For per-config sandboxing, all of these must point at `<EXPERIMENT_DIR>/<config_id>/sandbox/...` instead. Two design options:

**Option A — Environment variable convention (RECOMMENDED).**
- `DISCOVERY_SANDBOX=<path>` env var set per process
- Case spec's `watched_files` becomes a property: `watched_files = [WatchedFile(SANDBOX/modeling_vits.py, ORIGINAL/modeling_vits.py.original)]` where SANDBOX is read from env
- The validator/perf shadows `transformers.models.vits.modeling_vits` via the importlib pattern proven in the feasibility test (commit pending), pointing at `SANDBOX/modeling_vits.py`
- Case body prompt is a template: `f"edit {SANDBOX}/modeling_vits.py"` formatted at config-launch time

**Option B — Argument-passing through every layer.**
- Add `--sandbox` flag to validate.py, _measure_case.py, run_config.py
- Each layer threads it through to the next
- More plumbing, more places to forget

Lean: A. Env var is one mechanism, set once per process, observable everywhere. The shadowing happens in one place (a `_sandbox.py` helper), validator/perf import paths through it.

## API sketches

### `discovery/_sandbox.py` (new helper)

```python
"""Per-config sandbox primitives. Used by run_config.py + validator + perf."""
import os, sys, importlib.util, shutil
from pathlib import Path

def get_sandbox_dir() -> Path | None:
    """Returns DISCOVERY_SANDBOX env var as Path, or None if not set
    (back-compat: code outside parallel mode falls through to original paths)."""
    s = os.environ.get("DISCOVERY_SANDBOX")
    return Path(s) if s else None

def setup_sandbox(case, sandbox_dir: Path) -> None:
    """Copy each .original watched file into sandbox_dir. Idempotent."""
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    for wf in case.watched_files:
        if wf.original_backup.exists():
            shutil.copyfile(wf.original_backup, sandbox_dir / wf.path.name)

def shadow_module(dotted_name: str, sandbox_dir: Path) -> None:
    """Replace `dotted_name` in sys.modules with the version in sandbox_dir.
    Caller passes e.g. 'transformers.models.vits.modeling_vits'.
    Filename in sandbox is the basename of the original (e.g. modeling_vits.py)."""
    parent_dotted = ".".join(dotted_name.split(".")[:-1])
    leaf = dotted_name.split(".")[-1]
    # Force-load parents
    __import__(parent_dotted)
    sandbox_file = sandbox_dir / f"{leaf}.py"
    spec = importlib.util.spec_from_file_location(dotted_name, sandbox_file)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = parent_dotted
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)

def cleanup_sandbox(sandbox_dir: Path, keep_on_failure: bool = True, succeeded: bool = True) -> None:
    """Remove sandbox dir unless we want to keep it for inspection."""
    if keep_on_failure and not succeeded:
        return
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir)
```

### `discovery/run_config.py` (new entry point — owns one config)

```python
"""Run one discovery configuration end-to-end. Independent of all other configs."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from discovery._sandbox import setup_sandbox, cleanup_sandbox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--skill", default="none")
    parser.add_argument("--trial-label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sandbox = args.out_dir / "sandbox"

    # Set env var for downstream subprocess invocations
    os.environ["DISCOVERY_SANDBOX"] = str(sandbox)

    # Load case spec, set up sandbox
    import importlib
    case_mod = importlib.import_module(f"discovery.cases.{args.case}")
    case = case_mod.get_case_spec()
    setup_sandbox(case, sandbox)

    # Compose prompt with sandbox paths (case body is a template)
    # ... compose, invoke agent, capture diff, run validator + perf, write result.json
    # See runner.py:run_trial for the existing logic — refactored to use sandbox/
```

### `discovery/launch_parallel.py` (new top-level entry)

```python
"""Launch N parallel run_config processes for one experiment. Walk lifecycle
gate ONCE (not per-config). Wait for all. Invoke merge_results."""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument("--config-spec", required=True,
                        help="JSON or YAML listing the M configurations")
    parser.add_argument("--max-parallel", type=int, default=6,
                        help="Cap on concurrent processes (GPU memory budget)")
    add_lifecycle_args(parser)
    args = parser.parse_args()

    plan_path = args.experiment_dir / "plan.md"
    check_or_die(plan_path, args, launcher="discovery/launch_parallel.py")

    # Spawn N processes in a pool; wait
    # ...
    # Then call merge_results
```

### `discovery/merge_results.py` (new post-hoc aggregator)

```python
"""Glob per-config result.json files, compute cross-config aggregates,
write summary.md + summary.json."""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    results = {}
    for cfg_dir in sorted(args.in_dir.iterdir()):
        rj = cfg_dir / "result.json"
        if rj.exists():
            results[cfg_dir.name] = json.loads(rj.read_text())

    # Distribution: fix_status, fix_survives_perf, gb_types
    # Cross-trial flags: tier1-tier2-direction-mismatch (was inline in runner.py;
    # now post-hoc per-config so this is fine — flag computed at write_result time)
    # Build summary
```

## Lifecycle integration

- The launcher walks the lifecycle gate ONCE before spawning any config processes (smoke_test pass + plan.md gates ticked). Per-config processes do NOT re-walk the gate (would waste ~10s × N processes).
- Each per-config process is a "child" of the experiment for lifecycle purposes — its result feeds Gate 4's full-scale data.
- Bypass at the launcher level (`--lifecycle-bypass --reason "..."`) writes audit entry to the experiment plan.md, applies to all spawned configs.

## Migration plan from current `run_case.py`

Two paths:

**Path A — Replace.** `run_case.py` is deprecated; new code uses `launch_parallel.py`. Sequential mode still available as `--max-parallel 1`.

**Path B — Coexist.** `run_case.py` keeps doing sequential trials in-process (fine for development / debugging single trials). `launch_parallel.py` is the production path for multi-config experiments. Both reuse the same `run_config.py` building block (run_case becomes a thin orchestrator over run_config).

Lean: B. Keeps the simple-debugging path intact (a single trial via `python -m discovery.run_case`) while making parallel the production default.

## Open questions

1. **`max_parallel` default?** Lean: 6 (one per "skill × variant" cell at N=3, fits comfortably in 80GB GPU memory). Configurable.
2. **Cleanup policy for sandbox dirs?** Lean: keep on failure (for inspection); remove on success. Configurable via `--keep-sandbox` flag.
3. **Stream.jsonl size budget?** Each agent run produces ~10-50 MB of stream output. 12 configs × 50MB = 600 MB. Per-config dirs handle this fine; need cleanup policy if /tmp fills up.
4. **GPU memory pressure under maximum parallelism?** Need a smoke test: launch 12 parallel VITS forwards, measure peak GPU memory. Likely fine (12 × 18MB = 216MB), but worth verifying for larger cases (Aria, PaddleOCRVL).
5. **Inductor + Triton cache lock contention?** N parallel agents may all compile the SAME baseline modulus their edits → many compile attempts share cache keys → file-lock contention. Need to measure if this is a meaningful slowdown.
6. **Merge step error handling?** What if 2 of 12 configs failed? Lean: merge produces summary with the 10 successful + a "failed configs" section listing which ones and their error class.
7. **Re-run individual failed configs?** The independent-process design makes this trivial: just re-run the failed config_id. Worth a `--rerun cfg5,cfg7` mode.

## Testing plan

Per the lifecycle:

- *Layer 1 smoke* — synthetic test in `discovery/smoke_test.py`: launch 3 parallel run_config processes against a synthetic clean MLP "case"; verify all 3 produce isolated sandboxes + correct result.json + no cross-process file conflicts.
- *Single-config validation (Gate 2)* — run one config end-to-end via launch_parallel with `--max-parallel 1`. Same output as current run_case.py? Schema correct?
- *Small-batch validation (Gate 3)* — run 3 configs in parallel for VITS. Inspect: do all 3 sandboxes contain isolated edits? Is per-config result.json schema-clean? Does merge produce sensible summary?
- *Cache-lock contention measurement* — same 3-config run, measure: total wall time, vs sum of per-config wall times. Speedup at 3-way should be ~2.5-3x (less than 3 due to startup overhead + cache contention).
- *Full-scale (Gate 4)* — only after the above; run a 12-config experiment.

## Implementation order (LOC estimates per step)

1. *`_sandbox.py`* — primitives (~80 LOC). Standalone, easy to unit-test.
2. *Update validator + perf to honor DISCOVERY_SANDBOX env var* — ~30 LOC across `validate_runner.py`, `_measure_case.py`. Backward compat: if env var unset, fall through to current behavior.
3. *`run_config.py`* — single-config entry point (~120 LOC, mostly refactored from `runner.py:run_trial`).
4. *`launch_parallel.py`* — top-level launcher with lifecycle gate (~80 LOC).
5. *`merge_results.py`* — post-hoc aggregator (~60 LOC).
6. *Update case prompts to use sandbox paths via template format* — ~5-10 LOC per case (5 cases, ~50 LOC).
7. *Smoke test extension* — `test_parallel_runs_isolated` (~50 LOC).

Total: ~470 LOC across ~10 files. Estimated work: 1-2 focused days (Otter time).

## Risks

- *Inductor / Triton cache contention may serialize compiles* — bound by file-lock wait. Real impact only measurable with the cache-contention test.
- *agent prompt rewriting may surface bugs* — agents see different paths than they used to; downstream "agent ran fine" assertions may need updating.
- *case-spec back-compat* — if a case has hardcoded paths in its body, the env-var indirection might miss them. Need a sweep of all cases.
- *Cleanup discipline* — failed configs leave sandbox dirs around. Need a `discovery/clean_sandboxes.py` housekeeping tool eventually.

## Revision log

- *2026-04-27 (v0.1)* — Initial design after Peng asked for parallel discovery runner. Feasibility test commit pending (proved 2-way parallelism works at module-shadowing level).
