# Sweep CLI Redesign — Generic Compiler-Config Passthrough

**Status:** Proposal, awaiting Peng review
**Author:** Otter, 2026-04-27
**Motivation:** Animesh's #70 ask exposed that the sweep is hard-coded to one compile config. Future corpus users (compiler researchers, perf engineers, internal teams) will want to test their own settings without forking the harness.

---

## Problem

Today's `run_sweep.py` flags:
- `--source`, `--modes`, `--workers`, `--timeout`, `--output-dir`

The compile invocation in `worker.py` is hard-coded:
```python
compiled = torch.compile(model, backend="eager", dynamic=compile_dynamic)
```

A user who wants to try `fullgraph=True`, `mode="max-autotune"`, a different backend, or a custom `torch._dynamo.config` setting must edit the harness. That's a barrier and a fork-risk.

## Design

### Three new flag families

**1. `--compile-kwargs JSON` — passed straight to `torch.compile()`**

```bash
--compile-kwargs '{"fullgraph": true, "dynamic": true, "backend": "inductor", "mode": "max-autotune"}'
```

Anything `torch.compile()` accepts. JSON for type fidelity (true/false/int/str).

**2. `--dynamo-config KEY=VAL` (repeatable) — sets `torch._dynamo.config.<KEY> = <VAL>`**

```bash
--dynamo-config recompile_limit=128 --dynamo-config inline_inbuilt_nn_modules=true
```

Value parsed as JSON literal. Same pattern available for inductor: `--inductor-config KEY=VAL`.

**3. `--setup-script PATH` — run user code before each compile**

```bash
--setup-script ~/configs/animesh-logging-suppress.py
```

The script is `exec()`'d in the worker process before each model compiles. This is how Animesh's logging-suppression snippet (a multi-line `for ... add()` block) gets in. More general than key=val.

### Run-naming + output convention

**`--run-name SLUG`** — defaults `--output-dir` to `sweep_results/experiments/<slug>-YYYY-MM-DD/` and tags every result row's metadata with `run_name`.

Example:
```bash
run_sweep --run-name animesh-fullgraph \
  --compile-kwargs '{"fullgraph": true, "dynamic": true}' \
  --setup-script configs/logging-suppress.py
# → outputs to sweep_results/experiments/animesh-fullgraph-2026-04-28/
```

`--output-dir` (existing) still wins if both are given. The nightly cron stays on its current `--output-dir sweep_results/nightly/<date>/` convention — no change.

### Schema impact

`fullgraph=True` makes every break a hard failure; `graph_break_count` becomes meaningless. Two options:

- **Option α** (preferred): keep schema, add `compile_status` enum (`"success"`, `"graph_break"`, `"hard_failure"`, `"timeout"`) and `first_error_reason: str | None`. In the existing graceful-degradation mode, `compile_status="success"` for all (with `graph_break_count` being the signal). In hard-fail mode, `compile_status` distinguishes outcomes.

- **Option β**: keep `graph_break_count` semantically same; on hard-failure set it to `null` and surface error in `first_error_reason`. Simpler but loses signal granularity.

I lean α — it's the cleaner long-term shape and downstream tools (`file_issues.py`) can branch on `compile_status` cleanly.

### Reporting impact

`tools/file_issues.py sweep-report` currently assumes graceful-degradation data. Two changes:

- Refuse to run on experimental sweeps (`run_name != null`) by default — these don't update official issue tracker
- Add `--allow-experimental` opt-in for users who want the issue-bucket math against an experimental sweep's data

### Backward compatibility

Default behavior unchanged when no new flags are passed:
- Empty `--compile-kwargs` → today's `{"backend": "eager", "dynamic": <inherited>}` defaults
- No `--dynamo-config` / `--inductor-config` → no overrides
- No `--setup-script` → no preamble
- No `--run-name` → output dir = today's behavior

Nightly cron config stays bit-for-bit identical.

## Implementation plan

| Phase | Work | Agent time | Notes |
|---|---|---|---|
| **P1** | Plumb 4 new flags through CLI → worker subprocess (env vars or CLI passthrough) | 30 min | Mostly mechanical |
| **P2** | Implement `compile_status` + `first_error_reason` in worker; update result schema + JSON writers | 30 min | Schema change, needs care |
| **P3** | Implement `--setup-script` exec hook (sandbox concerns: just exec; users own what they pass) | 15 min | Trust user input by design |
| **P4** | Update `file_issues.py sweep-report` to handle new schema gracefully + refuse on experimental runs | 20 min | Backward compat |
| **P5** | Gate-0 smoke test: 3 models, baseline mode (no flags) → numbers identical to today | 15 min | Regression check |
| **P6** | Gate-0 smoke test: 3 models, Animesh's config (fullgraph + logging-suppress) → expected hard-fails | 20 min | Forward check |
| **P7** | Commit + push | 5 min | |
| **P8** | Launch `--run-name animesh-fullgraph` sweep on new torch (post-#181552) | 5 min active, ~3-5h wall | Background, nohup |
| **P9** | Per-issue walk + repros (next session) | 3-4h | Tomorrow morning |

**Total active tonight: ~2.5h** (P1-P7), then sweep runs overnight, P9 tomorrow morning.

## Open questions

1. **Schema option α vs β?** I lean α (richer, future-proof). Confirm.
2. **Output convention.** `sweep_results/experiments/<slug>-<date>/` — OK with you, or should experiments live elsewhere (e.g., `experiments/<slug>/sweep/`)? I'm reusing `sweep_results/` for one-stop discovery.
3. **`--setup-script` security model.** It's `exec()`. Anyone with write access to the script can run arbitrary code in the worker. That's fine for a single-user/internal corpus, but worth flagging for any future open-source release. OK to proceed with no sandbox?
4. **Issue #70 scope.** Once this redesign lands, #70 becomes "use the new flags + run + walk issues". Should I split #70 into two issues — one for the harness redesign, one for Animesh's specific run — or keep them bundled?

---

## Parallel track (already running)

- `~/envs/torch-181552/` venv created (mirrors current nightly's deps minus torch)
- `pip install` in progress (PID 1971588)
- PyTorch source build queued behind it (wrapper PID 1972636)
- Logs: `/tmp/torch-181552-pip.log`, `/tmp/torch-181552-build.log`
- ETA: ~30-45 min total → ready by ~23:15 ET
- New venv keeps cron's `~/envs/torch-nightly-cu126` untouched
