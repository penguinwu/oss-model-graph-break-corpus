# Running Sweeps

A sweep compiles every model in the corpus with `torch.compile(fullgraph=True)` and records the result. Use sweeps to measure compile quality for a specific PyTorch version or validate a compiler change.

## Quick start

```bash
# Set up a venv with torch + model libraries
python3 -m venv ~/envs/torch-test
source ~/envs/torch-test/bin/activate
pip install -r requirements.txt

# Run a sweep
python3 sweep/run_sweep.py sweep \
    --source hf diffusers custom \
    --workers 4 \
    --timeout 180
```

Or use the unified CLI (equivalent):

```bash
python3 tools/run_experiment.py sweep \
    --source hf diffusers custom
```

## Smoke test on a random subset

Don't want to run the full corpus? Use the experiment config with a `sample` source — pick N random models with a deterministic seed, so the same N models are sampled every time. Useful for fast validation of a PyTorch build, a custom backend, or compiler flags before committing to a full sweep.

```bash
# 1) Write a one-off config file
cat > /tmp/quick-sample.json <<'EOF'
{
  "name": "quick-sample",
  "description": "Random 20-model sample to validate a build",
  "models": {"source": "sample", "size": 20, "seed": 42},
  "configs": [{"name": "baseline", "compile_kwargs": {"fullgraph": true}}]
}
EOF

# 2) Run it
python3 tools/run_experiment.py experiment /tmp/quick-sample.json \
    --python ~/envs/torch-test/bin/python
```

Same `seed` → same 20 models, every time. Bump `size` (e.g. `50`, `100`) for tighter signal, drop it for a faster smoke. See [Running Experiments](running-experiments.md) for the full config schema and other model sources (`list`, `corpus_filter`, `new_since`).

## Two-pass architecture

Sweeps run in two passes:

1. **Identify** — compile each model with `fullgraph=True`, record pass/fail/error
2. **Explain** — for graph-breaking models, re-run with Dynamo logging to extract break reasons, locations, and counts

By default, `sweep` runs both passes. To run identify-only (faster):

```bash
python3 sweep/run_sweep.py sweep --identify-only \
    --source hf diffusers custom
```

## Using a specific Python binary

Point `--python` at your venv's Python to avoid activating it:

```bash
python3 sweep/run_sweep.py sweep \
    --python ~/envs/torch-test/bin/python \
    --source hf diffusers custom \
    --workers 4
```

Or set `SWEEP_PYTHON`:

```bash
SWEEP_PYTHON=~/envs/torch-test/bin/python \
    python3 sweep/run_sweep.py sweep --source hf diffusers custom
```

## Model sources

| Source | What it includes |
|--------|-----------------|
| `hf` | HuggingFace Transformers models (base + ForCausalLM + ForConditionalGeneration) |
| `diffusers` | HuggingFace Diffusers pipelines |
| `custom` | Custom model definitions in `sweep/custom_models/` |
| `timm` | PyTorch Image Models (requires explicit `--source timm`) |

Default sources: `hf diffusers custom`. TIMM requires explicit request.

## Dynamic shape testing

```bash
# Mark batch + sequence length dims as dynamic
python3 sweep/run_sweep.py sweep --dynamic mark --source hf diffusers custom

# All dims symbolic
python3 sweep/run_sweep.py sweep --dynamic true --source hf diffusers custom
```

## Correctness testing (Phase 3)

A separate `correctness` pass compares eager-mode outputs against compiled outputs on the same inputs (same seed, same shape) and surfaces numerical divergences introduced by the compiler. Runs only on models marked `fullgraph_ok` in the corpus — there's no point comparing outputs if compilation already failed.

```bash
python3 tools/run_experiment.py correctness \
    --workers 4
```

Output goes to `correctness/correctness_results.json`. Each entry records `status` (`match` / `divergence` / `nan_inf` / `shape_mismatch`), `max_diff`, `severity_ratio` (max_diff / atol), `compared_fields`, `skipped_fields`, and `first_divergence` (which output field diverged first).

Tolerance: HF-style fp32 atol=1e-6, rtol=1e-4. Recursive walker over `ModelOutput` float fields; integer/boolean/0-dim/None/Cache fields are skipped. Each divergence is a data point to file or explain — there is no "acceptance threshold." Sort by `severity_ratio` descending to triage filing order.

Methodology and design rationale: `design/design-doc.md` Section 8.

## Crash recovery

Sweeps checkpoint after every model. To resume after a crash:

```bash
python3 sweep/run_sweep.py sweep --resume --source hf diffusers custom
```

The `--resume` flag reads the checkpoint file and skips already-completed models.

## Output

Results go to `sweep_results/<label>/` (default: `sweep_results/nightly/<date>/` for cron, `sweep_results/experiments/<run-name>-<date>/` for `--run-name` runs):

```
sweep_results/baseline/pt2.11/
  identify_results.json          — pass/fail/error for each model
  explain_results.json           — graph break details for breaking models
  identify_checkpoint.jsonl      — incremental checkpoint for the identify pass
  explain_checkpoint.jsonl       — incremental checkpoint for the explain pass
  correctness_checkpoint.jsonl   — incremental checkpoint for the correctness pass (when run)
```

## Monitoring long sweeps

For full sweeps (~790 models × eval+train ≈ 1500 work items, ~4–6 hours on A100):

```bash
python3 sweep/sweep_watchdog.py
```

The watchdog monitors progress and auto-restarts on failures.

## Update the corpus after a sweep

```bash
python3 tools/update_corpus.py \
    --identify sweep_results/baseline/pt2.11/identify_results.json \
    --explain sweep_results/baseline/pt2.11/explain_results.json
```

Validate after updating:

```bash
python3 tools/validate.py
```

## Customizing the compile config

The sweep CLI accepts custom `torch.compile()` configuration directly. No need to spin up an experiment for one-off compiler tests:

| Flag | Purpose | Example |
|---|---|---|
| `--compile-kwargs JSON` | passed to `torch.compile()` | `--compile-kwargs '{"fullgraph": true, "dynamic": true, "backend": "inductor"}'` |
| `--dynamo-config KEY=VAL` (repeatable) | sets `torch._dynamo.config.<key>` | `--dynamo-config recompile_limit=128` |
| `--inductor-config KEY=VAL` (repeatable) | sets `torch._inductor.config.<key>` | `--inductor-config max_autotune=true` |
| `--setup-script PATH` | Python file `exec()`'d in each worker before compile | `--setup-script configs/my-prep.py` (for multi-line config that doesn't fit `KEY=VAL`) |
| `--run-name SLUG` | tags the run as experimental; defaults output to `sweep_results/experiments/<slug>-<date>/` and tags `metadata.run_name` | `--run-name my-experiment` |
| `--strict-known-errors` | exit non-zero if any unexpected `create_error` or `eager_error` appears (i.e. anything not pre-declared in `known_errors.json`) | use in CI / before shipping data downstream |

Example — fullgraph + dynamic shapes + a custom suppression patch:

```bash
python3 sweep/run_sweep.py sweep \
    --compile-kwargs '{"fullgraph": true, "dynamic": true, "backend": "eager"}' \
    --setup-script sweep/configs/animesh-logging-suppress.py \
    --run-name animesh-fullgraph
```

Defaults are bit-for-bit identical when no compile-config flags are passed — the cron baseline is unaffected.

**Issue tracker safety.** `tools/file_issues.py sweep-report` and `sweep-update` refuse to operate on plans tagged with `run_name` (i.e. experimental sweeps). Pass `--allow-experimental` to override; the issue tracker is normally fed only by the official cron baseline.

## Known errors gate (no silent infrastructure failures)

The sweep skips models pre-declared in [`sweep/known_errors.json`](../sweep/known_errors.json) AND validates that no NEW gated failure (currently `create_error` or `eager_error`) appears outside that list. This prevents setup/build/dep/model bugs from quietly masking real graph-break improvements or regressions.

**Workflow when a sweep flags an unexpected gated failure:**

1. The sweep prints a loud warning naming each unexpected (model, mode, status, error head).
2. Decide:
   - **Fix the underlying issue** (e.g. install a missing dep, fix a build flag, update a config) — preferred when feasible.
   - **Add an entry to `known_errors.json`** with a `reason` field if the failure is a stable known bug not worth fixing now.
3. Re-run the sweep. With `--strict-known-errors`, an unresolved gated failure causes a non-zero exit (use this in CI / before shipping data downstream).

**Entry shape:**

```json
{
  "status": "create_error",        // or "eager_error"
  "model": "ZambaForCausalLM",
  "modes": ["eval", "train"],
  "error_pattern": "tie_weights_keys",
  "added": "2026-04-28",
  "reason": "tie_weights_keys definition bug — see issue #XXX"
}
```

The validator matches `(model, status, mode, error_pattern_substring)`. To remove an entry: delete it; the next sweep re-tests the model and surfaces the new status.

## Tier-aware timeouts (large + very_large model registry)

Models recorded in [`sweep/large_models.json`](../sweep/large_models.json) get extended timeouts based on their tier:

| Tier | Timeout multiplier (vs `--timeout`) | Example models |
|---|---|---|
| (no tier; default) | 1× (the `--timeout` value) | most models — fast |
| `large` | 3× | AlignModel, ConvNextV2Model, ... |
| `very_large` | 9× | BltModel, Gemma3nModel, Gemma3nForConditionalGeneration, ... |

With default `--timeout 180`, that's 540s for `large` and 1620s for `very_large`. The launch logs the count + per-tier timeout. The registry self-grows: when a model first hits a timeout in a sweep, the orchestrator promotes it to the next tier and writes the entry back to `large_models.json`.

If a model in `very_large` still times out at 9× — that's a real PyTorch / model issue (e.g. a dynamo runaway during compile), worth filing upstream rather than just bumping the timeout further.

## Conventions

- Batch size must be >= 2 (PyTorch specializes on 0 and 1)
- Default backend is `eager` with `fullgraph=True` (tests Dynamo tracing). Override via `--compile-kwargs` (above) or run a multi-config [experiment](running-experiments.md)
- Never use 0 or 1 as input dimensions for dynamic shape testing
- Default sources: `hf diffusers custom` (TIMM/dynamic require explicit request)

## Next steps

- [Running Experiments](running-experiments.md) — test specific flag combinations or model subsets
- [Issue Management](issue-management.md) — classify graph breaks and update GitHub issues after a sweep
- [Understanding Results](understanding-results.md) — interpret sweep output
