# sweep/ — package architecture for repo developers and agents

The `sweep/` package is the core pipeline that runs models through `torch.compile` and classifies the result. Most sweep workflows are invoked via `tools/run_experiment.py` (subcommand-driven CLI); this README documents the modules behind that CLI for developers who need to modify the pipeline.

If you are adding a new sweep type, modifying status semantics, or fixing a worker bug, also read `skills/test-sweep-changes/SKILL.md` (5-gate validation workflow) and `skills/sweep_sanity_check.md` (post-completion guardrails).

---

## Module map

| Module | Role |
|---|---|
| **`run_sweep.py`** | Top-level orchestration entry. Implements `run_sweep`, `run_explain`, `run_correctness`, `check_env`, `run_test_mode`, `run_nightly`. Each is invoked from `tools/run_experiment.py`. Owns CLI arg validation, venv resolution (`_resolve_python`), output-dir setup, checkpoint resume, and result aggregation. |
| **`worker.py`** | Per-model worker subprocess. Implements `run_identify`, `run_explain`, `run_correctness` (Phase 3 hard-fullgraph version). Each takes a model spec + device + mode and returns a result row. **Hard-coded numerical comparison fields (`numeric_status`, `numeric_max_diff`, etc.) are produced inside `run_identify`** when `compile_kwargs` includes a real comparison path. |
| **`orchestrator.py`** | Worker pool manager. Spawns N workers, tracks progress, handles timeouts and worker death, writes streaming JSONL checkpoints. Used by `run_pass()` in `run_sweep.py`. |
| **`models.py`** | Model enumeration. `enumerate_hf()`, `enumerate_diffusers()`, `enumerate_timm()`, `enumerate_custom()` — each returns model specs (name + source + variant + hf_class + hf_config + input_type + constructor_args + inputs) by introspecting installed libraries. **Hydration source for sparse cohort entries** — see `run_correctness()` and `_hydrate_specs()`. |
| **`explain.py`** | Graph break analysis. `run_explain()` is invoked per model in the explain pass; produces `graph_break_count`, `graph_count`, `ops_per_graph`, `break_reasons`. |
| **`venv_setup.py`** | Pre-sweep venv provisioning when `--torch SPEC` is passed. Resolves the right cu variant venv (`~/envs/cu128/`, `~/envs/cu126/`), installs the requested torch version, and validates. Required reading for any sweep where the torch version is parameterized: see `~/.myclaw-shared/recipes/python-venv-bpf.md` for the BPF jail pattern. |
| **`results_loader.py`** | Load + normalize sweep result files (identify_results.json / explain_results.json / corpus.json) for analysis tools. Hides the streaming-jsonl vs aggregated-json schema split. |
| **`kernel_resolver.py`** | HF kernels cache helpers. Resolves model-class → kernel-class mappings. Used during model creation to validate the test exercises real HF code. |
| **`sweep_watchdog.py`** | Recurring watchdog that posts sweep progress to GChat (observation-only — does NOT auto-restart). Separate from `tools/sweep_watchdog_check.py` (one-shot decision script). Cron-installed by the watchdog setup in `skills/sweep.md`. Reads `sweep_state.json` + a streaming file (`identify_streaming.jsonl`). Both files are written by `run_sweep.py` natively AND by `tools/run_experiment.py run` (via the symlink + sweep_state extension landed 2026-05-07). |
| **`cohort_validator.py`** | Runtime validator for cohort files at sweep launch. `validate_cohort()` returns `ValidatedCohort` or raises `CohortValidationError` with stable `.code` field. Codes: `BARE_LIST_REJECTED`, `MISSING_METADATA_KEY`, `EMPTY_SOURCE_VERSIONS`, `PARTIAL_SOURCE_VERSIONS`, `VERSION_MISMATCH`, `STALE_COHORT`, `INVALID_MODELS_LIST`, `FILE_NOT_FOUND`, `INVALID_JSON`. Wired into `run_sweep.py`'s `--models` branch. Each failure code has a paired `--allow-*` CLI flag override; override use is recorded into `sweep_state.json` under `models_cohort_overrides_used`. |

## Registry files (in this directory)

| File | Role |
|---|---|
| `known_errors.json` | Models with stable known unfixed bugs that the sweep skips and validates against. See `_doc` and `_workflow` fields inside. **Strict mode** (`--strict-known-errors`) mechanically enforces "every create_error / eager_error must match a known entry." |
| `skip_models.json` | Flat list of models excluded from sweeps (timm/einops dependents, out-of-scope models). Should grow a per-entry reason field — open loop. |
| `large_models.json` | Models with wall_time_s > 120s at baseline settings. Get the 600s timeout instead of 180s. Updated by hand after sweep due-diligence; do not auto-commit changes. |
| `tracked_models.json` | Models that get extra logging / per-model tracking during sweeps. |
| `modellibs.json` | Specifies which transformers / diffusers / timm versions are pre-provisioned in `~/envs/modellibs/<lib>-<ver>/`. Consumed by `tools/bootstrap_modellibs.py`. |

## Where to look for common tasks

- **Adding a new sweep subcommand** → wire in `tools/run_experiment.py` + add `run_X` in `run_sweep.py` + add per-spec handler in `worker.py`
- **Changing how a model is created** → `models.py` (enumeration / spec) + `worker.py:create_model` (instantiation)
- **Changing what gets measured** → `worker.py:run_identify` (status + numeric fields) or `worker.py:run_explain` (break taxonomy)
- **Changing how status is classified** → `worker.py` per-pass implementations; status taxonomy summarized in `skills/sweep.md` §9
- **Adding a new known error** → edit `sweep/known_errors.json` per its `_workflow` instructions
- **Adding a new model library / version** → update `sweep/modellibs.json`, run `tools/bootstrap_modellibs.py`, then trigger a cohort-expansion run per `skills/sweep_sanity_check.md` APPLY-D
