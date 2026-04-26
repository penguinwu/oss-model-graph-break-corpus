#!/usr/bin/env python3
"""Unified front-end for model graph break testing.

Subcommands:
  Experiments (config-driven):
    template    Generate a starter experiment config
    validate    Validate a config file
    run         Run an experiment from a config file
    merge       Merge incremental results

  Corpus:
    corpus      Build/update corpus from experiment results

  Sweep (production workflow):
    sweep       Two-pass graph break sweep (identify + explain)
    explain     Explain-only pass from prior identify results
    validate-shapes   Two-shape correctness check

  Nightly:
    nightly     Automated nightly regression sweep (full workflow)

  Utilities:
    selftest        Smoke test (3 models, both passes)
    check-env       Pre-sweep environment validation
    refresh-nightly Upgrade nightly venv to latest PyTorch nightly

Usage:
  # Config-driven experiment
  python run_experiment.py run experiments/configs/my-test.json

  # Full production sweep
  python run_experiment.py sweep --source hf diffusers custom

  # Build corpus from experiment results
  python run_experiment.py corpus experiments/results/my-run/

  # Explain-only pass
  python run_experiment.py explain sweep_results/identify_results.json
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add sweep/ to path for orchestrator imports
REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "sweep"
sys.path.insert(0, str(SWEEP_DIR))

from orchestrator import run_pass, load_checkpoint, log_versions


# ── Known torch.compile() kwargs (for validation) ────────────────────────────
KNOWN_COMPILE_KWARGS = {
    "fullgraph",
    "dynamic",
    "backend",
    "mode",
    "options",
    "disable",
}

# ── Known dynamo config flags (for validation) ──────────────────────────────
KNOWN_DYNAMO_FLAGS = {
    "capture_scalar_outputs",
    "capture_dynamic_output_shape_ops",
    "automatic_dynamic_shapes",
    "assume_static_by_default",
    "specialize_int",
    "suppress_errors",
    "verbose",
    "cache_size_limit",
    "accumulated_cache_size_limit",
    "guard_nn_modules",
    "inline_inbuilt_nn_modules",
    "optimize_ddp",
}


def generate_template():
    """Print a starter experiment config with documentation."""
    template = {
        "_comment": "Experiment config — see experiments/README.md for field reference",
        "name": "my-experiment",
        "description": "Describe what you're testing and why",
        "models": {
            "_comment": "Model selection. source options: 'list', 'all', 'corpus_filter', 'sample'",
            "source": "list",
            "names": [
                "GPT2Model",
                "DistilBertModel",
                "ViTModel",
            ],
        },
        "configs": [
            {
                "name": "baseline",
                "compile_kwargs": {},
                "dynamo_flags": {},
                "_comment_compile_kwargs": "torch.compile() args: backend, fullgraph, mode, etc. Default: {backend: 'eager', fullgraph: true}",
            },
            {
                "name": "my_flag",
                "compile_kwargs": {},
                "dynamo_flags": {
                    "capture_scalar_outputs": True,
                },
            },
        ],
        "settings": {
            "device": "cuda",
            "modes": ["eval"],
            "workers": 4,
            "timeout_s": 180,
            "pass_num": 1,
            "_comment_pass": "1=identify (fullgraph check), 2=explain (graph break analysis)",
        },
    }
    print(json.dumps(template, indent=2))


def validate_config(config, strict=True):
    """Validate an experiment config. Returns list of errors (empty = valid)."""
    errors = []

    # Required top-level fields
    for field in ("name", "models", "configs", "settings"):
        if field not in config:
            errors.append(f"Missing required field: '{field}'")

    if errors:
        return errors  # Can't validate further without these

    # Validate models
    models = config["models"]
    if "source" not in models:
        errors.append("models.source is required")
    else:
        source = models["source"]
        if isinstance(source, list):
            # List of suite names
            valid_suites = {"hf", "diffusers", "custom", "timm"}
            for s in source:
                if s not in valid_suites:
                    errors.append(f"Unknown suite '{s}' in models.source — "
                                 f"valid suites: {', '.join(sorted(valid_suites))}")
        elif isinstance(source, str):
            valid_sources = {"list", "all", "corpus_filter", "sample", "new_since"}
            if source not in valid_sources:
                errors.append(f"Unknown models.source '{source}' — "
                             f"valid options: {', '.join(sorted(valid_sources))}")
            if source == "list" and "names" not in models:
                errors.append("models.names is required when source='list'")
            if source == "corpus_filter" and "status" not in models:
                errors.append("models.status is required when source='corpus_filter'")
            if source == "sample":
                if "size" not in models:
                    errors.append("models.size is required when source='sample'")
            if source == "new_since" and "baseline" not in models:
                errors.append("models.baseline is required when source='new_since'")
        else:
            errors.append("models.source must be a string or list of suite names")

    # Validate configs
    configs = config["configs"]
    if not isinstance(configs, list) or len(configs) == 0:
        errors.append("configs must be a non-empty list")
    else:
        config_names = set()
        for i, cfg in enumerate(configs):
            if "name" not in cfg:
                errors.append(f"configs[{i}].name is required")
            else:
                if cfg["name"] in config_names:
                    errors.append(f"Duplicate config name: '{cfg['name']}'")
                config_names.add(cfg["name"])

            # Validate compile_kwargs
            ckw = cfg.get("compile_kwargs", {})
            if strict:
                for key in ckw:
                    if key not in KNOWN_COMPILE_KWARGS:
                        import difflib
                        close = difflib.get_close_matches(key, KNOWN_COMPILE_KWARGS, n=1)
                        suggestion = f" — did you mean '{close[0]}'?" if close else ""
                        errors.append(f"Unknown compile kwarg '{key}' in "
                                     f"configs[{i}]{suggestion}")

            # Validate dynamo flags
            flags = cfg.get("dynamo_flags", {})
            if strict:
                for flag_name in flags:
                    if flag_name not in KNOWN_DYNAMO_FLAGS:
                        import difflib
                        close = difflib.get_close_matches(flag_name, KNOWN_DYNAMO_FLAGS, n=1)
                        suggestion = f" — did you mean '{close[0]}'?" if close else ""
                        errors.append(f"Unknown dynamo flag '{flag_name}' in "
                                     f"configs[{i}]{suggestion}")

    # Validate settings
    settings = config.get("settings", {})
    device = settings.get("device", "cuda")
    if device not in ("cuda", "cpu"):
        errors.append(f"settings.device must be 'cuda' or 'cpu', got '{device}'")

    modes = settings.get("modes", ["eval"])
    valid_modes = {"eval", "train"}
    for mode in modes:
        if mode not in valid_modes:
            errors.append(f"Unknown mode '{mode}' — valid options: eval, train")

    workers = settings.get("workers", 4)
    if not isinstance(workers, int) or workers < 1:
        errors.append(f"settings.workers must be a positive integer, got {workers}")

    timeout = settings.get("timeout_s", 180)
    if not isinstance(timeout, (int, float)) or timeout < 1:
        errors.append(f"settings.timeout_s must be a positive number, got {timeout}")

    pass_num = settings.get("pass_num", 1)
    if pass_num not in (1, 2):
        errors.append(f"settings.pass_num must be 1 or 2, got {pass_num}")

    dynamic = settings.get("dynamic", "static")
    if dynamic not in ("static", "mark", "true"):
        errors.append(f"settings.dynamic must be 'static', 'mark', or 'true', "
                     f"got '{dynamic}'")

    timeout_retry = settings.get("timeout_retry_s")
    if timeout_retry is not None:
        if not isinstance(timeout_retry, (int, float)) or timeout_retry < 1:
            errors.append(f"settings.timeout_retry_s must be a positive number, "
                         f"got {timeout_retry}")

    python_bin = settings.get("python_bin")
    if python_bin is not None and not isinstance(python_bin, str):
        errors.append(f"settings.python_bin must be a string path")

    return errors


def _enumerate_all_sources(filter_unconfigured=True):
    """Enumerate models from all sources (hf + diffusers + custom)."""
    sys.path.insert(0, str(SWEEP_DIR))
    from models import enumerate_hf, enumerate_diffusers
    try:
        from models import enumerate_custom
    except ImportError:
        enumerate_custom = lambda: []

    specs = enumerate_hf()
    diff_models = enumerate_diffusers()
    if filter_unconfigured:
        diff_models = [m for m in diff_models if m.get("has_config", True)]
    specs += diff_models
    try:
        specs += enumerate_custom()
    except Exception:
        pass
    return specs


def resolve_models(models_config, python_bin=None):
    """Resolve model selection config into a list of model specs.

    Returns list of dicts with at minimum {"name": ..., "source": ...}.
    """
    source = models_config["source"]

    # Suite list: ["hf", "diffusers", "custom"]
    if isinstance(source, list):
        sys.path.insert(0, str(SWEEP_DIR))
        from models import enumerate_hf, enumerate_diffusers
        try:
            from models import enumerate_custom
        except ImportError:
            enumerate_custom = lambda: []
        try:
            from models import enumerate_timm
        except ImportError:
            enumerate_timm = lambda: []

        suite_map = {
            "hf": enumerate_hf,
            "diffusers": lambda: [m for m in enumerate_diffusers()
                                  if m.get("has_config", True)],
            "custom": enumerate_custom,
            "timm": enumerate_timm,
        }
        specs = []
        for suite_name in source:
            fn = suite_map.get(suite_name)
            if fn:
                try:
                    specs.extend(fn())
                except Exception as e:
                    print(f"WARNING: Failed to enumerate {suite_name}: {e}")
        return specs

    if source == "list":
        # Explicit model list — enumerate all (including unconfigured) and filter
        names = set(models_config["names"])
        all_specs = _enumerate_all_sources(filter_unconfigured=False)
        specs = [s for s in all_specs if s["name"] in names]
        found = {s["name"] for s in specs}
        missing = names - found
        if missing:
            print(f"WARNING: {len(missing)} models not found in enumeration: "
                  f"{', '.join(sorted(missing)[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")
        return specs

    elif source == "all":
        return _enumerate_all_sources()

    elif source == "corpus_filter":
        status_filter = models_config["status"]
        corpus_path = REPO_ROOT / "corpus" / "corpus.json"
        if not corpus_path.exists():
            print(f"ERROR: Corpus not found at {corpus_path}")
            sys.exit(1)
        with open(corpus_path) as f:
            corpus = json.load(f)

        matching_names = set()
        for m in corpus.get("models", []):
            for mode_key in ("eval", "train"):
                md = m.get(mode_key, {})
                if md.get("status") == status_filter:
                    matching_names.add(m["name"])

        all_specs = _enumerate_all_sources()
        return [s for s in all_specs if s["name"] in matching_names]

    elif source == "sample":
        size = models_config["size"]
        seed = models_config.get("seed", 42)
        strategy = models_config.get("strategy", "random")

        # Get the base population
        base_config = models_config.get("from", {"source": "all"})
        if "status" in models_config:
            base_config = {"source": "corpus_filter", "status": models_config["status"]}
        all_specs = resolve_models(base_config, python_bin)

        import random
        rng = random.Random(seed)
        if strategy == "random":
            return rng.sample(all_specs, min(size, len(all_specs)))
        else:
            # Stratified — caller should provide pre-built sample
            return rng.sample(all_specs, min(size, len(all_specs)))

    elif source == "new_since":
        baseline_dir = Path(models_config["baseline"])
        all_specs = _enumerate_all_sources()
        all_names = {s["name"] for s in all_specs}

        # Load baseline results to find existing models
        existing_names = set()
        for results_file in baseline_dir.glob("*.jsonl"):
            with open(results_file) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        existing_names.add(r.get("name", r.get("model")))
                    except (json.JSONDecodeError, KeyError):
                        pass
        # Also check JSON results
        for results_file in baseline_dir.glob("*results*.json"):
            with open(results_file) as f:
                data = json.load(f)
                results = data if isinstance(data, list) else data.get("results", [])
                for r in results:
                    existing_names.add(r.get("name", r.get("model", "")))

        new_names = all_names - existing_names
        return [s for s in all_specs if s["name"] in new_names]

    else:
        print(f"ERROR: Unknown model source '{source}'")
        sys.exit(1)


def run_experiment(config, args):
    """Run an experiment from a validated config."""
    name = config["name"]
    settings = config.get("settings", {})
    device = settings.get("device", "cuda")
    modes = settings.get("modes", ["eval"])
    workers = settings.get("workers", 4)
    timeout_s = settings.get("timeout_s", 180)
    timeout_retry_s = settings.get("timeout_retry_s")
    pass_num = settings.get("pass_num", 1)
    dynamic = settings.get("dynamic", "static")

    # CLI overrides take precedence over config
    if hasattr(args, "workers") and args.workers is not None:
        workers = args.workers
    if hasattr(args, "timeout") and args.timeout is not None:
        timeout_s = args.timeout

    # Resolve python binary: CLI env var > config > system python
    python_bin = os.environ.get("SWEEP_PYTHON",
                                settings.get("python_bin", sys.executable))

    # Set up output directory
    resume_dir = getattr(args, "resume", None)
    output_override = getattr(args, "output", None)

    if resume_dir:
        # Resume writes to the same directory as the prior run
        output_dir = Path(resume_dir).resolve()
    elif output_override:
        output_dir = Path(output_override).resolve()
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = REPO_ROOT / "experiments" / "results" / f"{name}-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve models
    print(f"\nResolving models...")
    specs = resolve_models(config["models"], python_bin)

    configs = config["configs"]
    total_work = len(specs) * len(configs) * len(modes)

    print(f"{'=' * 70}")
    print(f"EXPERIMENT: {name}")
    print(f"  {config.get('description', '')}")
    print(f"{'=' * 70}")
    print(f"\nConfig:")
    print(json.dumps(config, indent=2))
    print()
    print(f"  Models: {len(specs)}")
    print(f"  Configs: {len(configs)}")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Total work items: {total_work}")
    print(f"  Device: {device}, Workers: {workers}, Timeout: {timeout_s}s, "
          f"Pass: {pass_num}, Dynamic: {dynamic}"
          + (f", Retry timeout: {timeout_retry_s}s" if timeout_retry_s else ""))
    # Rough ETA: ~15s per work item on average, divided by workers
    eta_s = (total_work * 15) / max(workers, 1)
    eta_min = eta_s / 60
    if eta_min > 60:
        print(f"  Estimated time: ~{eta_min / 60:.1f} hours")
    else:
        print(f"  Estimated time: ~{eta_min:.0f} minutes")

    # Dry-run: show what would run, then exit
    if getattr(args, "dry_run", False):
        print(f"\n--- DRY RUN --- (no work will be executed)")
        print(f"\nModels ({len(specs)}):")
        for s in specs:
            print(f"  - {s['name']} ({s.get('source', '?')})")
        print(f"\nConfigs ({len(configs)}):")
        for cfg in configs:
            flags = cfg.get("dynamo_flags", {})
            flag_str = json.dumps(flags) if flags else "(none)"
            print(f"  - {cfg['name']}: {flag_str}")
        print(f"\nOutput would go to: {output_dir}")
        return

    # Capture environment
    version_info = log_versions(python_bin)

    # Save resolved config (input + environment)
    resolved_config = {
        "experiment": config,
        "resolved": {
            "models": [s["name"] for s in specs],
            "model_count": len(specs),
        },
        "environment": version_info or {},
        "execution": {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_bin": python_bin,
        },
    }

    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(resolved_config, f, indent=2)
    print(f"\nConfig saved to {config_file}")

    # Results file (JSONL — one line per model/config/mode)
    results_file = output_dir / "results.jsonl"

    # Load existing results for resume
    resume_from = {}
    if resume_dir and results_file.exists():
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    model_name = r.get("name", r.get("model"))
                    key = (model_name, r["config"], r.get("mode", "eval"))
                    resume_from[key] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(resume_from)} results from prior run")

    # Open results file for appending
    results_fh = open(results_file, "a")

    # Signal handler: flush and mark interrupted on SIGTERM/SIGINT
    interrupted = False

    def _handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        results_fh.flush()
        results_fh.close()
        marker = output_dir / "INTERRUPTED"
        marker.write_text(
            f"Interrupted by signal {signum} at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
            f"Results written: {len(all_results)}\n"
        )
        print(f"\n  Interrupted (signal {signum}). "
              f"{len(all_results)} results saved to {results_file}")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    experiment_start = time.perf_counter()
    all_results = list(resume_from.values())

    def _make_experiment_result(r, cfg_name, dynamo_flags, compile_kwargs=None, retry=False):
        """Transform raw orchestrator result into experiment result."""
        experiment_result = {
            "name": r["name"],
            "source": r.get("source", "unknown"),
            "config": cfg_name,
            "mode": r.get("mode", "eval"),
            "status": r["status"],
            "wall_time_s": r.get("wall_time_s", 0),
        }
        for key in ("graph_count", "graph_break_count", "error",
                     "break_reasons", "ops_per_graph", "compile_times",
                     "create_time_s", "eager_time_s", "compile_time_s",
                     "gpu_mem_mb", "fullgraph_ok", "compile_kwargs",
                     "error_type"):
            if key in r:
                experiment_result[key] = r[key]
        if dynamo_flags:
            experiment_result["dynamo_flags"] = dynamo_flags
        if compile_kwargs:
            experiment_result.setdefault("compile_kwargs", compile_kwargs)
        if retry:
            experiment_result["retry"] = True
        return experiment_result

    def _write_result(experiment_result):
        """Write a single result to results file (streamed on each worker finish)."""
        line = json.dumps(experiment_result)
        results_fh.write(line + "\n")
        results_fh.flush()

    try:
        for cfg in configs:
            cfg_name = cfg["name"]
            dynamo_flags = cfg.get("dynamo_flags", {})
            compile_kwargs = cfg.get("compile_kwargs", {})

            print(f"\n{'─' * 70}")
            print(f"CONFIG: {cfg_name}")
            if compile_kwargs:
                print(f"  Compile: {json.dumps(compile_kwargs)}")
            if dynamo_flags:
                print(f"  Flags: {json.dumps(dynamo_flags)}")
            print(f"{'─' * 70}")

            # Build extra worker args for dynamo flags and compile kwargs
            extra_args = []
            if dynamo_flags:
                extra_args.extend(["--dynamo-flags", json.dumps(dynamo_flags)])
            if compile_kwargs:
                extra_args.extend(["--compile-kwargs", json.dumps(compile_kwargs)])

            # Filter out already-completed models for this config
            pending_specs = []
            for spec in specs:
                for mode in modes:
                    key = (spec["name"], cfg_name, mode)
                    if key not in resume_from:
                        if spec not in pending_specs:
                            pending_specs.append(spec)
                        break

            if not pending_specs:
                print(f"  All models already completed for this config")
                continue

            # Streaming callback: write each result as it arrives
            timed_out_specs = []

            def _on_result(r, _cfg=cfg_name, _flags=dynamo_flags, _ckw=compile_kwargs):
                exp_result = _make_experiment_result(r, _cfg, _flags, compile_kwargs=_ckw)
                _write_result(exp_result)
                all_results.append(exp_result)
                if r["status"] == "timeout":
                    spec = next((s for s in specs if s["name"] == r["name"]), None)
                    if spec and spec not in timed_out_specs:
                        timed_out_specs.append(spec)

            # Run via orchestrator with streaming callback
            dynamic_arg = dynamic if dynamic != "static" else False
            run_pass(
                python_bin, pending_specs, pass_num=pass_num,
                device=device, modes=modes, workers=workers,
                timeout_s=timeout_s, dynamic=dynamic_arg,
                extra_worker_args=extra_args,
                result_callback=_on_result,
            )

            # Auto-retry timed-out models with longer timeout
            if timed_out_specs and timeout_retry_s:
                print(f"\n  Retrying {len(timed_out_specs)} timed-out models "
                      f"with {timeout_retry_s}s timeout...")

                def _on_retry_result(r, _cfg=cfg_name, _flags=dynamo_flags, _ckw=compile_kwargs):
                    exp_result = _make_experiment_result(r, _cfg, _flags, compile_kwargs=_ckw, retry=True)
                    _write_result(exp_result)
                    for i, prev in enumerate(all_results):
                        prev_name = prev.get("name", prev.get("model"))
                        if (prev_name == exp_result["name"]
                                and prev["config"] == _cfg
                                and prev.get("mode") == exp_result.get("mode")):
                            all_results[i] = exp_result
                            break

                run_pass(
                    python_bin, timed_out_specs, pass_num=pass_num,
                    device=device, modes=modes, workers=workers,
                    timeout_s=timeout_retry_s, dynamic=dynamic_arg,
                    extra_worker_args=extra_args,
                    result_callback=_on_retry_result,
                )

    finally:
        if not interrupted:
            results_fh.close()
            # Deduplicate results file (retries append duplicates; keep last entry per key)
            seen = {}
            for r in all_results:
                key = (r.get("name", r.get("model")), r["config"], r.get("mode", "eval"))
                seen[key] = r
            with open(results_file, "w") as f:
                for r in seen.values():
                    f.write(json.dumps(r) + "\n")

    experiment_time = time.perf_counter() - experiment_start

    # Update config with execution end time
    resolved_config["execution"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    resolved_config["execution"]["duration_s"] = round(experiment_time, 1)
    resolved_config["execution"]["total_results"] = len(all_results)
    with open(config_file, "w") as f:
        json.dump(resolved_config, f, indent=2)

    # Generate summary
    _generate_summary(all_results, config, output_dir, experiment_time)

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE: {name}")
    print(f"  Duration: {experiment_time:.1f}s")
    print(f"  Results: {len(all_results)} entries")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")


def _generate_summary(results, config, output_dir, duration):
    """Generate a human-readable summary.md from experiment results."""
    name = config["name"]
    description = config.get("description", "")
    configs = config["configs"]

    # Group results by config
    by_config = {}
    for r in results:
        cfg_name = r.get("config", "default")
        by_config.setdefault(cfg_name, []).append(r)

    lines = [
        f"# {name}",
        "",
        f"{description}",
        "",
        f"**Duration:** {duration:.1f}s",
        f"**Models:** {len(set(r.get('name', r.get('model')) for r in results))}",
        f"**Configs:** {len(configs)}",
        "",
    ]

    for cfg in configs:
        cfg_name = cfg["name"]
        cfg_results = by_config.get(cfg_name, [])

        # Count by status
        by_status = {}
        for r in cfg_results:
            s = r.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        lines.append(f"## {cfg_name}")
        if cfg.get("dynamo_flags"):
            lines.append(f"Flags: `{json.dumps(cfg['dynamo_flags'])}`")
        lines.append("")

        for status, count in sorted(by_status.items()):
            lines.append(f"- {status}: {count}")
        lines.append("")

    # Cross-config comparison (if multiple configs)
    if len(configs) > 1:
        lines.append("## Comparison")
        lines.append("")

        baseline_name = configs[0]["name"]
        baseline_results = {(r.get("name", r.get("model")), r.get("mode", "eval")): r
                           for r in by_config.get(baseline_name, [])}

        for cfg in configs[1:]:
            cfg_name = cfg["name"]
            cfg_results = by_config.get(cfg_name, [])

            improvements = []
            regressions = []
            crashes = []

            for r in cfg_results:
                key = (r.get("name", r.get("model")), r.get("mode", "eval"))
                baseline = baseline_results.get(key)
                if not baseline:
                    continue

                b_status = baseline.get("status")
                r_status = r.get("status")
                b_graphs = baseline.get("graph_count", 0)
                r_graphs = r.get("graph_count", 0)

                if b_status in ("graph_break",) and r_status in ("graph_break",):
                    if r_graphs < b_graphs:
                        improvements.append(
                            f"{r.get('name', r.get('model'))}: {b_graphs} → {r_graphs} graphs")
                    elif r_graphs > b_graphs:
                        regressions.append(
                            f"{r.get('name', r.get('model'))}: {b_graphs} → {r_graphs} graphs")
                elif b_status in ("full_graph", "graph_break") and r_status not in ("full_graph", "graph_break"):
                    crashes.append(
                        f"{r.get('name', r.get('model'))}: {r_status} — {r.get('error', '')[:100]}")

            lines.append(f"### {cfg_name} vs {baseline_name}")
            lines.append("")
            if improvements:
                lines.append(f"**Improvements ({len(improvements)}):**")
                for imp in improvements:
                    lines.append(f"- {imp}")
                lines.append("")
            if regressions:
                lines.append(f"**Regressions ({len(regressions)}):**")
                for reg in regressions:
                    lines.append(f"- {reg}")
                lines.append("")
            if crashes:
                lines.append(f"**Crashes ({len(crashes)}):**")
                for crash in crashes:
                    lines.append(f"- {crash}")
                lines.append("")
            if not improvements and not regressions and not crashes:
                lines.append("No differences detected.")
                lines.append("")

    summary_path = output_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary saved to {summary_path}")


def _import_sweep_module():
    """Import run_sweep module from sweep/ directory."""
    sys.path.insert(0, str(SWEEP_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_sweep", SWEEP_DIR / "run_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_sweep_command(args):
    """Run a two-pass sweep (identify + explain) on enumerated models."""
    sweep_mod = _import_sweep_module()
    sweep_mod.run_sweep(args)


def run_explain_command(args):
    """Run explain-only pass from prior identify results."""
    sweep_mod = _import_sweep_module()
    sweep_mod.run_explain(args)


def run_validate_shapes_command(args):
    """Run two-shape correctness validation on clean models."""
    sweep_mod = _import_sweep_module()
    sweep_mod.run_validation(args)


def run_correctness_command(args):
    """Run Phase 3 correctness sweep (eager vs compiled forward output comparison)."""
    sweep_mod = _import_sweep_module()
    sweep_mod.run_correctness(args)


def run_selftest_command(args):
    """Run integration smoke test on a few models."""
    sweep_mod = _import_sweep_module()
    sweep_mod.run_test_mode(args)


def run_check_env_command(args):
    """Pre-sweep environment validation."""
    sweep_mod = _import_sweep_module()
    sweep_mod.check_env(args)


def run_refresh_nightly(args):
    """Upgrade nightly venv to the latest PyTorch nightly build.

    Tries CUDA variants in order (cu128, cu126) and falls back if primary
    has no newer build. This handles periods where one variant's CI is broken.
    """
    venv_dir = Path(args.venv).expanduser().resolve()
    pip = venv_dir / "bin" / "pip"
    python = venv_dir / "bin" / "python"

    if not pip.exists():
        print(f"ERROR: {pip} not found. Is --venv correct?")
        sys.exit(1)

    # Show current version
    result = subprocess.run(
        [str(python), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    old_version = result.stdout.strip() if result.returncode == 0 else "unknown"
    print(f"Current nightly: {old_version}")

    # cu126 fallback removed 2026-04-26: pypi.nvidia.com (cuda-toolkit dep) is
    # BPF-blocked for agent:claude_code identity. Restore once Tiger gets the
    # allowlist filed via 3PAI.
    cuda_variants = [args.cuda_variant]
    packages = ["torch", "torchvision", "torchaudio"]
    installed = False

    for variant in cuda_variants:
        index_url = f"https://download.pytorch.org/whl/nightly/{variant}"
        print(f"\nTrying {variant}: {', '.join(packages)}")
        print(f"Index: {index_url}")
        cmd = [str(pip), "install", "--pre", "--upgrade",
               "--index-url", index_url] + packages
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"WARNING: {variant} install failed")
            continue

        # Check if version actually changed
        check = subprocess.run(
            [str(python), "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        new_version = check.stdout.strip() if check.returncode == 0 else "unknown"

        if new_version != old_version:
            print(f"\nUpdated via {variant}: {old_version} → {new_version}")
            installed = True
            break
        elif variant != cuda_variants[-1]:
            print(f"  {variant}: no newer build (still {new_version}), trying next variant...")
        else:
            print(f"  {variant}: no newer build available")
            installed = True  # pip succeeded, just no update
            break

    if not installed:
        print("ERROR: All CUDA variants failed")
        sys.exit(1)

    if args.with_transformers:
        print("\nUpgrading transformers + diffusers...")
        result = subprocess.run(
            [str(pip), "install", "--upgrade", "transformers", "diffusers"])
        if result.returncode != 0:
            print("WARNING: transformers/diffusers upgrade failed")

    # Show final version
    result = subprocess.run(
        [str(python), "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    new_version = result.stdout.strip() if result.returncode == 0 else "unknown"
    print(f"\nDone: {old_version} → {new_version}")
    if old_version == new_version:
        print("  (no update available on any variant)")


CORPUS_PATH = REPO_ROOT / "corpus" / "corpus.json"
TRACKED_MODELS_FILE = REPO_ROOT / "sweep" / "tracked_models.json"


def _nightly_build_age_days(version_str):
    """Days since this nightly was built, from version like '2.12.0.dev20260407+cu128'.

    Returns None for non-nightly versions (source builds, releases).
    """
    m = re.search(r'dev(\d{8})', version_str)
    if not m:
        return None
    try:
        build_date = datetime.strptime(m.group(1), '%Y%m%d')
        return (datetime.now() - build_date).days
    except ValueError:
        return None


def _get_torch_git_version(python):
    """Get torch git commit hash from a python executable."""
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.version.git_version)"],
        capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _get_last_sweep_git_version():
    """Get git_version from the most recent nightly sweep."""
    nightly_dir = REPO_ROOT / "sweep_results" / "nightly"
    if not nightly_dir.exists():
        return None
    dates = sorted(d.name for d in nightly_dir.iterdir()
                   if d.is_dir() and d.name[:4].isdigit())
    if not dates:
        return None
    last = nightly_dir / dates[-1] / "versions.json"
    if not last.exists():
        return None
    with open(last) as f:
        data = json.load(f)
    return data.get("git_version") or data.get("torch_git")


def pipeline_preflight(python, device="cuda"):
    """Fast environment validation — catches 80% of failures in <30 seconds."""
    print(f"\n{'=' * 70}")
    print("PIPELINE: Preflight checks")
    print(f"{'=' * 70}")
    errors = []

    if not Path(python).exists():
        errors.append(f"Python not found: {python}")
    else:
        print(f"  OK: python {python}")

    result = subprocess.run(
        [python, "-c",
         "import torch; import torch._dynamo; "
         "print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.device_count()}')"],
        capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        errors.append(f"torch import failed: {result.stderr.strip()[:200]}")
    else:
        parts = result.stdout.strip().split("|")
        print(f"  OK: torch {parts[0]}, cuda={parts[1]}, devices={parts[2]}")
        if device == "cuda" and parts[1] != "True":
            errors.append("CUDA requested but torch.cuda.is_available() is False")
        if device == "cuda" and parts[2] == "0":
            errors.append("CUDA requested but no GPU devices found")

    result = subprocess.run(
        [python, "-c", "import transformers; print(transformers.__version__)"],
        capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        errors.append(f"transformers import failed: {result.stderr.strip()[:200]}")
    else:
        print(f"  OK: transformers {result.stdout.strip()}")

    if not CORPUS_PATH.exists():
        errors.append(f"Corpus not found: {CORPUS_PATH}")
    else:
        try:
            with open(CORPUS_PATH) as f:
                corpus = json.load(f)
            print(f"  OK: corpus ({len(corpus.get('models', []))} models)")
        except (json.JSONDecodeError, KeyError) as e:
            errors.append(f"Corpus parse error: {e}")

    sweep_script = REPO_ROOT / "sweep" / "run_sweep.py"
    if not sweep_script.exists():
        errors.append(f"Sweep script not found: {sweep_script}")
    else:
        print(f"  OK: sweep script")

    import shutil
    free_gb = shutil.disk_usage(str(REPO_ROOT)).free / (1024**3)
    if free_gb < 5:
        errors.append(f"Low disk: {free_gb:.1f}GB free (need 5GB)")
    else:
        print(f"  OK: {free_gb:.0f}GB free")

    if device == "cuda":
        result = subprocess.run(
            [python, "-c",
             "import torch; t = torch.randn(1000, 1000, device='cuda'); "
             "free, total = torch.cuda.mem_get_info(); "
             f"print(f'{{free/1e9:.1f}}GB free / {{total/1e9:.1f}}GB total')"],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            errors.append(f"GPU allocation failed: {result.stderr.strip()[:200]}")
        else:
            print(f"  OK: GPU {result.stdout.strip()}")

    if errors:
        print(f"\n  PREFLIGHT FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    ✗ {e}")
        return False
    print("  PREFLIGHT PASSED")
    return True


def pipeline_canary(python, output_dir, device="cuda", timeout=180):
    """Run 1 model end-to-end as a gate before the full sweep (~30s vs ~3h)."""
    print(f"\n{'=' * 70}")
    print("PIPELINE: Canary sweep (1 model)")
    print(f"{'=' * 70}")

    canary_dir = Path(output_dir) / "canary"
    canary_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, str(REPO_ROOT / "sweep" / "run_sweep.py"), "sweep",
        "--device", device,
        "--workers", "1",
        "--timeout", str(timeout),
        "--output-dir", str(canary_dir),
        "--identify-only",
        "--source", "hf",
        "--limit", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(REPO_ROOT), timeout=timeout + 60)
    if result.returncode != 0:
        print(f"  Canary exit code: {result.returncode}")
        if result.stderr:
            print(f"  stderr: {result.stderr[-500:]}")
        print("  CANARY FAILED — sweep infrastructure broken")
        return False

    results_file = canary_dir / "identify_results.json"
    if not results_file.exists():
        print("  CANARY FAILED — no results file produced")
        return False

    with open(results_file) as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        print("  CANARY FAILED — results file empty")
        return False

    print(f"  OK: {results[0].get('name', '?')} → {results[0].get('status', '?')}")
    print("  CANARY PASSED")
    return True


def pipeline_staleness_check(python, force=False):
    """Check if nightly build has changed since last sweep.

    Returns (git_version, is_stale). Stale = same git hash as last sweep.
    """
    git_ver = _get_torch_git_version(python)
    last_git = _get_last_sweep_git_version()

    if not last_git:
        print(f"  No prior sweep found — treating as fresh (git: {git_ver[:12]})")
        return git_ver, False

    if git_ver == last_git:
        print(f"\n  *** STALE NIGHTLY DETECTED ***")
        print(f"  Git commit {git_ver[:12]} is identical to last sweep.")
        if force:
            print(f"  --force specified — sweeping anyway.")
            return git_ver, False
        else:
            print(f"  Aborting to avoid wasting a sweep cycle.")
            return git_ver, True

    print(f"  Last sweep git: {last_git[:12]} → current: {git_ver[:12]} (new commits)")
    return git_ver, False


def pipeline_detect_and_explain(python, results_dir, device="cuda",
                                workers=4, timeout=180):
    """Detect status changes vs corpus, run explain on changed + tracked models."""
    print(f"\n{'=' * 70}")
    print("PIPELINE: Detect changes + targeted explain")
    print(f"{'=' * 70}")

    identify_file = Path(results_dir) / "identify_results.json"
    if not identify_file.exists():
        print(f"  No identify results — skipping explain")
        return

    with open(identify_file) as f:
        identify_data = json.load(f)
    new_results = {(r["name"], r["mode"]): r["status"]
                   for r in identify_data.get("results", [])}

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    old_results = {}
    for m in corpus["models"]:
        for mode in ("eval", "train"):
            status = m.get(mode, {}).get("status")
            if status:
                old_results[(m["name"], mode)] = status

    changed = set()
    for key, new_status in new_results.items():
        old_status = old_results.get(key)
        if old_status and old_status != new_status:
            changed.add(key[0])
    for key in new_results:
        if key not in old_results:
            changed.add(key[0])

    tracked = set()
    if TRACKED_MODELS_FILE.exists():
        with open(TRACKED_MODELS_FILE) as f:
            tracked = set(json.load(f).get("models", []))

    explain_set = changed | tracked
    explain_models = [r for r in identify_data.get("results", [])
                      if r["name"] in explain_set
                      and r.get("status") in ("graph_break", "compile_error")]

    print(f"  Status changes: {len(changed)} models")
    if changed:
        for name in sorted(changed)[:10]:
            old = old_results.get((name, "eval"), "?")
            new = new_results.get((name, "eval"), "?")
            print(f"    {name}: {old} → {new}")
        if len(changed) > 10:
            print(f"    ... and {len(changed) - 10} more")
    print(f"  Tracked models: {len(tracked)}")
    print(f"  Models needing explain: {len(explain_models)}")

    if not explain_models:
        print("  No models need explain — skipping")
        return

    explain_input = Path(results_dir) / "explain_targets.json"
    explain_data = {"metadata": identify_data.get("metadata", {}),
                    "results": explain_models}
    with open(explain_input, "w") as f:
        json.dump(explain_data, f, indent=2)

    explain_dir = Path(results_dir) / "explain"
    explain_dir.mkdir(exist_ok=True)

    cmd = [
        python, str(REPO_ROOT / "tools" / "run_experiment.py"), "explain",
        str(explain_input),
        "--device", device,
        "--workers", str(workers),
        "--timeout", str(timeout),
        "--output-dir", str(explain_dir),
    ]

    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        print(f"  Explain done in {elapsed:.0f}s")
    else:
        print(f"  Explain failed after {elapsed:.0f}s")


def pipeline_generate_summary(results_dir, nightly_version, summary_file):
    """Generate a human-readable nightly summary."""
    identify_file = Path(results_dir) / "identify_results.json"
    if not identify_file.exists():
        return

    with open(identify_file) as f:
        data = json.load(f)
    results = data.get("results", [])

    from collections import Counter
    status_counts = Counter(r.get("status", "unknown") for r in results)

    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    old_results = {}
    for m in corpus["models"]:
        for mode in ("eval", "train"):
            status = m.get(mode, {}).get("status")
            if status:
                old_results[(m["name"], mode)] = status

    regressions, fixes = [], []
    for r in results:
        key = (r["name"], r["mode"])
        old = old_results.get(key)
        new = r.get("status")
        if not old or old == new:
            continue
        if old == "full_graph" and new != "full_graph":
            regressions.append((r["name"], r["mode"], old, new))
        elif old != "full_graph" and new == "full_graph":
            fixes.append((r["name"], r["mode"], old, new))

    lines = [
        f"Nightly Sweep Summary — {time.strftime('%Y-%m-%d')}",
        f"PyTorch: {nightly_version}",
        f"Models tested: {len(results)}",
        "",
        "Status breakdown:",
    ]
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {status}: {count}")

    if fixes:
        lines.append(f"\nFixes ({len(fixes)}):")
        for name, mode, old, new in fixes:
            lines.append(f"  {name} ({mode}): {old} → {new}")
    if regressions:
        lines.append(f"\nRegressions ({len(regressions)}):")
        for name, mode, old, new in regressions:
            lines.append(f"  {name} ({mode}): {old} → {new}")
    if not fixes and not regressions:
        lines.append("\nNo status changes from corpus.")

    Path(summary_file).write_text("\n".join(lines) + "\n")


def run_nightly_command(args):
    """Nightly regression sweep — the full automated workflow.

    refresh → staleness check → preflight → canary → sweep →
    detect+explain → corpus → scan → summary
    """
    python = sys.executable
    tools_dir = Path(__file__).resolve().parent
    script = str(tools_dir / "run_experiment.py")
    repo_root = tools_dir.parent

    steps_done = []
    label = args.label or time.strftime("nightly-%Y%m%d")
    venv_dir = Path(args.venv).expanduser().resolve()
    sweep_python = str(venv_dir / "bin" / "python")

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = repo_root / "sweep_results" / "nightly" / time.strftime("%Y-%m-%d")
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run(cmd, desc, allow_fail=False):
        print(f"\n{'='*70}")
        print(f"NIGHTLY: {desc}")
        print(f"{'='*70}")
        print(f"  CMD: {' '.join(str(c) for c in cmd)}\n")
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0 and not allow_fail:
            print(f"\nNIGHTLY FAILED at step: {desc}")
            print(f"  Exit code: {result.returncode}")
            if steps_done:
                print(f"  Completed steps: {', '.join(steps_done)}")
            sys.exit(1)
        steps_done.append(desc)

    print(f"{'='*70}")
    print(f"Nightly sweep — {label}")
    print(f"{'='*70}")
    print(f"  Python: {sweep_python}")
    print(f"  Output: {output_dir}")
    print(f"  Start:  {time.strftime('%H:%M:%S')}")

    # ── Refresh venv ──────────────────────────────────────────────────
    _run([python, script, "refresh-nightly",
          "--venv", str(venv_dir), "--with-transformers"],
         "Refresh nightly venv")

    # ── Version + source build fallback ─────────────────────────────
    ver_result = subprocess.run(
        [sweep_python, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True)
    nightly_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "unknown"

    age_days = _nightly_build_age_days(nightly_version)
    if age_days is not None and age_days > 3:
        print(f"\n  *** PIP NIGHTLY IS STALE ({age_days} days old: {nightly_version}) ***")
        build_script = repo_root / "scripts" / "build-nightly-from-source.sh"
        if not build_script.exists():
            print(f"\n=== Nightly aborted: pip nightly stale and build script missing ===")
            print(f"  Build script expected at: {build_script}")
            print(f"  A sweep on stale pytorch produces no regression signal — aborting.")
            sys.exit(1)
        print(f"  Attempting build from source...")
        build_result = subprocess.run(
            ["bash", str(build_script), str(venv_dir)],
            cwd=str(repo_root))
        if build_result.returncode != 0:
            print(f"\n=== Nightly aborted: source build failed (exit {build_result.returncode}) ===")
            print(f"  Pip nightly was {age_days} days old; source build was the only path to fresh pytorch.")
            print(f"  A sweep on stale pytorch produces no regression signal — aborting.")
            sys.exit(1)
        ver_result = subprocess.run(
            [sweep_python, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        nightly_version = ver_result.stdout.strip() if ver_result.returncode == 0 else "unknown"
        print(f"  Source build succeeded: {nightly_version}")
        steps_done.append("Source build (pip stale)")
    elif age_days is not None:
        print(f"  Pip nightly is fresh ({age_days} days old)")

    # ── Staleness check ───────────────────────────────────────────────
    git_version, is_stale = pipeline_staleness_check(
        sweep_python, force=getattr(args, "force", False))
    if is_stale:
        print(f"\n=== Nightly aborted: build unchanged since last sweep ===")
        sys.exit(0)
    steps_done.append("Staleness check")

    # Save version info
    (output_dir / "versions.json").write_text(json.dumps({
        "pytorch": nightly_version,
        "git_version": git_version,
        "date": time.strftime("%Y-%m-%d"),
        "python": sweep_python,
    }, indent=2))

    # ── Preflight ─────────────────────────────────────────────────────
    device = getattr(args, "device", "cuda")
    if not pipeline_preflight(sweep_python, device=device):
        print(f"\n=== Nightly aborted: preflight failed ===")
        sys.exit(1)
    steps_done.append("Preflight")

    # ── Canary ────────────────────────────────────────────────────────
    if not pipeline_canary(sweep_python, output_dir, device=device,
                           timeout=args.timeout):
        print(f"\n=== Nightly aborted: canary failed ===")
        sys.exit(1)
    steps_done.append("Canary")

    # ── Sweep ─────────────────────────────────────────────────────────
    sweep_cmd = [sweep_python, script, "sweep",
                 "--source"] + args.source + [
                 "--modes"] + args.modes + [
                 "--workers", str(args.workers),
                 "--timeout", str(args.timeout),
                 "--output-dir", str(output_dir)]
    if args.identify_only:
        sweep_cmd.append("--identify-only")
    if args.resume:
        sweep_cmd.append("--resume")

    _run(sweep_cmd, "Sweep (identify + explain)")

    # Find results directory
    candidates = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name != "canary"],
        key=lambda p: p.stat().st_mtime)
    results_dir = str(candidates[-1]) if candidates else str(output_dir)
    if not Path(results_dir).exists():
        print(f"NIGHTLY: Could not find sweep results in {output_dir}")
        sys.exit(1)

    # ── Targeted explain on changed + tracked models ──────────────────
    pipeline_detect_and_explain(sweep_python, results_dir, device=device,
                                workers=args.workers, timeout=args.timeout)

    # ── Corpus update + scan ──────────────────────────────────────────
    _run_post_sweep(python, script, repo_root, results_dir, label, steps_done)

    # ── Summary ───────────────────────────────────────────────────────
    summary_file = Path(results_dir) / "summary.txt"
    pipeline_generate_summary(results_dir, nightly_version, summary_file)
    if summary_file.exists():
        print(f"\n{summary_file.read_text()}")


def _run_post_sweep(python, script, repo_root, results_dir, label, steps_done):
    """Run post-sweep pipeline steps: corpus → scan → summary."""
    tools_dir = Path(script).parent

    def _run(cmd, desc, allow_fail=False):
        print(f"\n{'='*70}")
        print(f"PIPELINE: {desc}")
        print(f"{'='*70}")
        print(f"  CMD: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0 and not allow_fail:
            print(f"\nPIPELINE FAILED at step: {desc}")
            sys.exit(1)
        steps_done.append(desc)

    # Step 3: corpus update
    _run([python, script, "corpus", results_dir], "Corpus update")

    # Step 4: git diff to show changes
    print(f"\n{'='*70}")
    print("PIPELINE: Corpus diff")
    print(f"{'='*70}")
    diff_result = subprocess.run(
        ["git", "diff", "--stat", "corpus/corpus.json"],
        cwd=str(repo_root), capture_output=True, text=True)
    if diff_result.stdout.strip():
        print(diff_result.stdout)
    else:
        print("  No changes to corpus.")
    steps_done.append("Corpus diff")

    # Step 5: issue report (sweep-report generates a reviewable plan)
    file_issues = str(tools_dir / "file_issues.py")
    if Path(file_issues).exists():
        explain_file = str(Path(results_dir) / "explain_results.json")
        identify_file = str(Path(results_dir) / "identify_results.json")
        if Path(explain_file).exists() and Path(identify_file).exists():
            _run([python, file_issues, "sweep-report",
                  "--explain", explain_file,
                  "--identify", identify_file],
                 "Issue report", allow_fail=True)

    # Step 6: summary
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Label: {label}")
    print(f"  Steps: {', '.join(steps_done)}")
    print(f"  Results: {results_dir}")
    changelog = Path(results_dir) / "changelog.md"
    if changelog.exists():
        print(f"  Changelog: {changelog}")
    print(f"\nNext: review changelog, then post to feedback space.")


def build_corpus(args):
    """Build/update corpus from experiment results.

    Bridges the experiment result format (results.jsonl with 'model' field)
    to the corpus format expected by update_corpus.py.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from update_corpus import (
        update_corpus as uc_update, format_changelog, index_by_name_mode,
        update_corpus_explain_only, CORPUS_PATH,
    )

    results_dir = Path(args.results_dir).resolve()
    results_file = results_dir / "results.jsonl"
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)

    # Load and normalize results
    raw_results = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "model" in r and "name" not in r:
                    r["name"] = r.pop("model")
                raw_results.append(r)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(raw_results)} results from {results_file}")

    # Load config.json for version info
    config_file = results_dir / "config.json"
    versions = {}
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        env = cfg.get("environment", {})
        versions = env.get("versions", env)

    # Determine pass type from experiment config
    is_explain_experiment = False
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        exp_settings = cfg.get("experiment", {}).get("settings", {})
        if exp_settings.get("pass_num") == 2:
            is_explain_experiment = True

    # Split into identify (pass 1) and explain (pass 2) results
    identify_results = []
    explain_results = []
    if is_explain_experiment:
        explain_results = raw_results
    else:
        identify_results = raw_results

    # Also handle --explain flag: load explain results from a separate directory
    if args.explain:
        explain_dir = Path(args.explain).resolve()
        explain_file = explain_dir / "results.jsonl"
        if not explain_file.exists():
            print(f"ERROR: {explain_file} not found")
            sys.exit(1)
        with open(explain_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "model" in r and "name" not in r:
                        r["name"] = r.pop("model")
                    explain_results.append(r)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(explain_results)} explain results (including from {explain_dir})")

    identify_idx = index_by_name_mode(identify_results) if identify_results else {}
    explain_idx = index_by_name_mode(explain_results) if explain_results else {}

    print(f"  Identify entries: {len(identify_idx)}")
    print(f"  Explain entries: {len(explain_idx)}")

    # Load corpus
    corpus_path = Path(CORPUS_PATH).resolve()
    output_dir = Path(args.output).resolve() if args.output else results_dir

    print(f"\nLoading corpus from {corpus_path}")
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"  {len(corpus['models'])} models")

    # Determine mode and run update
    if not identify_idx and explain_idx:
        # Explain-only mode
        print("\nMode: EXPLAIN-ONLY (no identify results)")
        corpus, changelog, health = update_corpus_explain_only(corpus, explain_idx)
        pre_merge_names = None
    else:
        print(f"\nMode: {'REPLACE' if args.replace else 'OVERLAY'}")
        corpus, changelog, pre_merge_names, health = uc_update(
            corpus, identify_idx, explain_idx, versions, replace=args.replace)

    # Print summary
    status_changes = [c for c in changelog if c["type"] == "status_change"]
    new_models = [c for c in changelog if c["type"] == "new_model"]
    explain_updated = [c for c in changelog if c["type"] == "explain_updated"]
    print(f"\nChanges: {len(changelog)} total")
    if status_changes:
        print(f"  Status changes: {len(status_changes)}")
    if new_models:
        print(f"  New models: {len(set(c['name'] for c in new_models))}")
    if explain_updated:
        print(f"  Explain updated: {len(explain_updated)}")

    if health:
        missing = health.get("missing_explain", [])
        if missing:
            print(f"\n  Warning: {len(set(n for n, m in missing))} models missing explain data")

    if args.dry_run:
        print("\n--- DRY RUN — no files written ---")
        return

    # Write corpus
    print(f"\nWriting corpus to {corpus_path}")
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")

    # Write changelog
    output_dir.mkdir(parents=True, exist_ok=True)
    changelog_path = output_dir / "changelog.md"
    if not identify_idx and explain_idx:
        # Simple explain-only changelog
        lines = [
            "# Explain-Only Merge Changelog\n",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            f"**Source:** `{results_dir}`",
            f"**Entries updated:** {len(explain_updated)}",
            "",
        ]
        with open(changelog_path, "w") as f:
            f.write("\n".join(lines) + "\n")
    else:
        changelog_text = format_changelog(
            changelog, str(results_dir), versions, corpus, pre_merge_names, health)
        with open(changelog_path, "w") as f:
            f.write(changelog_text)

    print(f"Changelog written to {changelog_path}")
    print("Done.")


def merge_results(source_dir, target_dir):
    """Merge incremental experiment results into an existing result set.

    Results are merged at the JSONL line level. If a (model, config, mode)
    entry exists in both source and target, source wins (newer data).
    The merge is idempotent — running it twice produces the same result.
    """
    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir).resolve()

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)
    if not target_dir.exists():
        print(f"ERROR: Target directory not found: {target_dir}")
        sys.exit(1)

    source_results = source_dir / "results.jsonl"
    target_results = target_dir / "results.jsonl"

    if not source_results.exists():
        print(f"ERROR: No results.jsonl in source: {source_dir}")
        sys.exit(1)
    if not target_results.exists():
        print(f"ERROR: No results.jsonl in target: {target_dir}")
        sys.exit(1)

    # Load target results indexed by (model, config, mode)
    target_index = {}
    with open(target_results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r.get("name", r.get("model")), r.get("config", "default"), r.get("mode", "eval"))
                target_index[key] = r
            except (json.JSONDecodeError, KeyError):
                continue

    target_count = len(target_index)

    # Load source results — these override target on conflict
    source_count = 0
    overrides = 0
    additions = 0
    with open(source_results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r.get("name", r.get("model")), r.get("config", "default"), r.get("mode", "eval"))
                if key in target_index:
                    overrides += 1
                else:
                    additions += 1
                target_index[key] = r
                source_count += 1
            except (json.JSONDecodeError, KeyError):
                continue

    # Write merged results
    with open(target_results, "w") as f:
        for r in target_index.values():
            f.write(json.dumps(r) + "\n")

    # Update target config.json with merge metadata
    target_config_file = target_dir / "config.json"
    if target_config_file.exists():
        with open(target_config_file) as f:
            target_config = json.load(f)

        merges = target_config.get("merges", [])
        merges.append({
            "source": str(source_dir),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "source_results": source_count,
            "overrides": overrides,
            "additions": additions,
        })
        target_config["merges"] = merges
        target_config["execution"]["total_results"] = len(target_index)

        with open(target_config_file, "w") as f:
            json.dump(target_config, f, indent=2)

    print(f"Merge complete:")
    print(f"  Source: {source_count} results from {source_dir.name}")
    print(f"  Target: {target_count} existing results")
    print(f"  Additions: {additions} new entries")
    print(f"  Overrides: {overrides} updated entries")
    print(f"  Total: {len(target_index)} results in {target_results}")


def main():
    parser = argparse.ArgumentParser(
        description="Config-driven experiment runner for model graph break testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── template ──────────────────────────────────────────────────────────
    sub_template = subparsers.add_parser(
        "template",
        help="Print a starter experiment config to stdout",
    )

    # ── validate ──────────────────────────────────────────────────────────
    sub_validate = subparsers.add_parser(
        "validate",
        help="Validate a config file and exit",
    )
    sub_validate.add_argument("config", metavar="CONFIG",
                              help="Path to JSON config file")

    # ── run ────────────────────────────────────────────────────────────────
    sub_run = subparsers.add_parser(
        "run",
        help="Run an experiment from a config file",
    )
    sub_run.add_argument("config", metavar="CONFIG",
                         help="Path to JSON config file")
    sub_run.add_argument("--output", metavar="DIR",
                         help="Output directory (default: experiments/results/<name>-<timestamp>/)")
    sub_run.add_argument("--resume", metavar="DIR",
                         help="Resume from a prior run's output directory "
                              "(reads checkpoint, writes to same dir)")
    sub_run.add_argument("--dry-run", action="store_true",
                         help="Resolve models and show work items without running")
    sub_run.add_argument("--workers", type=int, metavar="N",
                         help="Override worker count from config")
    sub_run.add_argument("--timeout", type=int, metavar="N",
                         help="Override timeout (seconds) from config")

    # ── merge ──────────────────────────────────────────────────────────────
    sub_merge = subparsers.add_parser(
        "merge",
        help="Merge incremental results into an existing result set",
    )
    # Note: argparse doesn't allow --from since "from" is a Python keyword,
    # so we use dest= to map it cleanly
    sub_merge.add_argument("--from", dest="merge_from", metavar="DIR", required=True,
                           help="Source results directory")
    sub_merge.add_argument("--into", dest="merge_into", metavar="DIR", required=True,
                           help="Target results directory (updated in-place)")

    # ── corpus ─────────────────────────────────────────────────────────────
    sub_corpus = subparsers.add_parser(
        "corpus",
        help="Build/update corpus from experiment results (identify + explain)",
        description="Converts experiment results into corpus format and generates "
                    "a changelog. Wraps update_corpus.py logic.",
    )
    sub_corpus.add_argument("results_dir", metavar="RESULTS_DIR",
                            help="Experiment results directory (must contain results.jsonl)")
    sub_corpus.add_argument("--output", metavar="DIR",
                            help="Output directory for corpus.json and changelog.md "
                                 "(default: same as results_dir)")
    sub_corpus.add_argument("--replace", action="store_true",
                            help="Full replacement: results become entire corpus "
                                 "(models not in results are dropped)")
    sub_corpus.add_argument("--dry-run", action="store_true",
                            help="Show what would change without writing")
    sub_corpus.add_argument("--explain", metavar="DIR",
                            help="Merge explain results from a separate experiment run")

    # ── sweep ─────────────────────────────────────────────────────────────
    DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "sweep_results")

    sub_sweep = subparsers.add_parser(
        "sweep",
        help="Two-pass graph break sweep (identify + explain)",
        description="Run the full two-pass sweep: identify (fullgraph=True check) "
                    "then explain (detailed break analysis on broken models).",
    )
    sweep_input = sub_sweep.add_mutually_exclusive_group()
    sweep_input.add_argument("--source", nargs="+",
                             default=["hf", "diffusers", "custom"],
                             choices=["timm", "hf", "diffusers", "custom", "all"],
                             help="Model libraries to enumerate (default: hf diffusers custom)")
    sweep_input.add_argument("--models",
                             help="JSON file with explicit model list")
    sub_sweep.add_argument("--stability", choices=["stable", "unstable"],
                           help="Filter by corpus stability")
    sub_sweep.add_argument("--limit", type=int,
                           help="Max models to test")
    sub_sweep.add_argument("--modes", nargs="+", default=["eval", "train"],
                           choices=["eval", "train"],
                           help="Modes to run (default: eval train)")
    sub_sweep.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    sub_sweep.add_argument("--workers", type=int, default=4)
    sub_sweep.add_argument("--timeout", type=int, default=180,
                           help="Per-model timeout in seconds (default: 180)")
    sub_sweep.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sub_sweep.add_argument("--resume", action="store_true",
                           help="Resume from checkpoint")
    sub_sweep.add_argument("--dynamic-dim", choices=["batch", "all"],
                           help="Dynamic shapes mode")
    sub_sweep.add_argument("--no-auto-retry", action="store_true",
                           help="Skip auto-retry of timed-out/errored models")
    sub_sweep.add_argument("--identify-only", action="store_true",
                           help="Stop after identify pass (skip explain)")

    # ── explain ────────────────────────────────────────────────────────────
    sub_explain = subparsers.add_parser(
        "explain",
        help="Explain-only pass from prior identify results",
        description="Run the explain pass on broken models from a prior identify sweep.",
    )
    sub_explain.add_argument("file", metavar="FILE",
                             help="Path to identify results JSON")
    sub_explain.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    sub_explain.add_argument("--workers", type=int, default=4)
    sub_explain.add_argument("--timeout", type=int, default=180)
    sub_explain.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sub_explain.add_argument("--resume", action="store_true")

    # ── validate-shapes ───────────────────────────────────────────────────
    sub_valshapes = subparsers.add_parser(
        "validate-shapes",
        help="Two-shape correctness check on clean models",
        description="Compile models with two different input shapes, "
                    "compare outputs against eager mode.",
    )
    valshapes_input = sub_valshapes.add_mutually_exclusive_group()
    valshapes_input.add_argument("--from", dest="from_file",
                                 help="Identify results JSON — validates full_graph models")
    valshapes_input.add_argument("--models",
                                 help="JSON file with explicit model list")
    sub_valshapes.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    sub_valshapes.add_argument("--workers", type=int, default=4)
    sub_valshapes.add_argument("--timeout", type=int, default=180)
    sub_valshapes.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sub_valshapes.add_argument("--resume", action="store_true")
    sub_valshapes.add_argument("--dynamic-dim", choices=["batch", "all"],
                               default="all",
                               help="Dynamic shapes (default: all)")
    sub_valshapes.add_argument("--limit", type=int)

    # ── correctness (Phase 3) ─────────────────────────────────────────────
    sub_correct = subparsers.add_parser(
        "correctness",
        help="Eager vs compiled forward output comparison (Phase 3)",
        description="Phase 3: compare eager and compiled forward outputs on clean models. "
                    "Defaults to corpus.json HF eval fullgraph_ok subset. "
                    "Surfaces compiler-introduced numerical errors — every divergence is signal.",
    )
    correct_input = sub_correct.add_mutually_exclusive_group()
    correct_input.add_argument("--from", dest="from_file",
                               help="Source JSON (corpus.json or identify_results.json)")
    correct_input.add_argument("--models",
                               help="JSON file with explicit model list")
    sub_correct.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    sub_correct.add_argument("--workers", type=int, default=4)
    sub_correct.add_argument("--timeout", type=int, default=180)
    sub_correct.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    sub_correct.add_argument("--resume", action="store_true")
    sub_correct.add_argument("--limit", type=int)

    # ── nightly ───────────────────────────────────────────────────────────
    sub_nightly = subparsers.add_parser(
        "nightly",
        help="Automated nightly regression sweep (full workflow)",
        description="refresh → staleness check → preflight → canary → sweep → "
                    "detect+explain → corpus → scan → summary. "
                    "One command, fully automated.",
    )
    sub_nightly.add_argument("--venv", default="~/envs/torch-nightly",
                             help="Nightly venv path (default: ~/envs/torch-nightly)")
    sub_nightly.add_argument("--label", metavar="NAME",
                             help="Run label (default: nightly-YYYYMMDD)")
    sub_nightly.add_argument("--source", nargs="+",
                             default=["hf", "diffusers", "custom"],
                             choices=["timm", "hf", "diffusers", "custom", "all"],
                             help="Model sources (default: hf diffusers custom)")
    sub_nightly.add_argument("--modes", nargs="+", default=["eval", "train"],
                             choices=["eval", "train"])
    sub_nightly.add_argument("--workers", type=int, default=4)
    sub_nightly.add_argument("--timeout", type=int, default=180)
    sub_nightly.add_argument("--output-dir", metavar="DIR",
                             help="Output dir (default: sweep_results/nightly/YYYY-MM-DD)")
    sub_nightly.add_argument("--identify-only", action="store_true",
                             help="Stop after identify pass (skip explain)")
    sub_nightly.add_argument("--resume", action="store_true",
                             help="Resume sweep from checkpoint")
    sub_nightly.add_argument("--force", action="store_true",
                             help="Run even if nightly build is unchanged")
    sub_nightly.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    # ── selftest ──────────────────────────────────────────────────────────
    sub_selftest = subparsers.add_parser(
        "selftest",
        help="Smoke test: 3 models, both passes, validate output",
    )
    sub_selftest.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    # ── check-env ─────────────────────────────────────────────────────────
    sub_checkenv = subparsers.add_parser(
        "check-env",
        help="Pre-sweep environment validation (version check)",
    )
    sub_checkenv.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    # ── refresh-nightly ──────────────────────────────────────────────────
    sub_refresh = subparsers.add_parser(
        "refresh-nightly",
        help="Upgrade nightly venv to latest PyTorch nightly build",
    )
    sub_refresh.add_argument("--venv", default="~/envs/torch-nightly",
                             help="Path to nightly venv (default: ~/envs/torch-nightly)")
    sub_refresh.add_argument("--with-transformers", action="store_true",
                             help="Also upgrade transformers and diffusers")
    sub_refresh.add_argument("--cuda-variant", default="cu128",
                             help="Primary CUDA variant (default: cu128, falls back to cu126)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "template":
        generate_template()
        return

    if args.command == "validate":
        config = _load_config(args.config)
        errors = validate_config(config)
        if errors:
            print("Validation FAILED:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)
        else:
            print("Validation PASSED")
            sys.exit(0)

    if args.command == "merge":
        merge_results(args.merge_from, args.merge_into)
        return

    if args.command == "corpus":
        build_corpus(args)
        return

    if args.command == "run":
        if args.output and args.resume:
            print("ERROR: --output and --resume are mutually exclusive. "
                  "--resume writes to the prior run's directory.")
            sys.exit(1)

        config = _load_config(args.config)
        errors = validate_config(config)
        if errors:
            print("Config validation FAILED:")
            for err in errors:
                print(f"  - {err}")
            sys.exit(1)

        run_experiment(config, args)
        return

    if args.command == "sweep":
        run_sweep_command(args)
        return

    if args.command == "explain":
        run_explain_command(args)
        return

    if args.command == "validate-shapes":
        run_validate_shapes_command(args)
        return

    if args.command == "correctness":
        run_correctness_command(args)
        return

    if args.command == "selftest":
        run_selftest_command(args)
        return

    if args.command == "check-env":
        run_check_env_command(args)
        return

    if args.command == "refresh-nightly":
        run_refresh_nightly(args)
        return

    if args.command == "nightly":
        run_nightly_command(args)
        return


def _load_config(path):
    """Load and return a JSON config file, exiting on errors."""
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
