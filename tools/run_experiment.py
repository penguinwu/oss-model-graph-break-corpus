#!/usr/bin/env python3
"""Experiment runner — config-driven model testing with self-describing results.

Run ad-hoc experiments (flag quality tests, config ablations, regression bisects)
using the same orchestrator infrastructure as the production sweep.

Usage:
  # Generate a starter config
  python run_experiment.py template > experiments/configs/my-test.json

  # Validate a config before running
  python run_experiment.py validate experiments/configs/my-test.json

  # Dry-run — resolve models, show work items, exit
  python run_experiment.py run experiments/configs/my-test.json --dry-run

  # Run an experiment
  python run_experiment.py run experiments/configs/my-test.json

  # Run with explicit output directory
  python run_experiment.py run experiments/configs/my-test.json \
      --output experiments/results/my-run/

  # Override workers/timeout from CLI (without editing config)
  python run_experiment.py run experiments/configs/my-test.json --workers 1

  # Resume an interrupted experiment
  python run_experiment.py run experiments/configs/my-test.json \
      --resume experiments/results/my-test-20260416-193000/

  # Merge incremental results into an existing result set
  python run_experiment.py merge \
      --from experiments/results/new-models/ \
      --into experiments/results/full-sweep/
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add sweep/ to path for orchestrator imports
REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = REPO_ROOT / "sweep"
sys.path.insert(0, str(SWEEP_DIR))

from orchestrator import run_pass, load_checkpoint, log_versions


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
                "dynamo_flags": {},
            },
            {
                "name": "my_flag",
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

            # Validate dynamo flags
            flags = cfg.get("dynamo_flags", {})
            if strict:
                for flag_name in flags:
                    if flag_name not in KNOWN_DYNAMO_FLAGS:
                        # Fuzzy match for common typos
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
            "diffusers": enumerate_diffusers,
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
        # Explicit model list — enumerate all and filter
        names = set(models_config["names"])
        sys.path.insert(0, str(SWEEP_DIR))
        from models import enumerate_hf, enumerate_diffusers
        try:
            from models import enumerate_custom
        except ImportError:
            enumerate_custom = lambda: []

        all_specs = enumerate_hf() + enumerate_diffusers()
        try:
            all_specs += enumerate_custom()
        except Exception:
            pass

        specs = [s for s in all_specs if s["name"] in names]
        found = {s["name"] for s in specs}
        missing = names - found
        if missing:
            print(f"WARNING: {len(missing)} models not found in enumeration: "
                  f"{', '.join(sorted(missing)[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")
        return specs

    elif source == "all":
        sys.path.insert(0, str(SWEEP_DIR))
        from models import enumerate_hf, enumerate_diffusers
        try:
            from models import enumerate_custom
        except ImportError:
            enumerate_custom = lambda: []
        specs = enumerate_hf() + enumerate_diffusers()
        try:
            specs += enumerate_custom()
        except Exception:
            pass
        return specs

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

        # Enumerate to get full specs
        sys.path.insert(0, str(SWEEP_DIR))
        from models import enumerate_hf, enumerate_diffusers
        all_specs = enumerate_hf() + enumerate_diffusers()
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
        # Find models in current enumeration that don't appear in baseline results
        sys.path.insert(0, str(SWEEP_DIR))
        from models import enumerate_hf, enumerate_diffusers
        all_specs = enumerate_hf() + enumerate_diffusers()
        all_names = {s["name"] for s in all_specs}

        # Load baseline results to find existing models
        existing_names = set()
        for results_file in baseline_dir.glob("*.jsonl"):
            with open(results_file) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        existing_names.add(r["model"])
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
    print(f"  Models: {len(specs)}")
    print(f"  Configs: {len(configs)}")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  Total work items: {total_work}")
    print(f"  Device: {device}, Workers: {workers}, Timeout: {timeout_s}s, "
          f"Pass: {pass_num}, Dynamic: {dynamic}"
          + (f", Retry timeout: {timeout_retry_s}s" if timeout_retry_s else ""))

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
    checkpoint_file = output_dir / "checkpoint.jsonl"

    # Load checkpoint for resume
    resume_from = {}
    if resume_dir and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r["model"], r["config"], r.get("mode", "eval"))
                    resume_from[key] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(resume_from)} results from checkpoint")

    # Open results file for appending
    results_fh = open(results_file, "a")
    ckpt_fh = open(checkpoint_file, "a")

    experiment_start = time.perf_counter()
    all_results = list(resume_from.values())

    try:
        for cfg in configs:
            cfg_name = cfg["name"]
            dynamo_flags = cfg.get("dynamo_flags", {})

            print(f"\n{'─' * 70}")
            print(f"CONFIG: {cfg_name}")
            if dynamo_flags:
                print(f"  Flags: {json.dumps(dynamo_flags)}")
            print(f"{'─' * 70}")

            # Build extra worker args for dynamo flags
            extra_args = []
            if dynamo_flags:
                extra_args = ["--dynamo-flags", json.dumps(dynamo_flags)]

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

            # Run via orchestrator
            dynamic_arg = dynamic if dynamic != "static" else False
            results = run_pass(
                python_bin, pending_specs, pass_num=pass_num,
                device=device, modes=modes, workers=workers,
                timeout_s=timeout_s, dynamic=dynamic_arg,
                extra_worker_args=extra_args,
            )

            # Process and save results
            timed_out_specs = []
            for r in results:
                experiment_result = {
                    "model": r["name"],
                    "config": cfg_name,
                    "mode": r.get("mode", "eval"),
                    "status": r["status"],
                    "wall_time_s": r.get("wall_time_s", 0),
                }
                # Copy relevant fields
                for key in ("graph_count", "graph_break_count", "error",
                           "break_reasons", "ops_per_graph", "compile_times"):
                    if key in r:
                        experiment_result[key] = r[key]

                if dynamo_flags:
                    experiment_result["dynamo_flags"] = dynamo_flags

                line = json.dumps(experiment_result)
                results_fh.write(line + "\n")
                results_fh.flush()
                ckpt_fh.write(line + "\n")
                ckpt_fh.flush()
                all_results.append(experiment_result)

                # Track timed-out models for retry
                if r["status"] == "timeout":
                    spec = next((s for s in specs if s["name"] == r["name"]), None)
                    if spec and spec not in timed_out_specs:
                        timed_out_specs.append(spec)

            # Auto-retry timed-out models with longer timeout
            if timed_out_specs and timeout_retry_s:
                print(f"\n  Retrying {len(timed_out_specs)} timed-out models "
                      f"with {timeout_retry_s}s timeout...")
                retry_results = run_pass(
                    python_bin, timed_out_specs, pass_num=pass_num,
                    device=device, modes=modes, workers=workers,
                    timeout_s=timeout_retry_s, dynamic=dynamic_arg,
                    extra_worker_args=extra_args,
                )
                for r in retry_results:
                    experiment_result = {
                        "model": r["name"],
                        "config": cfg_name,
                        "mode": r.get("mode", "eval"),
                        "status": r["status"],
                        "wall_time_s": r.get("wall_time_s", 0),
                        "retry": True,
                    }
                    for key in ("graph_count", "graph_break_count", "error",
                               "break_reasons", "ops_per_graph", "compile_times"):
                        if key in r:
                            experiment_result[key] = r[key]
                    if dynamo_flags:
                        experiment_result["dynamo_flags"] = dynamo_flags

                    # Overwrite the timeout entry in all_results
                    for i, prev in enumerate(all_results):
                        if (prev["model"] == experiment_result["model"]
                                and prev["config"] == cfg_name
                                and prev.get("mode") == experiment_result.get("mode")):
                            all_results[i] = experiment_result
                            break

                    line = json.dumps(experiment_result)
                    results_fh.write(line + "\n")
                    results_fh.flush()
                    ckpt_fh.write(line + "\n")
                    ckpt_fh.flush()

    finally:
        results_fh.close()
        ckpt_fh.close()

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
        f"**Models:** {len(set(r['model'] for r in results))}",
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
        baseline_results = {(r["model"], r.get("mode", "eval")): r
                           for r in by_config.get(baseline_name, [])}

        for cfg in configs[1:]:
            cfg_name = cfg["name"]
            cfg_results = by_config.get(cfg_name, [])

            improvements = []
            regressions = []
            crashes = []

            for r in cfg_results:
                key = (r["model"], r.get("mode", "eval"))
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
                            f"{r['model']}: {b_graphs} → {r_graphs} graphs")
                    elif r_graphs > b_graphs:
                        regressions.append(
                            f"{r['model']}: {b_graphs} → {r_graphs} graphs")
                elif b_status in ("full_graph", "graph_break") and r_status not in ("full_graph", "graph_break"):
                    crashes.append(
                        f"{r['model']}: {r_status} — {r.get('error', '')[:100]}")

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
                key = (r["model"], r.get("config", "default"), r.get("mode", "eval"))
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
                key = (r["model"], r.get("config", "default"), r.get("mode", "eval"))
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
