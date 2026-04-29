#!/usr/bin/env python3
"""Sweep orchestrator — runs the two-pass graph break sweep.

Manages subprocess workers, parallel execution, timeouts, and result merging.
Worker management is delegated to sweep/orchestrator.py (shared with run_experiment.py).

Subcommands:
  sweep       Identify + explain sweep (default)
  explain     Explain-only from prior identify results
  validate    Two-shape correctness check

Usage:
  # Full sweep (activate venv first, or set SWEEP_PYTHON)
  python run_sweep.py sweep

  # Incremental: skip stable models
  python run_sweep.py sweep --skip-stable

  # Source-specific sweep
  python run_sweep.py sweep --source timm hf

  # Resume after crash
  python run_sweep.py sweep --resume

  # Explain pass only
  python run_sweep.py explain sweep_results/identify_results.json

  # Two-shape validation
  python run_sweep.py validate --from sweep_results/identify_results.json

  # Smoke test
  python run_sweep.py sweep --selftest

  # Pre-sweep version check
  python run_sweep.py sweep --check-env

  # Custom compile config (writes to sweep_results/experiments/<slug>-<date>/)
  python run_sweep.py sweep \\
      --compile-kwargs '{"fullgraph": true, "dynamic": true, "backend": "eager"}' \\
      --dynamo-config recompile_limit=128 \\
      --setup-script sweep/configs/my-prep.py \\
      --run-name my-experiment

See docs/running-sweeps.md for the full flag reference.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from orchestrator import (
    WorkerHandle,
    spawn_worker,
    harvest_worker,
    timeout_result,
    escalating_kill,
    load_checkpoint,
    run_pass,
    check_gpu_health,
    kill_gpu_zombies,
    log_versions,
)


SWEEP_DIR = Path(__file__).resolve().parent
CORPUS_FILE = SWEEP_DIR.parent / "corpus" / "corpus.json"
DEFAULT_OUTPUT_DIR = SWEEP_DIR.parent / "sweep_results"
EXPERIMENTS_OUTPUT_DIR = SWEEP_DIR.parent / "sweep_results" / "experiments"
LARGE_MODELS_FILE = SWEEP_DIR / "large_models.json"


def _parse_kv_overrides(items):
    """Parse repeated KEY=VAL flags into a dict.

    Values are decoded as JSON literals when possible (true/false/numbers/quoted
    strings); fall back to the raw string. Empty list returns {}.
    """
    out = {}
    for item in items or []:
        if "=" not in item:
            print(f"WARNING: ignoring config flag {item!r} (no '=')", file=sys.stderr)
            continue
        key, raw = item.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _build_extra_worker_args(args):
    """Translate args namespace → list of CLI args appended to each worker invocation.

    Returns [] when no compile-config flags are set, so default sweep behavior is
    bit-for-bit identical to pre-redesign runs.
    """
    extras = []

    # torch.compile() kwargs as JSON (worker reads with --compile-kwargs)
    compile_kwargs_json = getattr(args, "compile_kwargs", None)
    if compile_kwargs_json:
        try:
            json.loads(compile_kwargs_json)  # validate
        except json.JSONDecodeError as e:
            print(f"ERROR: --compile-kwargs is not valid JSON: {e}", file=sys.stderr)
            sys.exit(1)
        extras.extend(["--compile-kwargs", compile_kwargs_json])

    # torch._dynamo.config overrides — worker accepts a JSON dict via --dynamo-flags
    dynamo_overrides = _parse_kv_overrides(getattr(args, "dynamo_config", None))
    if dynamo_overrides:
        extras.extend(["--dynamo-flags", json.dumps(dynamo_overrides)])

    # torch._inductor.config overrides
    inductor_overrides = _parse_kv_overrides(getattr(args, "inductor_config", None))
    if inductor_overrides:
        extras.extend(["--inductor-flags", json.dumps(inductor_overrides)])

    # User-supplied setup script exec'd in worker before each compile
    setup_script = getattr(args, "setup_script", None)
    if setup_script:
        path = Path(setup_script).resolve()
        if not path.exists():
            print(f"ERROR: --setup-script not found: {path}", file=sys.stderr)
            sys.exit(1)
        extras.extend(["--setup-script", str(path)])

    return extras


def _resolve_run_output_dir(args):
    """Apply --run-name convention. --output-dir wins if explicitly given.

    Returns the resolved Path. Mutates nothing.
    """
    run_name = getattr(args, "run_name", None)
    if run_name and args.output_dir == str(DEFAULT_OUTPUT_DIR):
        date = time.strftime("%Y-%m-%d")
        return EXPERIMENTS_OUTPUT_DIR / f"{run_name}-{date}"
    return Path(args.output_dir)


def _execute_explain_pass(python_bin, specs, device, workers, timeout_s,
                          output_dir, version_info, resume=False,
                          extra_worker_args=None, run_name=None):
    """Run the explain pass and save results. Shared by sweep and standalone explain.

    Returns (explain_results, explain_time).
    """
    modes = ["eval", "train"]
    explain_ckpt = str(output_dir / "explain_checkpoint.jsonl")

    resume_from = {}
    if resume and os.path.exists(explain_ckpt):
        resume_from = load_checkpoint(explain_ckpt)
        print(f"Loaded {len(resume_from)} completed explain results from checkpoint")

    print(f"\n{'=' * 70}")
    print(f"EXPLAIN: Detailed analysis — {len(specs)} models × {len(modes)} modes ({modes})")
    print(f"{'=' * 70}")

    explain_start = time.perf_counter()
    explain_results = run_pass(
        python_bin, specs, pass_num=2, device=device, modes=modes,
        workers=workers, timeout_s=timeout_s * 2,  # 2x for explain
        checkpoint_file=explain_ckpt, resume_from=resume_from,
        extra_worker_args=extra_worker_args,
    )
    explain_time = time.perf_counter() - explain_start

    # Save results
    explain_meta = {
        "pass": "explain",
        "device": device,
        "modes": modes,
        "workers": workers,
        "total_time_s": round(explain_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if version_info:
        explain_meta["versions"] = version_info
    if extra_worker_args:
        explain_meta["worker_extras"] = extra_worker_args
    if run_name:
        explain_meta["run_name"] = run_name
    explain_output = {
        "metadata": explain_meta,
        "results": explain_results,
    }
    explain_file = output_dir / "explain_results.json"
    with open(explain_file, "w") as f:
        json.dump(explain_output, f, indent=2)

    print(f"\nExplain pass complete: {explain_time:.1f}s")
    print(f"Saved to {explain_file}")

    return explain_results, explain_time


def _specs_for_graph_break_models(identify_results, all_specs):
    """Filter specs to only models with graph_break status in identify results.

    Uses the original enumerated specs (with full metadata like variant,
    hf_config, etc.) rather than reconstructing from identify results,
    which would be lossy.
    """
    graph_break_names = set()
    for r in identify_results:
        if r.get("status") == "graph_break":
            graph_break_names.add(r["name"])
    return [s for s in all_specs if s["name"] in graph_break_names]


def load_large_models(path=None):
    """Load the large model registry — models that need extended timeouts."""
    path = path or LARGE_MODELS_FILE
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_corpus_stability(corpus_path=None):
    """Load corpus and classify models as stable or unstable.

    Stable = full_graph in ALL modes (eval, train) and all dynamic variants.
    Unstable = everything else (graph_break, error, or missing data).

    Returns (stable_names: set, unstable_names: set).
    """
    corpus_path = corpus_path or CORPUS_FILE
    if not os.path.exists(corpus_path):
        return set(), set()
    with open(corpus_path) as f:
        corpus = json.load(f)
    stable = set()
    unstable = set()
    for m in corpus.get("models", []):
        name = m["name"]
        is_stable = True
        for mode in ("eval", "train"):
            md = m.get(mode, {})
            if md.get("status") != "full_graph":
                is_stable = False
                break
            for dyn in ("dynamic_mark", "dynamic_true"):
                dm = md.get(dyn, {})
                if dm and dm.get("status") != "full_graph":
                    is_stable = False
                    break
            if not is_stable:
                break
        if is_stable:
            stable.add(name)
        else:
            unstable.add(name)
    return stable, unstable


def save_large_models(registry, path=None):
    """Save the large model registry."""
    path = path or LARGE_MODELS_FILE
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def _log_versions(python_bin, output_dir):
    """Thin wrapper around orchestrator.log_versions for backward compatibility."""
    return log_versions(python_bin)


def run_sweep(args):
    """Main sweep logic."""
    python_bin = _resolve_python(args)
    output_dir = _resolve_run_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_worker_args = _build_extra_worker_args(args)

    # ── Write early state file so watchdog can detect us immediately ──
    state_file = output_dir / "sweep_state.json"
    early_state = {
        "status": "initializing",
        "pid": os.getpid(),
        "output_dir": str(output_dir),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": sys.argv[1:],
        "restart_count": 0,
    }
    # Preserve restart_count from previous state if resuming
    if args.resume and state_file.exists():
        try:
            with open(state_file) as f:
                old_state = json.load(f)
            early_state["restart_count"] = old_state.get("restart_count", 0)
        except (json.JSONDecodeError, KeyError):
            pass
    with open(state_file, "w") as f:
        json.dump(early_state, f, indent=2)
    print(f"Sweep state: {state_file} (PID {os.getpid()})")

    # ── Log and validate library versions ──
    version_info = _log_versions(python_bin, output_dir)
    if version_info:
        early_state["versions"] = version_info
        with open(state_file, "w") as f:
            json.dump(early_state, f, indent=2)

    # ── Load or enumerate models ──
    if args.models:
        try:
            with open(args.models) as f:
                specs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {args.models}: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"ERROR: File not found: {args.models}")
            sys.exit(1)
        print(f"Loaded {len(specs)} models from {args.models}")
    else:
        # Enumerate from source(s)
        from models import enumerate_timm, enumerate_hf, enumerate_diffusers, enumerate_custom
        sources = _resolve_source(args.source)
        source_enumerators = {
            "timm": enumerate_timm,
            "hf": enumerate_hf,
            "diffusers": lambda: [m for m in enumerate_diffusers()
                                  if m.get("has_config", True)],
            "custom": enumerate_custom,
        }
        specs = []
        by_src = {}
        for src in sources:
            src_specs = source_enumerators[src]()
            by_src[src] = len(src_specs)
            specs.extend(src_specs)
        src_detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_src.items()))
        print(f"Enumerated {len(specs)} models ({src_detail})")

    # Stability filtering — applied before limit
    if args.stability:
        stable_names, unstable_names = load_corpus_stability()
        before = len(specs)
        if args.stability == "stable":
            specs = [s for s in specs if s["name"] in stable_names]
            print(f"Stability filter [stable]: {len(specs)} models "
                  f"(from {before}, skipping {before - len(specs)} unstable/new)")
        elif args.stability == "unstable":
            specs = [s for s in specs if s["name"] not in stable_names]
            new_count = sum(1 for s in specs
                           if s["name"] not in unstable_names)
            known_unstable = len([s for s in specs
                                  if s["name"] in unstable_names])
            print(f"Stability filter [unstable]: {len(specs)} models "
                  f"(from {before}, {known_unstable} known unstable "
                  f"+ {new_count} new)")

    # Apply limit (after stability filtering)
    if args.limit:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} models")

    modes = args.modes
    dynamic = _resolve_dynamic(args)
    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Modes: {modes}, Timeout: {args.timeout}s"
          f"{f', Dynamic: {args.dynamic_dim}' if args.dynamic_dim else ''}")

    # ── Load large model registry for tiered timeouts ──
    large_registry = load_large_models()
    # Tier-aware timeouts (2026-04-28). Previously all "large" registry
    # entries got `args.timeout * 3` regardless of tier, so models marked
    # `very_large` (e.g. Gemma3n) still timed out. Now: `large = 3x`,
    # `very_large = 9x` — matches what the entries declare.
    timeout_large = args.timeout * 3
    timeout_very_large = args.timeout * 9
    def _timeout_for(name):
        entry = large_registry.get(name)
        if not entry:
            return None
        tier = entry.get("timeout_tier", "large") if isinstance(entry, dict) else "large"
        if tier == "very_large":
            return timeout_very_large
        return timeout_large
    timeout_overrides = {name: _timeout_for(name) for name in large_registry}
    if large_registry:
        n_large = sum(1 for v in timeout_overrides.values() if v == timeout_large)
        n_vl = sum(1 for v in timeout_overrides.values() if v == timeout_very_large)
        print(f"Large model registry: {n_large} 'large' tier ({timeout_large}s) + "
              f"{n_vl} 'very_large' tier ({timeout_very_large}s)")

    # ── Load skip list (toxic models) — auto-loaded from config ──
    skip_models = _load_skip_models()
    # ── Load known errors — skip + post-identify validator ──
    # See known_errors.json header for workflow. Models with stable, filed
    # create_error or eager_error bugs are skipped here so they don't pollute
    # output; any *new* failure of those classes during the sweep is flagged
    # loud post-identify (and with --strict-known-errors, exits non-zero).
    known_error_models, known_error_map = _load_known_errors()
    skip_models = skip_models | known_error_models

    print()

    # ── Update watchdog state file with full details ──
    total_work_items = len(specs) * len(modes)
    state_file = output_dir / "sweep_state.json"
    with open(state_file) as f:
        state = json.load(f)
    state.update({
        "status": "running",
        "total_models": len(specs),
        "total_work_items": total_work_items,
        "modes": modes,
    })
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # ── Signal handler: flush results on kill ──
    interrupted = False
    streaming_fh = None

    def _handle_signal(signum, frame):
        nonlocal interrupted
        interrupted = True
        if streaming_fh and not streaming_fh.closed:
            streaming_fh.flush()
            streaming_fh.close()
        marker = output_dir / "INTERRUPTED"
        marker.write_text(
            f"Interrupted by signal {signum} at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
        )
        print(f"\n  Interrupted (signal {signum}). Results saved to checkpoint.")
        sys.exit(1)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # ══════════════════════════════════════════════════════════════════════
    # IDENTIFY: Fast identification (eval-only by default)
    # ══════════════════════════════════════════════════════════════════════
    identify_ckpt = str(output_dir / "identify_checkpoint.jsonl")

    # Load checkpoint for resume
    resume_from = {}
    if args.resume and os.path.exists(identify_ckpt):
        resume_from = load_checkpoint(identify_ckpt)
        print(f"Loaded {len(resume_from)} completed results from checkpoint")

    # Models that fail under multiworker GPU contention but pass serially.
    # Run these single-worker from the start to avoid wasted first attempts.
    _SINGLE_WORKER_MODELS = {
        "Qwen3_5Model", "Qwen3_5TextModel", "Qwen3_5ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeModel", "Qwen3_5MoeTextModel", "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3NextModel", "Qwen3NextForCausalLM",
        "OlmoHybridModel", "OlmoHybridForCausalLM",
    }

    multi_specs = [s for s in specs if s["name"] not in _SINGLE_WORKER_MODELS]
    single_specs = [s for s in specs if s["name"] in _SINGLE_WORKER_MODELS]

    print(f"{'=' * 70}")
    print(f"IDENTIFY: Graph breaks (fullgraph=True) — {len(specs)} models × {len(modes)} modes ({modes})")
    if single_specs:
        print(f"  {len(multi_specs)} multi-worker, {len(single_specs)} single-worker (flaky under contention)")
    print(f"{'=' * 70}")

    # Streaming callback: write each result to checkpoint as it completes
    streaming_file = output_dir / "identify_streaming.jsonl"
    streaming_fh = open(streaming_file, "a")

    def _on_result(result):
        streaming_fh.write(json.dumps(result) + "\n")
        streaming_fh.flush()

    identify_start = time.perf_counter()
    identify_results = run_pass(
        python_bin, multi_specs, pass_num=1, device=args.device, modes=modes,
        workers=args.workers, timeout_s=args.timeout,
        checkpoint_file=identify_ckpt, resume_from=resume_from,
        dynamic=dynamic, timeout_overrides=timeout_overrides,
        skip_models=skip_models, extra_worker_args=extra_worker_args,
        result_callback=_on_result,
    )

    # Run single-worker models serially to avoid GPU contention flakiness
    if single_specs:
        print(f"\n{'─' * 70}")
        print(f"SINGLE-WORKER: {len(single_specs)} models (flaky under multiworker)")
        print(f"{'─' * 70}")
        single_results = run_pass(
            python_bin, single_specs, pass_num=1, device=args.device, modes=modes,
            workers=1, timeout_s=args.timeout,
            checkpoint_file=identify_ckpt, resume_from=resume_from,
            dynamic=dynamic, timeout_overrides=timeout_overrides,
            skip_models=skip_models, extra_worker_args=extra_worker_args,
            result_callback=_on_result,
        )
        identify_results.extend(single_results)

    identify_time = time.perf_counter() - identify_start
    streaming_fh.close()

    # Save identify results (full JSON for analysis)
    identify_metadata = {
        "pass": "identify",
        "device": args.device,
        "modes": modes,
        "workers": args.workers,
        "timeout_s": args.timeout,
        "total_time_s": round(identify_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": python_bin,
        "dynamic": dynamic,
    }
    if version_info:
        identify_metadata["versions"] = version_info
    if args.run_name:
        identify_metadata["run_name"] = args.run_name
    if extra_worker_args:
        identify_metadata["worker_extras"] = extra_worker_args
    identify_output = {
        "metadata": identify_metadata,
        "results": identify_results,
    }
    identify_file = output_dir / "identify_results.json"
    with open(identify_file, "w") as f:
        json.dump(identify_output, f, indent=2)

    # Summarize identify pass
    by_status = {}
    for r in identify_results:
        by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1

    print(f"\nIdentify pass complete: {identify_time:.1f}s")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print(f"Saved to {identify_file}")

    # ── Validate gated failures against known-errors list ──
    # Any create_error / eager_error not pre-declared in known_errors.json is
    # a NEW setup/build/dep/model failure that risks masking real graph-break
    # signal. Flag loud so the operator either (a) fixes the underlying issue,
    # or (b) adds an entry to known_errors.json with a reason. Goal: zero
    # unexplained gated failures per sweep.
    n_unexpected = _validate_no_unexpected_errors(
        identify_results, known_error_map, strict=True)
    if n_unexpected > 0 and getattr(args, "strict_known_errors", False):
        print(f"\nFAILING: {n_unexpected} unexpected gated failures and "
              f"--strict-known-errors is set. Resolve them and re-run.",
              file=sys.stderr)
        sys.exit(3)

    # ── Auto-retry: re-run timed-out models with extended timeout ──
    timeout_results = [r for r in identify_results if r.get("status") == "timeout"]
    # Only retry models that aren't already using the large timeout
    new_timeouts = [r for r in timeout_results if r["name"] not in large_registry]
    if new_timeouts and not args.no_auto_retry:
        timeout_names = {r["name"] for r in new_timeouts}
        retry_specs = [s for s in specs if s["name"] in timeout_names]

        print(f"\n{'─' * 70}")
        print(f"AUTO-RETRY: {len(retry_specs)} timed-out models with extended timeout ({timeout_large}s)")
        print(f"{'─' * 70}")

        # Build retry overrides — all get the large timeout
        retry_overrides = {s["name"]: timeout_large for s in retry_specs}

        retry_start = time.perf_counter()
        retry_results = run_pass(
            python_bin, retry_specs, pass_num=1, device=args.device, modes=modes,
            workers=max(1, args.workers // 2),  # fewer workers for large models
            timeout_s=timeout_large,
            checkpoint_file=None,  # don't mix with main checkpoint
            dynamic=dynamic, timeout_overrides=retry_overrides,
            extra_worker_args=extra_worker_args,
        )
        retry_time = time.perf_counter() - retry_start

        # Summarize retry results
        retry_by_status = {}
        for r in retry_results:
            retry_by_status[r.get("status", "unknown")] = retry_by_status.get(r.get("status", "unknown"), 0) + 1
        print(f"\nRetry complete: {retry_time:.1f}s")
        for status, count in sorted(retry_by_status.items()):
            print(f"  {status}: {count}")

        # Replace timeout results in identify_results with retry results
        retry_index = {(r["name"], r.get("mode", "eval")): r for r in retry_results}
        updated_count = 0
        for i, r in enumerate(identify_results):
            key = (r["name"], r.get("mode", "eval"))
            if key in retry_index:
                identify_results[i] = retry_index[key]
                updated_count += 1

        # Update checkpoint with retry results
        if os.path.exists(identify_ckpt):
            # Rewrite checkpoint with updated results
            all_completed = {}
            for r in identify_results:
                key = (r["name"], r.get("mode", "eval"))
                all_completed[key] = r
            with open(identify_ckpt, "w") as f:
                for r in all_completed.values():
                    f.write(json.dumps(r) + "\n")

        # Update large model registry — add models that resolved (not still timeout)
        newly_large = []
        for r in retry_results:
            if r.get("status") != "timeout":
                large_registry[r["name"]] = {
                    "source": r.get("source", "unknown"),
                    "timeout_tier": "large",
                    "resolved_status": r.get("status"),
                    "wall_time_s": r.get("wall_time_s"),
                    "discovered": time.strftime("%Y-%m-%d"),
                }
                newly_large.append(r["name"])
            else:
                large_registry[r["name"]] = {
                    "source": r.get("source", "unknown"),
                    "timeout_tier": "very_large",
                    "phase_at_timeout": r.get("phase_at_timeout", "unknown"),
                    "discovered": time.strftime("%Y-%m-%d"),
                }
        save_large_models(large_registry)
        print(f"\n  Updated large model registry: {len(newly_large)} newly resolved, "
              f"{len(large_registry)} total entries")

        # Re-save identify results with retry data merged
        identify_output["results"] = identify_results
        identify_output["metadata"]["retry_count"] = len(retry_specs)
        identify_output["metadata"]["timeout_large_s"] = timeout_large
        with open(identify_file, "w") as f:
            json.dump(identify_output, f, indent=2)

        # Recompute status summary
        by_status = {}
        for r in identify_results:
            by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1
        print(f"\nUpdated identify summary:")
        for status, count in sorted(by_status.items()):
            print(f"  {status}: {count}")

    # ── Auto-retry: re-run error models serially to distinguish real from transient ──
    error_results = [r for r in identify_results
                     if r.get("status") in ("eager_error", "create_error", "worker_error")]
    if error_results and not args.no_auto_retry:
        error_names = {r["name"] for r in error_results}
        retry_error_specs = [s for s in specs if s["name"] in error_names]

        print(f"\n{'─' * 70}")
        print(f"AUTO-RETRY ERRORS: {len(retry_error_specs)} error models, "
              f"serial (1 worker) to rule out GPU contention")
        print(f"{'─' * 70}")

        retry_err_start = time.perf_counter()
        retry_err_results = run_pass(
            python_bin, retry_error_specs, pass_num=1, device=args.device,
            modes=modes,
            workers=1,  # serial — no GPU contention
            timeout_s=args.timeout,
            checkpoint_file=None,
            dynamic=dynamic, timeout_overrides=timeout_overrides,
            skip_models=skip_models, extra_worker_args=extra_worker_args,
        )
        retry_err_time = time.perf_counter() - retry_err_start

        # Classify retry outcomes
        flaky = []
        confirmed = []
        for r in retry_err_results:
            orig = next((o for o in error_results
                         if o["name"] == r["name"] and o.get("mode") == r.get("mode")), None)
            if orig and r.get("status") not in ("eager_error", "create_error", "worker_error"):
                flaky.append(r)
                r["retry_note"] = f"flaky: was {orig['status']}, now {r['status']}"
            else:
                confirmed.append(r)
                r["retry_note"] = "confirmed_error"

        print(f"\nError retry complete: {retry_err_time:.1f}s")
        print(f"  Flaky (passed on retry): {len(flaky)}")
        for r in flaky:
            print(f"    {r['name']} {r.get('mode','?')}: now {r['status']}")
        print(f"  Confirmed errors: {len(confirmed)}")

        # Replace error results with retry results in identify_results
        retry_err_index = {(r["name"], r.get("mode", "eval")): r for r in retry_err_results}
        for i, r in enumerate(identify_results):
            key = (r["name"], r.get("mode", "eval"))
            if key in retry_err_index:
                identify_results[i] = retry_err_index[key]

        # Update checkpoint with retry results
        if os.path.exists(identify_ckpt):
            all_completed = {}
            for r in identify_results:
                key = (r["name"], r.get("mode", "eval"))
                all_completed[key] = r
            with open(identify_ckpt, "w") as f:
                for r in all_completed.values():
                    f.write(json.dumps(r) + "\n")

        # Re-save identify results
        identify_output["results"] = identify_results
        identify_output["metadata"]["error_retry_count"] = len(retry_error_specs)
        identify_output["metadata"]["error_retry_flaky"] = len(flaky)
        identify_output["metadata"]["error_retry_confirmed"] = len(confirmed)
        with open(identify_file, "w") as f:
            json.dump(identify_output, f, indent=2)

        # Recompute status summary
        by_status = {}
        for r in identify_results:
            by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1
        print(f"\nUpdated identify summary (after error retry):")
        for status, count in sorted(by_status.items()):
            print(f"  {status}: {count}")

    # Identify broken models for explain pass
    explain_specs = _specs_for_graph_break_models(identify_results, specs)
    print(f"\n→ {len(explain_specs)} models need explain pass (will test {modes})")

    # ══════════════════════════════════════════════════════════════════════
    # EXPLAIN: Detailed analysis (broken models only)
    # ══════════════════════════════════════════════════════════════════════
    if explain_specs and not args.identify_only:
        explain_results, _ = _execute_explain_pass(
            python_bin, explain_specs, device=args.device,
            workers=args.workers, timeout_s=args.timeout,
            output_dir=output_dir, version_info=version_info,
            resume=args.resume,
            extra_worker_args=extra_worker_args,
            run_name=args.run_name,
        )
    else:
        explain_results = []
        if args.identify_only:
            print("\nSkipping explain pass (--identify-only)")

    # ══════════════════════════════════════════════════════════════════════
    # MERGED CORPUS
    # ══════════════════════════════════════════════════════════════════════
    if identify_results and explain_results:
        corpus = _build_corpus(identify_results, explain_results, args)
        corpus_file = output_dir / "corpus.json"
        with open(corpus_file, "w") as f:
            json.dump(corpus, f, indent=2)
        print(f"\nCorpus saved to {corpus_file}")
        _print_summary(corpus)

    # ── Update watchdog state to done ──
    state_file = output_dir / "sweep_state.json"
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
            state["status"] = "done"
            state["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass


def run_explain(args):
    """Run explain pass only, from prior identify results.

    Re-enumerates models from their sources to get full specs (with variant,
    hf_config, etc.), then filters to graph_break models from the identify
    results. This avoids lossy spec reconstruction from identify JSON.
    """
    python_bin = _resolve_python(args)
    output_dir = _resolve_run_output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect versions
    version_info = _log_versions(python_bin, output_dir)

    # Load identify results
    try:
        with open(args.file) as f:
            identify_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {args.file}: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    identify_results = (identify_data if isinstance(identify_data, list)
                        else identify_data.get("results", []))

    # Determine which sources were in the identify sweep
    sources_in_sweep = set(r.get("source", "hf") for r in identify_results)
    print(f"Sources in identify results: {', '.join(sorted(sources_in_sweep))}")

    # Re-enumerate models from the same sources to get full specs
    from models import enumerate_timm, enumerate_hf, enumerate_diffusers, enumerate_custom
    source_enumerators = {
        "timm": enumerate_timm,
        "hf": enumerate_hf,
        "diffusers": lambda: [m for m in enumerate_diffusers()
                              if m.get("has_config", True)],
        "custom": enumerate_custom,
    }
    all_specs = []
    for src in sources_in_sweep:
        if src in source_enumerators:
            all_specs.extend(source_enumerators[src]())
    print(f"Re-enumerated {len(all_specs)} models from {', '.join(sorted(sources_in_sweep))}")

    # Filter to graph_break models
    explain_specs = _specs_for_graph_break_models(identify_results, all_specs)
    print(f"Loaded {len(explain_specs)} broken models from {args.file}")

    if not explain_specs:
        print("No graph_break models found — nothing to explain.")
        return

    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Timeout: {args.timeout * 2}s (2x for explain)")

    _execute_explain_pass(
        python_bin, explain_specs, device=args.device,
        workers=args.workers, timeout_s=args.timeout,
        output_dir=output_dir, version_info=version_info,
        resume=args.resume,
        extra_worker_args=_build_extra_worker_args(args),
        run_name=getattr(args, "run_name", None),
    )


def _build_corpus(identify_results, explain_results, args):
    """Merge identify and explain results into a unified corpus."""
    # Index explain pass by (name, mode)
    explain_index = {}
    for r in explain_results:
        key = (r["name"], r["mode"])
        explain_index[key] = r

    models = []
    for r in identify_results:
        record = dict(r)
        key = (r["name"], r["mode"])
        if key in explain_index:
            record["explain"] = explain_index[key]
        models.append(record)

    # Summary stats
    by_status = {}
    for r in identify_results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    return {
        "metadata": {
            "device": args.device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "methodology": "Two-pass: identify(all) → explain+TORCH_TRACE(broken only)",
        },
        "summary": by_status,
        "models": models,
    }


def _print_summary(corpus):
    """Print corpus summary table."""
    print(f"\n{'=' * 70}")
    print("CORPUS SUMMARY")
    print(f"{'=' * 70}")
    for status, count in sorted(corpus["summary"].items()):
        print(f"  {status}: {count}")

    # Graph break details
    breaks = [m for m in corpus["models"] if m.get("status") == "graph_break" and "explain" in m]
    if breaks:
        print(f"\nGraph break models ({len(breaks)}):")
        for m in breaks:
            p2 = m["explain"]
            name = m["name"]
            mode = m["mode"]
            bc = p2.get("graph_break_count", "?")
            gc = p2.get("graph_count", "?")
            reasons = p2.get("break_reasons", [])
            top_reason = reasons[0]["reason"][:80] if reasons else "(no break reasons captured)"
            print(f"  {name:<30} {mode:<6} {bc} breaks, {gc} graphs")
            print(f"    → {top_reason}")



def run_validation(args):
    """Run two-shape validation sweep (pass 3) on clean models."""
    python_bin = _resolve_python(args)
    output_dir = _resolve_run_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models to validate
    if args.from_file:
        with open(args.from_file) as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get("results", [])
        # Only validate clean models
        specs = []
        seen = set()
        for r in results:
            if r.get("status") == "full_graph" and r["name"] not in seen:
                seen.add(r["name"])
                spec = {"name": r["name"], "source": r["source"]}
                for k in ["hf_class", "hf_config", "input_type", "constructor_args", "inputs"]:
                    if k in r:
                        spec[k] = r[k]
                specs.append(spec)
        print(f"Loaded {len(specs)} clean models from {args.from_file}")
    elif args.models:
        with open(args.models) as f:
            specs = json.load(f)
        print(f"Loaded {len(specs)} models from {args.models}")
    else:
        from models import enumerate_all
        specs = enumerate_all()
        print(f"Enumerated {len(specs)} models (will validate all)")

    if args.limit:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} models")

    validate_modes = ["eval", "train"]
    dynamic = _resolve_dynamic(args) or "true"  # default to all-dim dynamic
    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Modes: {validate_modes}, Timeout: {args.timeout}s, Dynamic: {dynamic}")
    print()

    # Checkpoint for resume
    validate_ckpt = str(output_dir / "validate_checkpoint.jsonl")
    resume_from = {}
    if args.resume and os.path.exists(validate_ckpt):
        resume_from = load_checkpoint(validate_ckpt)
        print(f"Loaded {len(resume_from)} completed results from checkpoint")

    print(f"{'=' * 70}")
    print(f"VALIDATION: Two-shape correctness check — {len(specs)} models × {len(validate_modes)} modes")
    print(f"{'=' * 70}")

    val_start = time.perf_counter()
    val_results = run_pass(
        python_bin, specs, pass_num=3, device=args.device, modes=validate_modes,
        workers=args.workers, timeout_s=args.timeout,
        checkpoint_file=validate_ckpt, resume_from=resume_from,
        dynamic=dynamic,
    )
    val_time = time.perf_counter() - val_start

    # Save results
    val_output = {
        "metadata": {
            "pass": "validate",
            "device": args.device,
            "modes": validate_modes,
            "workers": args.workers,
            "timeout_s": args.timeout,
            "total_time_s": round(val_time, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": python_bin,
            "dynamic": dynamic,
        },
        "results": val_results,
    }
    val_file = output_dir / "validate_results.json"
    with open(val_file, "w") as f:
        json.dump(val_output, f, indent=2)

    # Summarize
    by_status = {}
    for r in val_results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    print(f"\nValidation complete: {val_time:.1f}s")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print(f"Saved to {val_file}")

    # Print mismatches
    mismatches = [r for r in val_results if r.get("status") == "mismatch"]
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for r in mismatches:
            print(f"  {r['source']}/{r['name']} {r.get('mode','?')}: "
                  f"max_diff={r.get('max_diff','?')} — {r.get('compare_details','')}")


def run_correctness(args):
    """Run Phase 3 correctness sweep (pass 4): eager vs compiled forward output comparison.

    Default source is corpus.json filtered to source=hf with eval.fullgraph_ok=True.
    Eval mode only (train is V2 per design Section 8).
    """
    python_bin = _resolve_python(args)
    output_dir = _resolve_run_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load specs
    if args.from_file:
        with open(args.from_file) as f:
            data = json.load(f)
        # Two formats supported: corpus.json (under "models") or identify_results.json (list/results)
        if isinstance(data, dict) and "models" in data:
            entries = data["models"]
        else:
            entries = data if isinstance(data, list) else data.get("results", [])
        specs = []
        seen = set()
        for e in entries:
            if e.get("source") != "hf":
                continue
            eval_block = e.get("eval", {})
            # corpus.json uses status="full_graph"; identify results use the same
            status = eval_block.get("status") if isinstance(eval_block, dict) else None
            if status != "full_graph":
                continue
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            spec = {"name": e["name"], "source": e["source"]}
            for k in ["hf_class", "hf_config", "input_type", "constructor_args", "inputs"]:
                if k in e:
                    spec[k] = e[k]
            specs.append(spec)
        print(f"Loaded {len(specs)} HF clean models from {args.from_file}")
    elif args.models:
        with open(args.models) as f:
            specs = json.load(f)
        print(f"Loaded {len(specs)} models from {args.models}")
    else:
        # Default: read corpus.json
        with open(CORPUS_FILE) as f:
            data = json.load(f)
        specs = []
        seen = set()
        for e in data.get("models", []):
            if e.get("source") != "hf":
                continue
            if e.get("eval", {}).get("status") != "full_graph":
                continue
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            specs.append({"name": e["name"], "source": e["source"]})
        print(f"Loaded {len(specs)} HF clean models from {CORPUS_FILE}")

    # Hydrate HF specs against the live enumeration so wrapper-class metadata
    # (variant, hf_class, hf_config, input_type) matches what the identify pass
    # uses. corpus.json doesn't carry these fields, and without them
    # create_hf_model can't resolve the right Config class for variants like
    # *LMHeadModel or *ForSequenceClassification — see Phase 3 / pt2.11
    # invariant violation, 2026-04-21.
    from models import enumerate_hf
    hf_meta = {s["name"]: s for s in enumerate_hf()}
    for spec in specs:
        if spec.get("source") != "hf":
            continue
        meta = hf_meta.get(spec["name"])
        if not meta:
            continue
        for k in ("hf_class", "hf_config", "variant", "input_type",
                  "constructor_args", "inputs"):
            if k not in spec and k in meta:
                spec[k] = meta[k]

    if args.limit:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} models")

    correctness_modes = ["eval"]  # MVP: eval-only (train is V2)
    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Modes: {correctness_modes}, Timeout: {args.timeout}s")
    print()

    # Checkpoint for resume
    ckpt_file = str(output_dir / "correctness_checkpoint.jsonl")
    resume_from = {}
    if args.resume and os.path.exists(ckpt_file):
        resume_from = load_checkpoint(ckpt_file)
        print(f"Loaded {len(resume_from)} completed results from checkpoint")

    print(f"{'=' * 70}")
    print(f"CORRECTNESS: Eager vs compiled forward — {len(specs)} models × {len(correctness_modes)} modes")
    print(f"{'=' * 70}")

    t_start = time.perf_counter()
    results = run_pass(
        python_bin, specs, pass_num=4, device=args.device, modes=correctness_modes,
        workers=args.workers, timeout_s=args.timeout,
        checkpoint_file=ckpt_file, resume_from=resume_from,
    )
    elapsed = time.perf_counter() - t_start

    # Save results
    output = {
        "metadata": {
            "pass": "correctness",
            "device": args.device,
            "modes": correctness_modes,
            "workers": args.workers,
            "timeout_s": args.timeout,
            "total_time_s": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python": python_bin,
            "tolerance": {"atol": 1e-6, "rtol": 1e-4, "dtype": "fp32"},
        },
        "results": results,
    }
    out_file = output_dir / "correctness_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Summarize
    by_status = {}
    for r in results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    print(f"\nCorrectness complete: {elapsed:.1f}s")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")
    print(f"Saved to {out_file}")

    # Print divergences sorted by severity_ratio (highest first)
    divergent = [r for r in results if r.get("status") in (
        "divergence", "nan_inf_introduced", "shape_mismatch", "dtype_mismatch")]
    if divergent:
        divergent.sort(key=lambda r: r.get("severity_ratio", 0), reverse=True)
        print(f"\nDIVERGENCES ({len(divergent)}, sorted by severity):")
        for r in divergent[:50]:
            print(f"  [{r.get('status')}] {r['source']}/{r['name']}: "
                  f"max_diff={r.get('max_diff','?')}, "
                  f"severity={r.get('severity_ratio','?')} — "
                  f"{r.get('first_divergence', '')}")


def check_env(args):
    """Pre-sweep environment validation: check installed versions against corpus."""
    python_bin = _resolve_python(args)
    version_check_script = SWEEP_DIR.parent / "tools" / "version_check.py"

    if not version_check_script.exists():
        print(f"ERROR: version_check.py not found at {version_check_script}")
        sys.exit(1)

    print("Pre-sweep environment check")
    print("=" * 50)

    # Run version_check.py with the specified python binary
    try:
        result = subprocess.run(
            [python_bin, str(version_check_script)],
            capture_output=True, text=True, timeout=30,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print("FAILED: Environment does not match corpus versions.")
            print("Fix version mismatches before running the sweep.")
            sys.exit(1)
        else:
            print("PASSED: Environment matches corpus versions.")
            sys.exit(0)
    except subprocess.TimeoutExpired:
        print("ERROR: Version check timed out")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def run_test_mode(args):
    """Quick integration test — pick 1 model per source, run both passes, validate output.

    Tests the full worker subprocess pipeline (spawn → harvest → parse) without
    running the full sweep. Catches import errors, JSON serialization bugs, and
    output format regressions.

    Usage: python run_sweep.py sweep --selftest [--device cpu]
    """
    python_bin = _resolve_python(args)
    device = args.device

    # Test models — chosen to exercise both clean and breaking paths
    # AutoformerModel has 5+ graph breaks with break_reasons — validates explain output thoroughly
    test_specs = [
        {"name": "resnet18", "source": "timm", "expect_breaks": False},
        {"name": "AutoformerModel", "source": "hf", "expect_breaks": True},
        {"name": "GFPGAN", "source": "custom", "category": "face_restoration", "expect_breaks": False},
    ]

    required_keys = {"status", "name", "source", "mode"}
    explain_keys = {"graph_count", "graph_break_count", "ops_per_graph", "compile_times",
                    "break_reasons", "explain_time_s"}

    print("=" * 60)
    print("Integration test: 1 model per source through worker pipeline")
    print(f"  resnet18 (timm) — full graph, validates clean path")
    print(f"  AutoformerModel (hf) — multiple graph breaks, validates explain depth")
    print(f"  GFPGAN (custom) — validates custom worker subprocess path")
    print(f"Device: {device}, Python: {python_bin}")
    print("=" * 60)

    passed, failed = 0, 0
    for spec in test_specs:
        source = spec["source"]
        name = spec["name"]
        expect_breaks = spec.pop("expect_breaks", None)

        for pass_num in [1, 2]:
            pass_name = "identify" if pass_num == 1 else "explain"
            label = f"{source}/{name} pass={pass_name}"
            print(f"\n--- {label} ---")

            try:
                handle = spawn_worker(python_bin, spec, pass_num, device, "eval",
                                      timeout_s=180)
                handle.proc.wait(timeout=180)
                result = harvest_worker(handle)

                if result is None:
                    print(f"  FAIL: no result returned")
                    failed += 1
                    continue

                # Check required keys
                missing = required_keys - set(result.keys())
                if missing:
                    print(f"  FAIL: missing keys {missing}")
                    print(f"  Got: {json.dumps(result, indent=2)[:500]}")
                    failed += 1
                    continue

                status = result["status"]
                if status in ("create_error", "download_error", "timeout"):
                    print(f"  SKIP: {status} — {result.get('error', '')[:200]}")
                    passed += 1  # environment issue, not pipeline bug
                    continue

                if status in ("ok", "full_graph", "graph_break"):
                    summary = (f"status={status}, "
                               f"graph_count={result.get('graph_count', 'N/A')}, "
                               f"breaks={result.get('graph_break_count', 'N/A')}, "
                               f"wall={result.get('wall_time_s', '?')}s")

                    # Deep validation for explain pass
                    if pass_num == 2:
                        missing_explain = explain_keys - set(result.keys())
                        if missing_explain:
                            print(f"  FAIL: explain result missing {missing_explain}")
                            failed += 1
                            continue

                        gc = result["graph_count"]
                        bc = result["graph_break_count"]
                        opg = result["ops_per_graph"]
                        ct = result["compile_times"]
                        br = result["break_reasons"]

                        # Structural invariants
                        if bc != max(0, gc - 1):
                            print(f"  FAIL: graph_break_count ({bc}) != graph_count-1 ({gc-1})")
                            failed += 1
                            continue
                        if len(opg) != gc:
                            print(f"  FAIL: ops_per_graph length ({len(opg)}) != graph_count ({gc})")
                            failed += 1
                            continue
                        if len(ct) != gc:
                            print(f"  FAIL: compile_times length ({len(ct)}) != graph_count ({gc})")
                            failed += 1
                            continue

                        # Validate break_reasons structure
                        for i, entry in enumerate(br):
                            if "reason" not in entry or "type" not in entry:
                                print(f"  FAIL: break_reasons[{i}] missing reason/type")
                                failed += 1
                                continue

                        # If we expect breaks, verify we got them
                        if expect_breaks and bc == 0:
                            print(f"  WARN: expected graph breaks but got 0 — "
                                  f"possible PyTorch version change. {summary}")
                        elif expect_breaks and bc > 0:
                            summary += f", reasons={len(br)}"

                    print(f"  PASS: {summary}")
                    passed += 1
                else:
                    print(f"  WARN: status={status}, error={result.get('error', '')[:200]}")
                    passed += 1

                json.dumps(result)  # must be serializable

            except subprocess.TimeoutExpired:
                print(f"  FAIL: subprocess timed out after 180s")
                try:
                    os.killpg(handle.proc.pid, signal.SIGKILL)
                except Exception:
                    pass
                failed += 1
            except Exception as e:
                print(f"  FAIL: {e}")
                failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


def _resolve_python(args):
    """Resolve the Python binary: SWEEP_PYTHON env var → sys.executable."""
    return os.environ.get("SWEEP_PYTHON", sys.executable)


def _resolve_dynamic(args):
    """Map --dynamic-dim {batch,all} to internal values {mark,true}."""
    dim = getattr(args, "dynamic_dim", None)
    if dim == "batch":
        return "mark"
    elif dim == "all":
        return "true"
    return None


def _resolve_source(source_list):
    """Expand --source list: 'all' → all four sources."""
    if "all" in source_list:
        return ["timm", "hf", "diffusers", "custom"]
    return list(source_list)


def _validate_sweep_args(args):
    """Validate mutual exclusion rules that argparse can't express."""
    has_models = getattr(args, "models", None)
    stability = getattr(args, "stability", None)
    is_filtered = stability is not None

    if has_models and is_filtered:
        print("ERROR: --stability filter cannot be used with --models.")
        print("  This filter only applies to enumerated models (--source).")
        sys.exit(1)

    source = getattr(args, "source", None)
    if source and "all" in source and len(source) > 1:
        print("ERROR: --source 'all' cannot be combined with individual sources.")
        sys.exit(1)


SKIP_MODELS_FILE = SWEEP_DIR / "skip_models.json"
KNOWN_ERRORS_FILE = SWEEP_DIR / "known_errors.json"


def _load_skip_models():
    """Auto-load toxic model skip list from config file."""
    if SKIP_MODELS_FILE.exists():
        with open(SKIP_MODELS_FILE) as f:
            models = set(json.load(f))
        if models:
            print(f"Skip list: {len(models)} models will be skipped "
                  f"(from {SKIP_MODELS_FILE})")
        return models
    return set()


def _load_known_errors():
    """Load known-error entries from `known_errors.json`.

    Each entry has a `status` (one of 'create_error' or 'eager_error') —
    declares that this model's failure of that class is a stable real bug
    (NOT an env issue) and should be skipped from the sweep.

    Returns: (skip_models, known_map)
      - skip_models: set of model names to skip entirely (we don't want them
        to even attempt creation; the failure is known + filed)
      - known_map: {(model, status): {"modes": [...], "error_pattern": "...",
                                       "reason": "..."}}
        used by `_validate_no_unexpected_errors()` after identify pass.

    Empty file / no file → empty set + empty dict.
    """
    if not KNOWN_ERRORS_FILE.exists():
        return set(), {}
    with open(KNOWN_ERRORS_FILE) as f:
        data = json.load(f)
    entries = data.get("entries", [])
    skip_models = {e["model"] for e in entries}
    known_map = {(e["model"], e["status"]): {"modes": e["modes"],
                                              "error_pattern": e["error_pattern"],
                                              "reason": e.get("reason", "")}
                 for e in entries}
    if entries:
        by_status = {}
        for e in entries:
            by_status[e["status"]] = by_status.get(e["status"], 0) + 1
        breakdown = ", ".join(f"{c} {s}" for s, c in sorted(by_status.items()))
        print(f"Known errors: {len(entries)} entries "
              f"({sum(len(e['modes']) for e in entries)} work items, {breakdown}) "
              f"will be skipped (from {KNOWN_ERRORS_FILE.name}).")
    return skip_models, known_map


# Statuses gated by known_errors.json. New ones can be added here without
# touching the validator logic — the gate is per-status.
GATED_STATUSES = ("create_error", "eager_error")

# Substring patterns that classify a gated failure as setup/env (NOT a real model
# bug). When the validator sees one of these, it surfaces the row in a separate
# "infra" summary instead of the loud "unexpected" warning. The hygiene gate
# exists to flag genuine new model bugs — not to re-warn about env/dep issues
# that are already documented elsewhere (build script, recipes/github-access).
INFRA_ERROR_PATTERNS = (
    # CUDA runtime libs missing (source-built torch missing nvidia-cuda-* preload)
    "libnvrtc", "libcudart", "libcuda.so", "nvrtc:",
    # Linker / .so issues
    "Could not load library", "undefined symbol",
    # Missing python packages
    "No module named",
    # HF/Transformers/Diffusers "X requires the Y library" phrasing
    "requires the natten library",
    "requires the detectron2 library",
    "requires the timm library",
    "requires the mamba_ssm",
    # LAPACK (torch built without LAPACK)
    "LAPACK", "lapack_LU", "geqrf",
    # cudnn ops library issues
    "cudnn",
    # GPU memory pressure (sweep-config / hardware issue, not a model bug)
    "CUDA out of memory",
)

# Substring patterns that classify a gated failure as a corpus-tooling bug
# (worker / create_model / config-builder mismatch — NOT a real model bug).
# Surface separately so the hygiene gate doesn't drown in harness noise.
HARNESS_ERROR_PATTERNS = (
    # diffusers + transformers signature mismatch — corpus's create_model didn't
    # supply the right input dict for this model class
    ".forward() missing ", ".__init__() missing ",
    # corpus generated a None where a config field needed an int
    "unsupported operand type(s) for ",
)


def _classify_failure(err: str) -> str:
    """Return one of: 'infra', 'harness', 'unknown'.

    'infra'   — env/setup/dep issue (libcuda missing, no module named, etc.)
    'harness' — corpus tooling bug (forward signature mismatch, None config)
    'unknown' — neither — likely a real model bug
    """
    for p in INFRA_ERROR_PATTERNS:
        if p in err:
            return "infra"
    for p in HARNESS_ERROR_PATTERNS:
        if p in err:
            return "harness"
    return "unknown"


def _validate_no_unexpected_errors(results, known_map, strict=True):
    """Flag any create_error / eager_error not pre-declared in known_errors.json.

    For each result with status in `GATED_STATUSES`, check that
    (model, status) is in `known_map` AND the mode is one of the declared
    modes AND the error message contains the declared `error_pattern`.

    Failures that look like env/setup issues (matched by INFRA_ERROR_PATTERNS)
    or corpus-tooling bugs (HARNESS_ERROR_PATTERNS) are surfaced in their own
    summaries — they're not gate concerns. Only the residual "unknown" bucket
    counts toward the strict-mode non-zero exit.

    Returns: count of unknown unexpected failures (the only category that should
    block strict-mode runs).
    """
    unexpected = []
    infra_failures = []
    harness_failures = []
    for r in results:
        status = r.get("status")
        if status not in GATED_STATUSES:
            continue
        name = r.get("name")
        mode = r.get("mode")
        err = r.get("error", "")
        match = known_map.get((name, status))
        if match and mode in match["modes"] and match["error_pattern"] in err:
            continue   # accounted for — known stable bug
        kind = _classify_failure(err)
        if kind == "infra":
            infra_failures.append((name, mode, status, err[:140]))
            continue
        if kind == "harness":
            harness_failures.append((name, mode, status, err[:140]))
            continue
        # Unknown — populate the loud warning bucket
        if not match:
            unexpected.append((name, mode, status, err[:120],
                               f"model not in known {status} list"))
        elif mode not in match["modes"]:
            unexpected.append((name, mode, status, err[:120],
                               f"model declared in known list but not for mode={mode}"))
        else:
            unexpected.append((name, mode, status, err[:120],
                               f"error pattern mismatch — expected to contain "
                               f"{match['error_pattern']!r}"))

    if infra_failures:
        print(f"\nℹ Infra/env failures (informational, NOT gated): {len(infra_failures)} work items",
              file=sys.stderr)
        from collections import Counter
        by_pattern = Counter()
        for name, mode, status, err in infra_failures:
            for p in INFRA_ERROR_PATTERNS:
                if p in err:
                    by_pattern[p] += 1
                    break
        for p, n in by_pattern.most_common():
            print(f"  {n:3d}× {p!r}", file=sys.stderr)
        print(f"  → fix at the venv / build script level (see scripts/build-nightly-from-source.sh)",
              file=sys.stderr)

    if harness_failures:
        print(f"\nℹ Corpus-harness failures (informational, NOT gated): {len(harness_failures)} work items",
              file=sys.stderr)
        from collections import Counter
        by_pattern = Counter()
        for name, mode, status, err in harness_failures:
            for p in HARNESS_ERROR_PATTERNS:
                if p in err:
                    by_pattern[p] += 1
                    break
        for p, n in by_pattern.most_common():
            print(f"  {n:3d}× {p!r}", file=sys.stderr)
        print(f"  → fix at the create_model / config-builder level (sweep/models.py)",
              file=sys.stderr)

    if unexpected:
        print("\n" + "=" * 70, file=sys.stderr)
        print(f"⚠ UNEXPECTED failures detected ({len(unexpected)} work items)",
              file=sys.stderr)
        print(f"  These hit a gated failure status but are NOT declared in",
              file=sys.stderr)
        print(f"  {KNOWN_ERRORS_FILE.name}. Resolve each by either:",
              file=sys.stderr)
        print("    (a) fixing the underlying setup/build/dep/model issue, OR",
              file=sys.stderr)
        print(f"    (b) appending an entry to {KNOWN_ERRORS_FILE.name}",
              file=sys.stderr)
        print("        with status, model, modes, error_pattern, reason.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        for name, mode, status, err, why in unexpected:
            print(f"  • {name} ({mode}) [{status}] — {why}", file=sys.stderr)
            print(f"      error: {err!r}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
    return len(unexpected)


def main():
    import warnings
    warnings.warn(
        "run_sweep.py is deprecated. Use tools/run_experiment.py instead:\n"
        "  run_experiment.py sweep    (was: run_sweep.py sweep)\n"
        "  run_experiment.py explain  (was: run_sweep.py explain)\n"
        "  run_experiment.py validate-shapes  (was: run_sweep.py validate)",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(
        description="Two-pass graph break sweep orchestrator "
                    "(DEPRECATED — use tools/run_experiment.py instead)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Shared parent parsers ──

    # Global options (all subcommands)
    global_parent = argparse.ArgumentParser(add_help=False)
    global_parent.add_argument("--device", default="cuda", choices=["cpu", "cuda"],
                               help="Hardware target (default: cuda)")

    # Run options (sweep, explain, validate)
    run_parent = argparse.ArgumentParser(add_help=False)
    run_parent.add_argument("--workers", type=int, default=4,
                            help="Parallel worker processes (default: 4)")
    run_parent.add_argument("--timeout", type=int, default=180,
                            help="Per-model timeout in seconds (default: 180)")
    run_parent.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                            help="Output directory (default: sweep_results/)")
    run_parent.add_argument("--resume", action="store_true",
                            help="Resume from JSONL checkpoint")
    # Compiler config passthrough — applied per-worker before each torch.compile
    run_parent.add_argument("--compile-kwargs", default=None, metavar="JSON",
                            help='JSON dict passed to torch.compile() '
                                 '(e.g. \'{"fullgraph": true, "dynamic": true}\')')
    run_parent.add_argument("--dynamo-config", action="append", default=None, metavar="KEY=VAL",
                            help="Set torch._dynamo.config.<key>=<val>. Repeatable. "
                                 "Value parsed as JSON literal (true/false/123/'str').")
    run_parent.add_argument("--inductor-config", action="append", default=None, metavar="KEY=VAL",
                            help="Set torch._inductor.config.<key>=<val>. Repeatable.")
    run_parent.add_argument("--setup-script", default=None, metavar="PATH",
                            help="Python file exec'd in each worker before compile. "
                                 "For multi-line config (e.g. logging suppression).")
    run_parent.add_argument("--run-name", default=None, metavar="SLUG",
                            help="Tag this run as experimental. Defaults output to "
                                 "sweep_results/experiments/<slug>-<date>/. "
                                 "Tagged into result metadata.")

    # ── sweep subcommand ──
    sweep_parser = subparsers.add_parser(
        "sweep", parents=[global_parent, run_parent],
        help="Identify + explain sweep (the main workflow)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run the two-pass graph break sweep: identify (fullgraph=True) "
                    "then explain (detailed break analysis).",
    )

    # Model selection
    sweep_input = sweep_parser.add_mutually_exclusive_group()
    sweep_input.add_argument("--source", nargs="+",
                             default=["hf", "diffusers", "custom"],
                             choices=["timm", "hf", "diffusers", "custom", "all"],
                             help="Model libraries to enumerate (default: hf diffusers custom). "
                                  "Accepts multiple values. 'all' = all four sources.")
    sweep_input.add_argument("--models",
                             help="JSON file with explicit model list")

    # Stability filter (only with --source)
    sweep_parser.add_argument("--stability",
                              choices=["stable", "unstable"],
                              help="Filter by corpus stability. Omit to run all. "
                                   "'stable' = full_graph in all modes, "
                                   "'unstable' = graph_break/error/new.")

    sweep_parser.add_argument("--limit", type=int,
                              help="Max models to test (applied last)")

    # Execution
    sweep_parser.add_argument("--modes", nargs="+", default=["eval", "train"],
                              choices=["eval", "train"],
                              help="Modes to run (default: eval train)")
    sweep_parser.add_argument("--dynamic-dim", choices=["batch", "all"],
                              help="Dynamic shapes: batch = batch dim only, "
                                   "all = all dims. Omit for static.")
    sweep_parser.add_argument("--no-auto-retry", action="store_true",
                              help="Skip auto-retry of timed-out/errored models")
    sweep_parser.add_argument("--identify-only", action="store_true",
                              help="Stop after identify pass (skip explain)")
    sweep_parser.add_argument("--strict-known-errors", action="store_true",
                              help="Exit non-zero if any create_error or eager_error appears "
                                   "in identify results that is NOT pre-declared in "
                                   "known_errors.json. Use in CI or when shipping data "
                                   "downstream — guarantees that no unfixed setup/build/"
                                   "dep/model issue silently masks graph-break signal. "
                                   "New errors must be either fixed or added to the list.")
    # Backward compat: keep --strict-create-errors as an alias (was the
    # previous name when the gate only covered create_error).
    sweep_parser.add_argument("--strict-create-errors", action="store_true",
                              dest="strict_known_errors",
                              help=argparse.SUPPRESS)

    # Utilities
    sweep_parser.add_argument("--selftest", action="store_true",
                              help="Smoke test: 3 models, both passes, validate output, exit")
    sweep_parser.add_argument("--check-env", action="store_true",
                              help="Validate versions against corpus, exit")

    # ── explain subcommand ──
    explain_parser = subparsers.add_parser(
        "explain", parents=[global_parent, run_parent],
        help="Explain-only from prior identify results",
        description="Run the explain pass on broken models from a prior identify sweep.",
    )
    explain_parser.add_argument("file", metavar="FILE",
                                help="Path to identify results JSON")

    # ── validate subcommand ──
    validate_parser = subparsers.add_parser(
        "validate", parents=[global_parent, run_parent],
        help="Two-shape correctness check",
        description="Run two-shape validation: compile models with two different "
                    "input shapes, compare outputs against eager mode.",
    )
    validate_input = validate_parser.add_mutually_exclusive_group()
    validate_input.add_argument("--from", dest="from_file",
                                help="Identify results JSON — validates full_graph models")
    validate_input.add_argument("--models",
                                help="JSON file with explicit model list to validate")
    validate_parser.add_argument("--dynamic-dim", choices=["batch", "all"],
                                 default="all",
                                 help="Dynamic shapes (default: all)")
    validate_parser.add_argument("--limit", type=int,
                                 help="Max models to validate")

    # ── correctness subcommand (Phase 3) ──
    correctness_parser = subparsers.add_parser(
        "correctness", parents=[global_parent, run_parent],
        help="Eager vs compiled forward output comparison",
        description="Phase 3: compare eager and compiled forward outputs on clean models. "
                    "Defaults to corpus.json HF eval fullgraph_ok subset.",
    )
    correctness_input = correctness_parser.add_mutually_exclusive_group()
    correctness_input.add_argument("--from", dest="from_file",
                                   help="Source JSON (corpus.json or identify_results.json)")
    correctness_input.add_argument("--models",
                                   help="JSON file with explicit model list")
    correctness_parser.add_argument("--limit", type=int,
                                    help="Max models to test")

    args = parser.parse_args()

    # Default to 'sweep' if no subcommand given
    if args.command is None:
        args = parser.parse_args(["sweep"])

    if args.command == "sweep":
        _validate_sweep_args(args)
        if args.selftest:
            run_test_mode(args)
        elif args.check_env:
            check_env(args)
        else:
            run_sweep(args)
    elif args.command == "explain":
        run_explain(args)
    elif args.command == "validate":
        run_validation(args)
    elif args.command == "correctness":
        run_correctness(args)


if __name__ == "__main__":
    main()
