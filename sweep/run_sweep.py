#!/usr/bin/env python3
"""Sweep orchestrator — runs the two-pass graph break sweep.

Manages subprocess workers, parallel execution, timeouts, and result merging.

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
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path


SWEEP_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SWEEP_DIR / "worker.py"
CUSTOM_WORKER_SCRIPT = SWEEP_DIR.parent / "corpora" / "custom-models" / "worker.py"
CORPUS_FILE = SWEEP_DIR.parent / "corpus" / "corpus.json"
DEFAULT_OUTPUT_DIR = SWEEP_DIR.parent / "sweep_results"
LARGE_MODELS_FILE = SWEEP_DIR / "large_models.json"


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


def check_gpu_health():
    """Check GPU is in a healthy state. Returns (ok, message)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,gpu_bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return False, f"nvidia-smi failed: {result.stderr[:200]}"
        line = result.stdout.strip().split("\n")[0]
        used, total, _ = [x.strip() for x in line.split(",")]
        used_mb, total_mb = int(used), int(total)
        usage_pct = used_mb / total_mb * 100
        if usage_pct > 80:
            return False, f"GPU memory {usage_pct:.0f}% used ({used_mb}/{total_mb} MiB)"
        return True, f"GPU OK ({used_mb}/{total_mb} MiB, {usage_pct:.0f}%)"
    except Exception as e:
        return False, f"GPU health check error: {e}"


def kill_gpu_zombies():
    """Kill any orphaned Python processes holding GPU memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                pid = int(line.strip())
                # Only kill worker.py subprocesses, not ourselves
                try:
                    cmdline = open(f"/proc/{pid}/cmdline").read()
                    if "worker.py" in cmdline:
                        os.kill(pid, signal.SIGKILL)
                        print(f"  Killed zombie GPU process {pid}", flush=True)
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    pass
    except Exception:
        pass


class WorkerHandle:
    """Tracks a running worker subprocess."""
    __slots__ = ("proc", "spec", "mode", "pass_num", "timeout_s",
                 "start_time", "kill_stage", "stdout_path", "stderr_path")

    def __init__(self, proc, spec, mode, pass_num, timeout_s, start_time,
                 stdout_path, stderr_path):
        self.proc = proc
        self.spec = spec
        self.mode = mode
        self.pass_num = pass_num
        self.timeout_s = timeout_s
        self.start_time = start_time
        self.kill_stage = 0   # 0=running, 1=TERM sent, 2=KILL sent, 3=abandoned
        self.stdout_path = stdout_path
        self.stderr_path = stderr_path


def spawn_worker(python_bin, spec, pass_num, device, mode, timeout_s,
                 dynamic=False):
    """Spawn a worker subprocess in its own process group.

    Uses temp files for stdout/stderr to avoid pipe buffer deadlocks.
    Returns a WorkerHandle for non-blocking tracking.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    worker_script = CUSTOM_WORKER_SCRIPT if spec.get("source") == "custom" else WORKER_SCRIPT
    cmd = [
        python_bin, str(worker_script),
        "--model-json", json.dumps(spec),
        "--pass-num", str(pass_num),
        "--device", device,
        "--mode", mode,
    ]
    if dynamic:
        cmd.extend(["--dynamic", dynamic])

    # Use temp files instead of PIPE to avoid buffer deadlocks
    stdout_fd, stdout_path = tempfile.mkstemp(prefix=f"sweep_out_{spec['name']}_", suffix=".json")
    stderr_fd, stderr_path = tempfile.mkstemp(prefix=f"sweep_err_{spec['name']}_", suffix=".log")

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=stdout_fd, stderr=stderr_fd,
        preexec_fn=os.setsid,  # new process group for clean kill
    )
    # Close our copy of the FDs (the child has its own)
    os.close(stdout_fd)
    os.close(stderr_fd)

    return WorkerHandle(proc, spec, mode, pass_num, timeout_s, time.time(),
                        stdout_path, stderr_path)


def harvest_worker(handle):
    """Collect result from a completed worker. Cleans up temp files."""
    wall_time = time.time() - handle.start_time
    stdout, stderr = "", ""

    try:
        with open(handle.stdout_path) as f:
            stdout = f.read()
    except Exception:
        pass
    try:
        with open(handle.stderr_path) as f:
            stderr = f.read()
    except Exception:
        pass

    # Cleanup temp files
    for p in (handle.stdout_path, handle.stderr_path):
        try:
            os.unlink(p)
        except OSError:
            pass

    if handle.proc.returncode == 0 and stdout.strip():
        try:
            result = json.loads(stdout.strip().split("\n")[-1])
            result["wall_time_s"] = round(wall_time, 2)
            return result
        except (json.JSONDecodeError, IndexError):
            # Worker printed non-JSON output — capture both streams for debugging
            return {
                "name": handle.spec["name"],
                "source": handle.spec["source"],
                "mode": handle.mode,
                "pass": handle.pass_num,
                "status": "worker_error",
                "error": f"Bad JSON from worker stdout",
                "stdout_tail": stdout[-300:],
                "stderr_tail": (stderr or "")[-300:],
                "returncode": 0,
                "wall_time_s": round(wall_time, 2),
            }

    # Worker crashed — extract the actual error from stderr, stripping
    # subprocess command strings that leak implementation details.
    error_text = (stderr or "no output")[-500:]
    # If stderr contains a CalledProcessError repr, extract the actual error
    # from the last traceback line instead of the command string.
    if error_text.startswith("Command '") or "CalledProcessError" in error_text:
        lines = stderr.strip().splitlines() if stderr else []
        # Look for the last meaningful error line (skip command repr)
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("Command '") and not line.startswith("Traceback"):
                error_text = line[:500]
                break

    return {
        "name": handle.spec["name"],
        "source": handle.spec["source"],
        "mode": handle.mode,
        "pass": handle.pass_num,
        "status": "worker_error",
        "error": error_text,
        "returncode": handle.proc.returncode,
        "wall_time_s": round(wall_time, 2),
    }


def timeout_result(handle):
    """Build a timeout result, extracting phase from stderr if possible."""
    phase = "unknown"
    try:
        with open(handle.stderr_path) as f:
            for line in reversed(f.read().splitlines()):
                if line.startswith("PHASE:"):
                    phase = line.split(":", 1)[1]
                    break
    except Exception:
        pass

    # Cleanup temp files
    for p in (handle.stdout_path, handle.stderr_path):
        try:
            os.unlink(p)
        except OSError:
            pass

    return {
        "name": handle.spec["name"],
        "source": handle.spec["source"],
        "mode": handle.mode,
        "pass": handle.pass_num,
        "status": "timeout" if handle.kill_stage < 3 else "zombie",
        "phase_at_timeout": phase,
        "wall_time_s": handle.timeout_s,
    }


def escalating_kill(handle):
    """Escalate kill: SIGTERM → SIGKILL on process group. Returns True if process is dead."""
    try:
        pgid = os.getpgid(handle.proc.pid)
    except (ProcessLookupError, OSError):
        return True  # already dead

    if handle.kill_stage == 0:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return True
        handle.kill_stage = 1
        return False
    elif handle.kill_stage == 1:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            return True
        handle.kill_stage = 2
        return False
    else:
        # Stage 2+: already sent SIGKILL, mark as abandoned
        handle.kill_stage = 3
        return True


def load_checkpoint(checkpoint_file):
    """Load completed (name, mode) pairs from JSONL checkpoint file."""
    completed = {}  # (name, mode) -> result dict
    if not os.path.exists(checkpoint_file):
        return completed
    with open(checkpoint_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["name"], r["mode"])
                completed[key] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_pass(python_bin, specs, pass_num, device, modes, workers, timeout_s,
             checkpoint_file=None, resume_from=None, dynamic=False,
             timeout_overrides=None, skip_models=None):
    """Run a full pass using a non-blocking poll loop with process group isolation.

    The orchestrator never blocks on any single worker. Timed-out workers are
    killed with escalating signals (TERM → KILL → abandon). GPU pressure
    reduces parallelism instead of aborting the sweep.

    Args:
        checkpoint_file: Path to JSONL file for incremental result saving
        resume_from: Dict of (name, mode) -> result to skip
        timeout_overrides: Dict of model_name -> timeout_s for large models
        skip_models: Set of model names to skip entirely
    """
    # Escalation timings (seconds after timeout)
    TERM_GRACE = 5     # wait this long after timeout before SIGTERM
    KILL_GRACE = 10    # wait this long after SIGTERM before SIGKILL
    ABANDON_GRACE = 20 # wait this long after SIGKILL before abandoning

    # GPU recovery settings
    GPU_RECOVERY_WAIT = 30    # seconds between GPU health checks
    GPU_RECOVERY_RETRIES = 4  # max retries (total wait = 4 × 30s = 2 min)

    # Build work queue
    pending = deque()
    for spec in specs:
        if skip_models and spec["name"] in skip_models:
            continue
        for mode in modes:
            key = (spec["name"], mode)
            if resume_from and key in resume_from:
                continue  # Already done
            pending.append((spec, mode))

    # Load any already-completed results
    results = list(resume_from.values()) if resume_from else []
    skipped = len(results)
    completed = skipped
    total = len(pending) + skipped

    if skipped:
        print(f"  Resuming: {skipped} already completed, {len(pending)} remaining")

    # Open checkpoint file for appending
    ckpt_fh = open(checkpoint_file, "a") if checkpoint_file else None

    active = {}        # pid -> WorkerHandle
    max_workers = workers
    current_max = max_workers
    gpu_degraded = False

    try:
        while pending or active:
            now = time.time()

            # ── 1. Harvest completed workers ──
            for pid in list(active):
                handle = active[pid]
                ret = handle.proc.poll()  # non-blocking
                if ret is not None:
                    # If we were killing this worker (timeout), record as timeout
                    if handle.kill_stage > 0:
                        result = timeout_result(handle)
                    else:
                        result = harvest_worker(handle)
                    results.append(result)
                    completed += 1
                    if ckpt_fh:
                        ckpt_fh.write(json.dumps(result) + "\n")
                        ckpt_fh.flush()
                    _print_progress(completed, total, result)
                    del active[pid]

            # ── 2. Handle timed-out workers with escalating kill ──
            for pid in list(active):
                handle = active[pid]
                elapsed = now - handle.start_time

                if handle.kill_stage == 0 and elapsed > handle.timeout_s:
                    # Stage 0 → 1: send SIGTERM
                    escalating_kill(handle)

                elif handle.kill_stage == 1 and elapsed > handle.timeout_s + TERM_GRACE:
                    # Stage 1 → 2: SIGTERM didn't work, send SIGKILL
                    escalating_kill(handle)

                elif handle.kill_stage >= 2 and elapsed > handle.timeout_s + TERM_GRACE + KILL_GRACE:
                    # Check if finally dead
                    ret = handle.proc.poll()
                    if ret is not None:
                        result = timeout_result(handle)
                        results.append(result)
                        completed += 1
                        if ckpt_fh:
                            ckpt_fh.write(json.dumps(result) + "\n")
                            ckpt_fh.flush()
                        _print_progress(completed, total, result)
                        del active[pid]
                    elif elapsed > handle.timeout_s + TERM_GRACE + KILL_GRACE + ABANDON_GRACE:
                        # Truly stuck — abandon and move on
                        handle.kill_stage = 3
                        result = timeout_result(handle)
                        results.append(result)
                        completed += 1
                        if ckpt_fh:
                            ckpt_fh.write(json.dumps(result) + "\n")
                            ckpt_fh.flush()
                        _print_progress(completed, total, result)
                        del active[pid]
                        print(f"  ✗ Abandoned zombie: {handle.spec['name']} "
                              f"({handle.mode})", flush=True)

            # ── 3. GPU health check ──
            if device == "cuda" and pending and len(active) < current_max:
                ok, msg = check_gpu_health()
                if not ok:
                    kill_gpu_zombies()
                    recovered = False
                    for attempt in range(GPU_RECOVERY_RETRIES):
                        time.sleep(GPU_RECOVERY_WAIT)
                        kill_gpu_zombies()
                        ok, msg = check_gpu_health()
                        if ok:
                            recovered = True
                            break
                        print(f"  ⚠ GPU recovery attempt {attempt + 1}/"
                              f"{GPU_RECOVERY_RETRIES}: {msg}", flush=True)
                    if not recovered:
                        # Degrade to fewer workers instead of aborting
                        current_max = max(1, current_max // 2)
                        gpu_degraded = True
                        print(f"  ⚠ GPU pressure: reducing workers to "
                              f"{current_max}", flush=True)

            # ── 4. Recover worker count if GPU recovered ──
            if gpu_degraded and device == "cuda" and current_max < max_workers:
                ok, _ = check_gpu_health()
                if ok:
                    current_max = min(current_max * 2, max_workers)
                    if current_max >= max_workers:
                        gpu_degraded = False
                        print(f"  ✓ GPU recovered: workers back to "
                              f"{current_max}", flush=True)

            # ── 5. Spawn new workers to fill available slots ──
            while pending and len(active) < current_max:
                spec, mode = pending.popleft()
                model_timeout = timeout_s
                if timeout_overrides and spec["name"] in timeout_overrides:
                    model_timeout = timeout_overrides[spec["name"]]
                try:
                    handle = spawn_worker(
                        python_bin, spec, pass_num, device, mode,
                        model_timeout, dynamic
                    )
                    active[handle.proc.pid] = handle
                except Exception as e:
                    # If we can't even spawn, record as error and continue
                    result = {
                        "name": spec["name"], "source": spec["source"],
                        "mode": mode, "pass": pass_num,
                        "status": "launch_error", "error": str(e)[:500],
                        "wall_time_s": 0,
                    }
                    results.append(result)
                    completed += 1
                    if ckpt_fh:
                        ckpt_fh.write(json.dumps(result) + "\n")
                        ckpt_fh.flush()
                    _print_progress(completed, total, result)

            # ── 6. Poll interval ──
            if active:
                time.sleep(5)

    finally:
        # Cleanup: kill all remaining active workers
        for pid, handle in list(active.items()):
            try:
                pgid = os.getpgid(handle.proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            # Wait briefly for process to die, then cleanup temp files
            try:
                handle.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            for p in (handle.stdout_path, handle.stderr_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        if ckpt_fh:
            ckpt_fh.close()

    return results


def _print_progress(completed, total, result):
    """Print progress line for each completed model."""
    name = result.get("name", "?")
    source = result.get("source", "?")
    mode = result.get("mode", "?")
    status = result.get("status", "?")
    wall = result.get("wall_time_s", 0)

    status_str = {
        "full_graph": "FULL_GRAPH",
        "graph_break": "BREAK",
        "ok": "OK",
        "pass": "PASS",
        "mismatch": "MISMATCH",
        "create_error": "CREATE_ERR",
        "eager_error": "EAGER_ERR",
        "compile_error": "COMPILE_ERR",
        "timeout": "TIMEOUT",
        "worker_error": "WORKER_ERR",
        "launch_error": "LAUNCH_ERR",
        "zombie": "ZOMBIE",
    }.get(status, status.upper())

    extra = ""
    if result.get("graph_break_count"):
        extra = f" ({result['graph_break_count']} breaks)"
    if result.get("phase_at_timeout"):
        extra = f" (in {result['phase_at_timeout']})"

    print(f"  [{completed:>4}/{total:>4}] {source}/{name:<30} {mode:<6} "
          f"{status_str:<12} {wall:>5.1f}s{extra}", flush=True)


def _log_versions(python_bin, output_dir):
    """Detect PyTorch, transformers, and diffusers versions.

    Returns a dict with version info, embedded into results metadata by the caller.
    """
    script = (
        "import json, sys; d = {}; "
        "import torch; d['torch'] = torch.__version__; d['torch_git'] = torch.version.git_version; "
        "d['transformers'] = None; d['diffusers'] = None; "
        "exec('try:\\n import transformers\\n d[\"transformers\"] = transformers.__version__\\nexcept ImportError: pass'); "
        "exec('try:\\n import diffusers\\n d[\"diffusers\"] = diffusers.__version__\\nexcept ImportError: pass'); "
        "d['python'] = sys.version; print(json.dumps(d))"
    )
    try:
        result = subprocess.run(
            [python_bin, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            versions = json.loads(result.stdout.strip())
            print(f"Environment: torch={versions.get('torch')}, "
                  f"transformers={versions.get('transformers')}, "
                  f"diffusers={versions.get('diffusers')}")
            return versions
        else:
            print(f"WARNING: Could not detect library versions: {result.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: Version check failed: {e}")
    return None


def run_sweep(args):
    """Main sweep logic."""
    python_bin = _resolve_python(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    timeout_large = args.timeout * 3  # 3x base timeout for large models
    timeout_overrides = {name: timeout_large for name in large_registry}
    if large_registry:
        print(f"Large model registry: {len(large_registry)} models will use {timeout_large}s timeout")

    # ── Load skip list (toxic models) — auto-loaded from config ──
    skip_models = _load_skip_models()

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

    identify_start = time.perf_counter()
    identify_results = run_pass(
        python_bin, multi_specs, pass_num=1, device=args.device, modes=modes,
        workers=args.workers, timeout_s=args.timeout,
        checkpoint_file=identify_ckpt, resume_from=resume_from,
        dynamic=dynamic, timeout_overrides=timeout_overrides,
        skip_models=skip_models,
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
            skip_models=skip_models,
        )
        identify_results.extend(single_results)

    identify_time = time.perf_counter() - identify_start

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
            skip_models=skip_models,
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
    explain_names = set()
    for r in identify_results:
        if r.get("status") == "graph_break":
            explain_names.add(r["name"])

    explain_specs = [s for s in specs if s["name"] in explain_names]
    print(f"\n→ {len(explain_specs)} models need explain pass (will test {modes})")

    # ══════════════════════════════════════════════════════════════════════
    # EXPLAIN: Detailed analysis (broken models only)
    # ══════════════════════════════════════════════════════════════════════
    if explain_specs and not args.identify_only:
        explain_ckpt = str(output_dir / "explain_checkpoint.jsonl")

        # Load checkpoint for resume
        resume_from = {}
        if args.resume and os.path.exists(explain_ckpt):
            resume_from = load_checkpoint(explain_ckpt)
            print(f"Loaded {len(resume_from)} completed explain results from checkpoint")

        print(f"\n{'=' * 70}")
        print(f"EXPLAIN: Detailed analysis — {len(explain_specs)} models × {len(modes)} modes ({modes})")
        print(f"{'=' * 70}")

        explain_start = time.perf_counter()
        explain_results = run_pass(
            python_bin, explain_specs, pass_num=2, device=args.device, modes=modes,
            workers=args.workers, timeout_s=args.timeout * 2,  # more time for explain()
            checkpoint_file=explain_ckpt, resume_from=resume_from,
        )
        explain_time = time.perf_counter() - explain_start

        # Save explain results
        explain_meta = {
            "pass": "explain",
            "device": args.device,
            "modes": modes,
            "workers": args.workers,
            "total_time_s": round(explain_time, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if version_info:
            explain_meta["versions"] = version_info
        explain_output = {
            "metadata": explain_meta,
            "results": explain_results,
        }
        explain_file = output_dir / "explain_results.json"
        with open(explain_file, "w") as f:
            json.dump(explain_output, f, indent=2)

        print(f"\nExplain pass complete: {explain_time:.1f}s")
        print(f"Saved to {explain_file}")
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
    """Run explain pass only, from prior identify results."""
    python_bin = _resolve_python(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect versions
    version_info = _log_versions(python_bin, output_dir)

    # Load identify results and filter to graph_break models
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
    explain_names = set()
    for r in identify_results:
        if r.get("status") == "graph_break":
            explain_names.add(r["name"])

    # Rebuild specs from identify results
    specs = []
    seen = set()
    for r in identify_results:
        if r["name"] in explain_names and r["name"] not in seen:
            seen.add(r["name"])
            spec = {"name": r["name"], "source": r["source"]}
            for k in ["hf_class", "hf_config", "input_type",
                       "constructor_args", "inputs", "variant"]:
                if k in r:
                    spec[k] = r[k]
            # Derive variant from name if not in identify results (older sweeps)
            if "variant" not in spec and r["source"] == "hf":
                name = r["name"]
                if name.endswith("ForCausalLM"):
                    spec["variant"] = "causal_lm"
                elif name.endswith("ForConditionalGeneration"):
                    spec["variant"] = "conditional_generation"
            specs.append(spec)
    print(f"Loaded {len(specs)} broken models from {args.file}")

    modes = ["eval", "train"]
    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Modes: {modes}, Timeout: {args.timeout * 2}s (2x for explain)")

    explain_ckpt = str(output_dir / "explain_checkpoint.jsonl")
    resume_from = {}
    if args.resume and os.path.exists(explain_ckpt):
        resume_from = load_checkpoint(explain_ckpt)
        print(f"Loaded {len(resume_from)} completed explain results from checkpoint")

    print(f"\n{'=' * 70}")
    print(f"EXPLAIN: Detailed analysis — {len(specs)} models × {len(modes)} modes ({modes})")
    print(f"{'=' * 70}")

    explain_start = time.perf_counter()
    explain_results = run_pass(
        python_bin, specs, pass_num=2, device=args.device, modes=modes,
        workers=args.workers, timeout_s=args.timeout * 2,
        checkpoint_file=explain_ckpt, resume_from=resume_from,
    )
    explain_time = time.perf_counter() - explain_start

    explain_meta = {
        "pass": "explain",
        "device": args.device,
        "modes": modes,
        "workers": args.workers,
        "total_time_s": round(explain_time, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if version_info:
        explain_meta["versions"] = version_info
    explain_output = {
        "metadata": explain_meta,
        "results": explain_results,
    }
    explain_file = output_dir / "explain_results.json"
    with open(explain_file, "w") as f:
        json.dump(explain_output, f, indent=2)

    print(f"\nExplain pass complete: {explain_time:.1f}s")
    print(f"Saved to {explain_file}")


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
    output_dir = Path(args.output_dir)
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


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass graph break sweep orchestrator",
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


if __name__ == "__main__":
    main()
