#!/usr/bin/env python3
"""Sweep orchestrator — runs the two-pass graph break sweep.

Manages subprocess workers, parallel execution, timeouts, and result merging.

Resilience features:
  - Checkpoint: each result written to JSONL immediately on completion
  - Resume: --resume skips already-completed (name, mode) pairs
  - Batched submission: processes work items in batches to limit memory
  - SIGTERM handler: flushes checkpoint on graceful shutdown
  - Two-tier timeout: models that timeout are auto-retried with extended
    timeout and added to large_models.json for future runs

Optimizations:
  - Pass 1 runs eval-only by default (fast identification)
  - Pass 2 runs both eval+train on broken models only
  - 16 parallel workers (A100 has headroom)
  - --source all auto-excludes unconfigured diffusers models
  - Known large models get extended timeout upfront (no wasted short attempt)

Usage:
  # Full sweep (auto-resumes from checkpoint if exists)
  python run_sweep.py --device cuda --python ~/envs/graph-break-corpus/bin/python

  # Explicit resume after crash
  python run_sweep.py --resume --device cuda --python ~/envs/graph-break-corpus/bin/python

  # Pass 2 only (from prior pass 1 results)
  python run_sweep.py --pass2-from results_pass1.json --device cuda

  # Custom model list
  python run_sweep.py --models models.json --device cuda

  # Two-shape validation on clean models from a prior sweep
  python run_sweep.py --validate-from sweep_results/pass1_results.json --device cuda
"""
import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path


SWEEP_DIR = Path(__file__).resolve().parent
WORKER_SCRIPT = SWEEP_DIR / "worker.py"
DEFAULT_OUTPUT_DIR = SWEEP_DIR.parent / "sweep_results"
LARGE_MODELS_FILE = SWEEP_DIR / "large_models.json"


def load_large_models(path=None):
    """Load the large model registry — models that need extended timeouts."""
    path = path or LARGE_MODELS_FILE
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


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
                 trace_dir=None, dynamic=False):
    """Spawn a worker subprocess in its own process group.

    Uses temp files for stdout/stderr to avoid pipe buffer deadlocks.
    Returns a WorkerHandle for non-blocking tracking.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if trace_dir:
        model_trace_dir = os.path.join(trace_dir, f"{spec['name']}_{mode}")
        os.makedirs(model_trace_dir, exist_ok=True)
        env["TORCH_TRACE"] = model_trace_dir

    cmd = [
        python_bin, str(WORKER_SCRIPT),
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
             trace_dir=None, checkpoint_file=None, resume_from=None, dynamic=False,
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
                        model_timeout, trace_dir, dynamic
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
        "clean": "CLEAN",
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
    """Log PyTorch, transformers, and diffusers versions at sweep start."""
    script = (
        "import json, sys; d = {}; "
        "import torch; d['torch'] = torch.__version__; d['torch_git'] = torch.version.git_version; "
        "try:\n import transformers; d['transformers'] = transformers.__version__\n"
        "except ImportError: d['transformers'] = None\n"
        "try:\n import diffusers; d['diffusers'] = diffusers.__version__\n"
        "except ImportError: d['diffusers'] = None\n"
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
            version_file = output_dir / "versions.json"
            with open(version_file, "w") as f:
                json.dump(versions, f, indent=2)
            return versions
        else:
            print(f"WARNING: Could not detect library versions: {result.stderr.strip()}")
    except Exception as e:
        print(f"WARNING: Version check failed: {e}")
    return None


def run_sweep(args):
    """Main sweep logic."""
    python_bin = args.python or sys.executable
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
    elif args.pass2_from:
        # Load pass 1 results and filter to graph_break models
        try:
            with open(args.pass2_from) as f:
                pass1_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {args.pass2_from}: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print(f"ERROR: File not found: {args.pass2_from}")
            sys.exit(1)
        pass1_results = pass1_data if isinstance(pass1_data, list) else pass1_data.get("results", [])
        broken_names = set()
        for r in pass1_results:
            if r.get("status") == "graph_break":
                broken_names.add(r["name"])
        # Rebuild specs from pass 1 results
        specs = []
        seen = set()
        for r in pass1_results:
            if r["name"] in broken_names and r["name"] not in seen:
                seen.add(r["name"])
                specs.append({"name": r["name"], "source": r["source"]})
                # Copy any extra fields from the original spec
                for k in ["hf_class", "hf_config", "input_type", "constructor_args", "inputs"]:
                    if k in r:
                        specs[-1][k] = r[k]
        print(f"Loaded {len(specs)} broken models from {args.pass2_from}")
    else:
        # Enumerate from source
        from models import enumerate_timm, enumerate_hf, enumerate_diffusers, enumerate_all
        if args.source == "all":
            specs = enumerate_all()
            # Count by source for reporting
            by_src = {}
            for s in specs:
                by_src[s["source"]] = by_src.get(s["source"], 0) + 1
            src_detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_src.items()))
            print(f"Enumerated {len(specs)} models from all sources ({src_detail})")
        elif args.source == "hf+diffusers":
            specs = enumerate_hf()
            diffusers_specs = [m for m in enumerate_diffusers() if m.get("has_config", False)]
            specs.extend(diffusers_specs)
            print(f"Enumerated {len(specs)} models (hf: {len(specs) - len(diffusers_specs)}, diffusers: {len(diffusers_specs)})")
        elif args.source == "timm":
            specs = enumerate_timm()
            print(f"Enumerated {len(specs)} models from timm")
        elif args.source == "hf":
            specs = enumerate_hf()
            print(f"Enumerated {len(specs)} models from hf")
        elif args.source == "diffusers":
            specs = enumerate_diffusers()
            print(f"Enumerated {len(specs)} models from diffusers")

    # Apply limit
    if args.limit:
        specs = specs[:args.limit]
        print(f"Limited to {len(specs)} models")

    # Filter by has_config for diffusers
    if args.has_config_only:
        before = len(specs)
        specs = [s for s in specs if s.get("has_config", True)]
        print(f"Filtered to {len(specs)} models with configs (from {before})")

    pass1_modes = args.pass1_modes
    pass2_modes = args.pass2_modes
    print(f"Device: {args.device}, Workers: {args.workers}, "
          f"Pass1 modes: {pass1_modes}, Pass2 modes: {pass2_modes}, Timeout: {args.timeout}s")

    # ── Load large model registry for tiered timeouts ──
    large_models_path = Path(args.large_models) if args.large_models else LARGE_MODELS_FILE
    large_registry = load_large_models(large_models_path)
    timeout_overrides = {name: args.timeout_large for name in large_registry}
    if large_registry:
        print(f"Large model registry: {len(large_registry)} models will use {args.timeout_large}s timeout")

    # ── Load skip list (toxic models) ──
    skip_models = set()
    if args.skip_models and os.path.exists(args.skip_models):
        with open(args.skip_models) as f:
            skip_models = set(json.load(f))
        print(f"Skip list: {len(skip_models)} models will be skipped")
    print()

    # ── Update watchdog state file with full details ──
    total_work_items = len(specs) * len(pass1_modes)
    state_file = output_dir / "sweep_state.json"
    with open(state_file) as f:
        state = json.load(f)
    state.update({
        "status": "running",
        "total_models": len(specs),
        "total_work_items": total_work_items,
        "modes": pass1_modes,
    })
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════
    # PASS 1: Fast identification (eval-only by default)
    # ══════════════════════════════════════════════════════════════════════
    if not args.pass2_from:
        pass1_ckpt = str(output_dir / "pass1_checkpoint.jsonl")

        # Load checkpoint for resume
        resume_from = {}
        if args.resume and os.path.exists(pass1_ckpt):
            resume_from = load_checkpoint(pass1_ckpt)
            print(f"Loaded {len(resume_from)} completed results from checkpoint")

        print(f"{'=' * 70}")
        print(f"PASS 1: Identifying graph breaks (fullgraph=True) — {len(specs)} models × {len(pass1_modes)} modes ({pass1_modes})")
        print(f"{'=' * 70}")

        pass1_start = time.perf_counter()
        pass1_results = run_pass(
            python_bin, specs, pass_num=1, device=args.device, modes=pass1_modes,
            workers=args.workers, timeout_s=args.timeout,
            checkpoint_file=pass1_ckpt, resume_from=resume_from,
            dynamic=args.dynamic, timeout_overrides=timeout_overrides,
            skip_models=skip_models,
        )
        pass1_time = time.perf_counter() - pass1_start

        # Save pass 1 results (full JSON for analysis)
        pass1_output = {
            "metadata": {
                "pass": 1,
                "device": args.device,
                "modes": pass1_modes,
                "workers": args.workers,
                "timeout_s": args.timeout,
                "total_time_s": round(pass1_time, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "python": python_bin,
                "dynamic": args.dynamic,
            },
            "results": pass1_results,
        }
        pass1_file = output_dir / "pass1_results.json"
        with open(pass1_file, "w") as f:
            json.dump(pass1_output, f, indent=2)

        # Summarize pass 1
        by_status = {}
        for r in pass1_results:
            by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1

        print(f"\nPass 1 complete: {pass1_time:.1f}s")
        for status, count in sorted(by_status.items()):
            print(f"  {status}: {count}")
        print(f"Saved to {pass1_file}")

        # ── Auto-retry: re-run timed-out models with extended timeout ──
        timeout_results = [r for r in pass1_results if r.get("status") == "timeout"]
        # Only retry models that aren't already using the large timeout
        new_timeouts = [r for r in timeout_results if r["name"] not in large_registry]
        if new_timeouts and not args.no_auto_retry:
            timeout_names = {r["name"] for r in new_timeouts}
            retry_specs = [s for s in specs if s["name"] in timeout_names]

            print(f"\n{'─' * 70}")
            print(f"AUTO-RETRY: {len(retry_specs)} timed-out models with extended timeout ({args.timeout_large}s)")
            print(f"{'─' * 70}")

            # Build retry overrides — all get the large timeout
            retry_overrides = {s["name"]: args.timeout_large for s in retry_specs}

            retry_start = time.perf_counter()
            retry_results = run_pass(
                python_bin, retry_specs, pass_num=1, device=args.device, modes=pass1_modes,
                workers=max(1, args.workers // 2),  # fewer workers for large models
                timeout_s=args.timeout_large,
                checkpoint_file=None,  # don't mix with main checkpoint
                dynamic=args.dynamic, timeout_overrides=retry_overrides,
            )
            retry_time = time.perf_counter() - retry_start

            # Summarize retry results
            retry_by_status = {}
            for r in retry_results:
                retry_by_status[r.get("status", "unknown")] = retry_by_status.get(r.get("status", "unknown"), 0) + 1
            print(f"\nRetry complete: {retry_time:.1f}s")
            for status, count in sorted(retry_by_status.items()):
                print(f"  {status}: {count}")

            # Replace timeout results in pass1_results with retry results
            retry_index = {(r["name"], r.get("mode", "eval")): r for r in retry_results}
            updated_count = 0
            for i, r in enumerate(pass1_results):
                key = (r["name"], r.get("mode", "eval"))
                if key in retry_index:
                    pass1_results[i] = retry_index[key]
                    updated_count += 1

            # Update checkpoint with retry results
            if os.path.exists(pass1_ckpt):
                # Rewrite checkpoint with updated results
                all_completed = {}
                for r in pass1_results:
                    key = (r["name"], r.get("mode", "eval"))
                    all_completed[key] = r
                with open(pass1_ckpt, "w") as f:
                    for r in all_completed.values():
                        f.write(json.dumps(r) + "\n")

            # Update large model registry — add models that resolved (not still timeout)
            newly_large = []
            for r in retry_results:
                if r.get("status") != "timeout":
                    # Model completed with extended timeout → it's a "large" model
                    large_registry[r["name"]] = {
                        "source": r.get("source", "unknown"),
                        "timeout_tier": "large",
                        "resolved_status": r.get("status"),
                        "wall_time_s": r.get("wall_time_s"),
                        "discovered": time.strftime("%Y-%m-%d"),
                    }
                    newly_large.append(r["name"])
                else:
                    # Still timed out even with extended timeout — record as very_large
                    large_registry[r["name"]] = {
                        "source": r.get("source", "unknown"),
                        "timeout_tier": "very_large",
                        "phase_at_timeout": r.get("phase_at_timeout", "unknown"),
                        "discovered": time.strftime("%Y-%m-%d"),
                    }
            save_large_models(large_registry, large_models_path)
            print(f"\n  Updated large model registry: {len(newly_large)} newly resolved, "
                  f"{len(large_registry)} total entries")

            # Re-save pass 1 results with retry data merged
            pass1_output["results"] = pass1_results
            pass1_output["metadata"]["retry_count"] = len(retry_specs)
            pass1_output["metadata"]["timeout_large_s"] = args.timeout_large
            with open(pass1_file, "w") as f:
                json.dump(pass1_output, f, indent=2)

            # Recompute status summary
            by_status = {}
            for r in pass1_results:
                by_status[r.get("status", "unknown")] = by_status.get(r.get("status", "unknown"), 0) + 1
            print(f"\nUpdated pass 1 summary:")
            for status, count in sorted(by_status.items()):
                print(f"  {status}: {count}")

        # Identify broken models for pass 2
        broken_names = set()
        for r in pass1_results:
            if r.get("status") == "graph_break":
                broken_names.add(r["name"])

        broken_specs = [s for s in specs if s["name"] in broken_names]
        print(f"\n→ {len(broken_specs)} models need pass 2 (will test {pass2_modes})")
    else:
        broken_specs = specs
        pass1_results = None

    # ══════════════════════════════════════════════════════════════════════
    # PASS 2: Detailed analysis (broken models only, eval+train)
    # ══════════════════════════════════════════════════════════════════════
    if broken_specs and not args.pass1_only:
        pass2_ckpt = str(output_dir / "pass2_checkpoint.jsonl")

        # Load checkpoint for resume
        resume_from = {}
        if args.resume and os.path.exists(pass2_ckpt):
            resume_from = load_checkpoint(pass2_ckpt)
            print(f"Loaded {len(resume_from)} completed pass 2 results from checkpoint")

        print(f"\n{'=' * 70}")
        print(f"PASS 2: Detailed analysis — {len(broken_specs)} models × {len(pass2_modes)} modes ({pass2_modes})")
        print(f"{'=' * 70}")

        trace_dir = str(output_dir / "traces") if not args.skip_traces else None
        # Don't nuke existing traces on resume
        if trace_dir and os.path.exists(trace_dir) and not args.resume:
            shutil.rmtree(trace_dir)

        pass2_start = time.perf_counter()
        pass2_results = run_pass(
            python_bin, broken_specs, pass_num=2, device=args.device, modes=pass2_modes,
            workers=args.workers, timeout_s=args.timeout * 2,  # more time for explain()
            trace_dir=trace_dir,
            checkpoint_file=pass2_ckpt, resume_from=resume_from,
        )
        pass2_time = time.perf_counter() - pass2_start

        # Save pass 2 results
        pass2_output = {
            "metadata": {
                "pass": 2,
                "device": args.device,
                "modes": pass2_modes,
                "workers": args.workers,
                "total_time_s": round(pass2_time, 1),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "traces": trace_dir or "disabled",
            },
            "results": pass2_results,
        }
        pass2_file = output_dir / "pass2_results.json"
        with open(pass2_file, "w") as f:
            json.dump(pass2_output, f, indent=2)

        print(f"\nPass 2 complete: {pass2_time:.1f}s")
        print(f"Saved to {pass2_file}")

        # Run tlparse if traces were collected
        if trace_dir and os.path.exists(trace_dir):
            print(f"\n{'─' * 50}")
            print("Running tlparse on traces...")
            tlparse_dir = str(output_dir / "tlparse_output")
            if os.path.exists(tlparse_dir):
                shutil.rmtree(tlparse_dir)
            _run_tlparse(trace_dir, tlparse_dir)
    else:
        pass2_results = []
        if args.pass1_only:
            print("\nSkipping pass 2 (--pass1-only)")

    # ══════════════════════════════════════════════════════════════════════
    # MERGED CORPUS
    # ══════════════════════════════════════════════════════════════════════
    if pass1_results and pass2_results:
        corpus = _build_corpus(pass1_results, pass2_results, args)
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


def _build_corpus(pass1_results, pass2_results, args):
    """Merge pass 1 and pass 2 results into a unified corpus."""
    # Index pass 2 by (name, mode)
    p2_index = {}
    for r in pass2_results:
        key = (r["name"], r["mode"])
        p2_index[key] = r

    models = []
    for r in pass1_results:
        record = dict(r)
        key = (r["name"], r["mode"])
        if key in p2_index:
            record["pass2"] = p2_index[key]
        models.append(record)

    # Summary stats
    by_status = {}
    for r in pass1_results:
        s = r.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    return {
        "metadata": {
            "device": args.device,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "methodology": "Two-pass: fullgraph(all) → explain+TORCH_TRACE(broken only)",
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
    breaks = [m for m in corpus["models"] if m.get("status") == "graph_break" and "pass2" in m]
    if breaks:
        print(f"\nGraph break models ({len(breaks)}):")
        for m in breaks:
            p2 = m["pass2"]
            name = m["name"]
            mode = m["mode"]
            bc = p2.get("graph_break_count", "?")
            gc = p2.get("graph_count", "?")
            reasons = p2.get("break_reasons", [])
            top_reason = reasons[0]["reason"][:80] if reasons else "(no break reasons captured)"
            print(f"  {name:<30} {mode:<6} {bc} breaks, {gc} graphs")
            print(f"    → {top_reason}")


def _run_tlparse(trace_dir, output_dir):
    """Run tlparse on all trace subdirectories."""
    for model_dir in sorted(Path(trace_dir).iterdir()):
        if not model_dir.is_dir():
            continue
        parsed_dir = os.path.join(output_dir, model_dir.name)
        try:
            result = subprocess.run(
                ["tlparse", str(model_dir), "-o", parsed_dir],
                capture_output=True, text=True, timeout=60,
            )
            if os.path.exists(parsed_dir):
                break_files = list(Path(parsed_dir).rglob("dynamo_graph_break_reason_*.txt"))
                print(f"  {model_dir.name}: {len(break_files)} break reason files")
        except Exception as e:
            print(f"  {model_dir.name}: tlparse error — {e}")


def run_validation(args):
    """Run two-shape validation sweep (pass 3) on clean models."""
    python_bin = args.python or sys.executable
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models to validate
    if args.validate_from:
        with open(args.validate_from) as f:
            data = json.load(f)
        results = data if isinstance(data, list) else data.get("results", [])
        # Only validate clean models
        specs = []
        seen = set()
        for r in results:
            if r.get("status") == "clean" and r["name"] not in seen:
                seen.add(r["name"])
                spec = {"name": r["name"], "source": r["source"]}
                for k in ["hf_class", "hf_config", "input_type", "constructor_args", "inputs"]:
                    if k in r:
                        spec[k] = r[k]
                specs.append(spec)
        print(f"Loaded {len(specs)} clean models from {args.validate_from}")
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

    validate_modes = args.pass1_modes  # reuse pass1-modes for validation
    dynamic = args.dynamic or "true"  # default to dynamic=True for validation
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


def main():
    parser = argparse.ArgumentParser(
        description="Two-pass graph break sweep orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source / input
    parser.add_argument("--source", default="all",
                        choices=["timm", "hf", "diffusers", "hf+diffusers", "all"],
                        help="Model sources (hf+diffusers = HF + Diffusers without TIMM)")
    parser.add_argument("--models", help="JSON file with model specs (overrides --source)")
    parser.add_argument("--pass2-from", help="JSON file with pass 1 results (skip to pass 2)")
    parser.add_argument("--limit", type=int, help="Max models to test")
    parser.add_argument("--has-config-only", action="store_true",
                        help="Only test models with known input configs")

    # Execution
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--pass1-modes", nargs="+", default=["eval", "train"],
                        choices=["eval", "train"],
                        help="Modes for pass 1 (default: eval+train)")
    parser.add_argument("--pass2-modes", nargs="+", default=["eval", "train"],
                        choices=["eval", "train"],
                        help="Modes for pass 2 (default: eval+train on broken models)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=180,
                        help="Per-model timeout in seconds (pass 2 gets 2x)")
    parser.add_argument("--timeout-large", type=int, default=600,
                        help="Extended timeout for large models and auto-retry (default: 600s)")
    parser.add_argument("--large-models",
                        help="Path to large model registry JSON (default: sweep/large_models.json)")
    parser.add_argument("--no-auto-retry", action="store_true",
                        help="Skip auto-retry of timed-out models with extended timeout")
    parser.add_argument("--python", help="Python binary path (default: current interpreter)")

    # Output
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--skip-traces", action="store_true",
                        help="Skip TORCH_TRACE in pass 2")
    parser.add_argument("--pass1-only", action="store_true",
                        help="Only run pass 1 (identification)")
    parser.add_argument("--dynamic", nargs="?", const="true", default=None,
                        choices=["true", "mark"],
                        help="Dynamic shapes: 'true' = all dims, 'mark' = realistic dims only")
    parser.add_argument("--validate", action="store_true",
                        help="Run two-shape validation (pass 3) instead of pass 1/2")
    parser.add_argument("--validate-from",
                        help="JSON file with pass 1 results — validate only 'clean' models")

    # Resilience
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint (skip completed models)")
    parser.add_argument("--skip-models",
                        help="JSON file with list of model names to skip (toxic models)")

    args = parser.parse_args()

    if args.validate or args.validate_from:
        run_validation(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
