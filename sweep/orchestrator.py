#!/usr/bin/env python3
"""Shared orchestrator — parallel worker dispatch, checkpointing, GPU health.

Extracted from run_sweep.py to enable reuse by both the sweep pipeline
and the experiment runner. This module contains zero sweep-specific logic;
it manages the lifecycle of worker subprocesses.

Used by:
  - sweep/run_sweep.py (production sweeps)
  - tools/run_experiment.py (ad-hoc experiments)
"""
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
                 dynamic=False, extra_args=None):
    """Spawn a worker subprocess in its own process group.

    Uses temp files for stdout/stderr to avoid pipe buffer deadlocks.
    Returns a WorkerHandle for non-blocking tracking.

    Args:
        extra_args: Optional list of additional CLI args to pass to the worker
                    (e.g., ['--dynamo-flags', '{"capture_scalar_outputs": true}'])
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Optional: HF kernels-community kernel resolution for models that need them.
    # GATED behind SWEEP_USE_KERNEL_RESOLVER=1 to delay the per-model rollout
    # decision (see issue tracking MRA harness work). When enabled, sets
    # LOCAL_KERNELS=... in the worker subprocess env using sweep/kernel_resolver.py.
    if os.environ.get("SWEEP_USE_KERNEL_RESOLVER") == "1":
        try:
            from sweep.kernel_resolver import resolve_kernels_for_model
            torch_ver = subprocess.check_output(
                [python_bin, "-c", "import torch; print(torch.__version__)"],
                stderr=subprocess.DEVNULL, timeout=10,
            ).decode().strip()
            local_kernels_value = resolve_kernels_for_model(spec.get("name", ""), torch_ver)
            if local_kernels_value:
                env["LOCAL_KERNELS"] = local_kernels_value
        except Exception:
            # Defensive: never let kernel-resolver failures break the sweep
            pass

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
    if extra_args:
        cmd.extend(extra_args)

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


_KNOWN_PHASES = frozenset({
    "create", "eager", "compile", "done",
    "eager_retry_image_tokens",
    "eager_a", "shape_b", "compiled_b", "eager_b", "compare",
})


def timeout_result(handle):
    """Build a timeout result, extracting phase from stderr if possible."""
    phase = "unknown"
    try:
        with open(handle.stderr_path) as f:
            for line in reversed(f.read().splitlines()):
                if line.startswith("PHASE:"):
                    candidate = line.split(":", 1)[1].strip()
                    if candidate in _KNOWN_PHASES or candidate.startswith("chaos_"):
                        phase = candidate
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
             timeout_overrides=None, skip_models=None, extra_worker_args=None,
             result_callback=None):
    """Run a full pass using a non-blocking poll loop with process group isolation.

    The orchestrator never blocks on any single worker. Timed-out workers are
    killed with escalating signals (TERM → KILL → abandon). GPU pressure
    reduces parallelism instead of aborting the sweep.

    Args:
        checkpoint_file: Path to JSONL file for incremental result saving
        resume_from: Dict of (name, mode) -> result to skip
        timeout_overrides: Dict of model_name -> timeout_s for large models
        skip_models: Set of model names to skip entirely
        extra_worker_args: Optional list of additional CLI args for workers
        result_callback: Optional callable(result_dict) invoked as each worker
            finishes. Use for streaming results to disk.
    """
    # Escalation timings (seconds after timeout)
    TERM_GRACE = 5     # wait this long after timeout before SIGTERM
    KILL_GRACE = 10    # wait this long after SIGTERM before SIGKILL
    ABANDON_GRACE = 20 # wait this long after SIGKILL before abandoning

    # GPU recovery settings
    GPU_RECOVERY_WAIT = 30    # seconds between GPU health checks
    GPU_RECOVERY_RETRIES = 4  # max retries (total wait = 4 × 30s = 2 min)

    # Stagger between back-to-back worker spawns inside the same scheduling pass.
    # Without this, two subprocess.Popen calls fire near-simultaneously, both
    # children import torch + initialize cudnn 9.x at the same wall-clock moment,
    # and one or both can hit "Invalid handle. Cannot load symbol cudnnGetVersion"
    # from racing on the libcudnn dlopen. PT 2.12 baseline 2026-04-30 had 16/24
    # auto-retried errors flake-pass on serial retry; cudnn-signature lines were
    # ~15 of those. Tunable via SWEEP_SPAWN_STAGGER_S env var (set to 0 to
    # disable). Only applies between consecutive spawns in the same iteration —
    # steady-state spawns (one slot at a time) pay zero overhead.
    SPAWN_STAGGER_S = float(os.environ.get("SWEEP_SPAWN_STAGGER_S", "3.0"))

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
    pass_start_time = time.time()

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
                    if result_callback:
                        result_callback(result)
                    if ckpt_fh:
                        ckpt_fh.write(json.dumps(result) + "\n")
                        ckpt_fh.flush()
                    _print_progress(completed, total, result, pass_start_time)
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
                        if result_callback:
                            result_callback(result)
                        if ckpt_fh:
                            ckpt_fh.write(json.dumps(result) + "\n")
                            ckpt_fh.flush()
                        _print_progress(completed, total, result, pass_start_time)
                        del active[pid]
                    elif elapsed > handle.timeout_s + TERM_GRACE + KILL_GRACE + ABANDON_GRACE:
                        # Truly stuck — abandon and move on
                        handle.kill_stage = 3
                        result = timeout_result(handle)
                        results.append(result)
                        completed += 1
                        if result_callback:
                            result_callback(result)
                        if ckpt_fh:
                            ckpt_fh.write(json.dumps(result) + "\n")
                            ckpt_fh.flush()
                        _print_progress(completed, total, result, pass_start_time)
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
                        model_timeout, dynamic, extra_args=extra_worker_args
                    )
                    active[handle.proc.pid] = handle
                    # Stagger before the next consecutive spawn to give the
                    # just-spawned worker a head start on cudnn lazy-load
                    # initialization (see SPAWN_STAGGER_S comment for context).
                    if SPAWN_STAGGER_S > 0 and pending and len(active) < current_max:
                        time.sleep(SPAWN_STAGGER_S)
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
                    if result_callback:
                        result_callback(result)
                    if ckpt_fh:
                        ckpt_fh.write(json.dumps(result) + "\n")
                        ckpt_fh.flush()
                    _print_progress(completed, total, result, pass_start_time)

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


def _format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


def _print_progress(completed, total, result, start_time=None):
    """Print progress line for each completed model."""
    name = result.get("name", "?")
    source = result.get("source", "?")
    mode = result.get("mode", "?")
    status = result.get("status", "?")
    wall = result.get("wall_time_s", 0)

    status_str = {
        "full_graph": "FULL_GRAPH",
        "graph_break": "BREAK",
        "success": "SUCCESS",
        "error": "ERROR",
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

    pct = 100 * completed / total if total else 0
    timing = ""
    if start_time:
        elapsed = time.time() - start_time
        if completed > 0:
            remaining = (elapsed / completed) * (total - completed)
            timing = f"  [{_format_duration(elapsed)} elapsed, ~{_format_duration(remaining)} remaining]"

    print(f"  [{completed:>4}/{total:>4} {pct:>3.0f}%] {source}/{name:<30} {mode:<6} "
          f"{status_str:<12} {wall:>5.1f}s{extra}{timing}", flush=True)


def log_versions(python_bin):
    """Detect PyTorch, transformers, and diffusers versions.

    Returns a dict with version info, or None on failure.
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
