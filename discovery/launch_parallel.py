"""Parallel discovery launcher (Stage 1 — subprocess per config).

Walks the lifecycle gate ONCE, spawns N parallel `run_config.py` processes
(throttled by `--max-parallel`), waits for completion, then invokes
`merge_results.py` to build a summary.

Usage:
    python -m discovery.launch_parallel \\
        --case vits_model_train \\
        --variants V8 \\
        --skills none,/path/to/skills/debug-graph-breaks/SKILL.md \\
        --n 3 \\
        --experiment-dir /tmp/runs/v8-parallel-test \\
        --plan /path/to/experiment/plan.md

Configs auto-generated as cross-product of {variants × skills × trials},
e.g. above = 1 × 2 × 3 = 6 configs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _build_config_matrix(case: str, variants: list[str], skills: list[str], n: int) -> list[dict]:
    """Cartesian product of variants × skills × trials → flat config list."""
    configs = []
    for variant in variants:
        for skill in skills:
            arm = "noskill" if skill == "none" else Path(skill).stem
            for trial_idx in range(1, n + 1):
                config_id = f"{arm}_{variant}_{trial_idx}"
                configs.append({
                    "config_id": config_id,
                    "case": case,
                    "variant": variant,
                    "skill": skill,
                    "trial_label": config_id,
                })
    return configs


def _run_one_config(cfg: dict, experiment_dir: Path, timeout: int, python_bin: str) -> tuple[str, int, float]:
    """Spawn `run_config` for one cfg; wait for completion. Returns (config_id, exit_code, elapsed)."""
    cfg_dir = experiment_dir / cfg["config_id"]
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg_dir / "run_config.log"
    cmd = [
        python_bin, "-m", "discovery.run_config",
        "--case", cfg["case"],
        "--variant", cfg["variant"],
        "--skill", cfg["skill"],
        "--trial-label", cfg["trial_label"],
        "--out-dir", str(cfg_dir),
        "--timeout", str(timeout),
    ]
    t0 = time.time()
    with open(log_path, "w") as logf:
        res = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=False, cwd=str(REPO))
    return cfg["config_id"], res.returncode, time.time() - t0


def _wait_for_slot(running: dict, max_parallel: int) -> None:
    """Block until at least one slot frees up. `running` maps Popen→config_id."""
    while len(running) >= max_parallel:
        for proc in list(running.keys()):
            if proc.poll() is not None:
                cfg_id = running.pop(proc)
                print(f"  ✓ {cfg_id} completed (exit={proc.returncode})", file=sys.stderr, flush=True)
        if len(running) >= max_parallel:
            time.sleep(1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True, help="case id, e.g. vits_model_train")
    p.add_argument("--variants", required=True,
                   help="comma-separated variant ids, e.g. V0,V8")
    p.add_argument("--skills", default="none",
                   help="comma-separated skill arms ('none' = no skill, else path to skill md)")
    p.add_argument("--n", type=int, default=1, help="trials per (variant, skill) cell")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-trial agent timeout (default 1800 = 30 min)")
    p.add_argument("--experiment-dir", required=True, type=Path,
                   help="output dir; per-config dirs land underneath")
    p.add_argument("--max-parallel", type=int, default=6,
                   help="max concurrent configs (default 6)")
    p.add_argument("--python", default="/home/pengwu/envs/torch211/bin/python",
                   help="python interpreter for child processes")
    p.add_argument("--plan", default=None, type=Path,
                   help="experiment plan.md path (lifecycle gate target)")
    p.add_argument("--skip-merge", action="store_true",
                   help="skip merge_results step (for debugging)")
    from discovery._lifecycle_gate import add_lifecycle_args, check_or_die
    add_lifecycle_args(p)
    args = p.parse_args()

    # Lifecycle gate — fires ONCE for the whole batch (per design)
    plan_path = args.plan or (args.experiment_dir / "plan.md")
    check_or_die(plan_path, args, launcher="discovery/launch_parallel.py")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)

    # Build config matrix
    variants = args.variants.split(",")
    skills = args.skills.split(",")
    configs = _build_config_matrix(args.case, variants, skills, args.n)
    K = len(configs)
    print(f"=== launching {K} configs (max_parallel={args.max_parallel}) ===", file=sys.stderr)
    for cfg in configs:
        print(f"  - {cfg['config_id']}: case={cfg['case']} variant={cfg['variant']} skill={cfg['skill']}", file=sys.stderr)

    # Spawn pool
    t_batch_start = time.time()
    running: dict = {}  # Popen → config_id
    completed: list = []  # (config_id, returncode, elapsed)
    for cfg in configs:
        _wait_for_slot(running, args.max_parallel)
        cfg_dir = args.experiment_dir / cfg["config_id"]
        cfg_dir.mkdir(parents=True, exist_ok=True)
        log_path = cfg_dir / "run_config.log"
        cmd = [
            args.python, "-m", "discovery.run_config",
            "--case", cfg["case"],
            "--variant", cfg["variant"],
            "--skill", cfg["skill"],
            "--trial-label", cfg["trial_label"],
            "--out-dir", str(cfg_dir),
            "--timeout", str(args.timeout),
        ]
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO))
        proc._start_time = time.time()
        proc._logf = logf
        running[proc] = cfg["config_id"]
        print(f"  → spawned {cfg['config_id']} pid={proc.pid}", file=sys.stderr, flush=True)

    # Wait for all remaining
    while running:
        for proc in list(running.keys()):
            if proc.poll() is not None:
                cfg_id = running.pop(proc)
                proc._logf.close()
                completed.append((cfg_id, proc.returncode, time.time() - proc._start_time))
                print(f"  ✓ {cfg_id} completed (exit={proc.returncode})", file=sys.stderr, flush=True)
        if running:
            time.sleep(1)

    batch_elapsed = time.time() - t_batch_start
    n_failed = sum(1 for _, rc, _ in completed if rc != 0)
    print(f"=== batch complete: {K} configs, {n_failed} failed, {batch_elapsed:.0f}s wall ===", file=sys.stderr)

    # Write batch summary
    batch_summary = {
        "experiment_dir": str(args.experiment_dir),
        "case": args.case,
        "variants": variants,
        "skills": skills,
        "n_per_cell": args.n,
        "k_total": K,
        "max_parallel": args.max_parallel,
        "batch_elapsed_s": batch_elapsed,
        "configs": [
            {"config_id": cid, "exit_code": rc, "elapsed_s": el}
            for cid, rc, el in completed
        ],
    }
    (args.experiment_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2))

    # Merge results
    if not args.skip_merge:
        merge_cmd = [
            args.python, "-m", "discovery.merge_results",
            "--in-dir", str(args.experiment_dir),
            "--out", str(args.experiment_dir / "summary.md"),
        ]
        print(f"=== merging results ===", file=sys.stderr)
        subprocess.run(merge_cmd, cwd=str(REPO), check=False)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
