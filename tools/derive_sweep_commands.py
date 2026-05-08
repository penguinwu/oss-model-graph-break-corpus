#!/usr/bin/env python3
"""Derive launch commands (gate / sample / full) from one experiment config.

Codifies the gate → sample → full launch sequence as deterministic
transformations of a single source config. Eliminates drift between stages by
construction: every stage uses the SAME flags as the full launch except for
cohort sub-sample size + output dir.

Anti-pattern this exists to prevent (per local CLAUDE.md "Approval-Triggered
Action Discipline" + 2026-05-07 generic-gate-false-confidence reflection):
hand-typing a "customized gate" config that diverges from the planned full
launch (different stack, workers, modes, or missing flags). The derived
commands are mechanically guaranteed to match.

Usage:
    # Validate first (also catches sha256 drift on inner cohort source)
    tools/derive_sweep_commands.py <config> --stage <gate|sample|full> --validate

    # Print bash to stdout (don't run)
    tools/derive_sweep_commands.py <config> --stage <gate|sample|full> --emit

    # Run directly (gate/sample record success state; full refuses
    # without prior gate+sample passing — override with --allow-skip-gate)
    tools/derive_sweep_commands.py <config> --stage <gate|sample|full> --run

Stage semantics:
    gate   — sub-sample 5 models from original cohort, deterministic seed.
             ~5 min compute. Catches launch-path bugs before larger commitment.
    sample — sub-sample 20 models. ~15-30 min. Catches cohort/stack issues
             at moderate cost before full sweep.
    full   — original cohort. Hours of compute. Only after gate + sample pass.

Required source-config fields (REQUIRED by derive even though optional in
run_experiment.py schema):
    settings.python_bin       — absolute path to the python interpreter
                                (e.g., ~/envs/torch-nightly-cu126/bin/python).
                                Without this, emitted bash uses bare `python3`
                                from PATH — drift risk (the failure mode this
                                exists to prevent).
    settings.modellib_pins    — dict like {"transformers": "5.6.2", ...}.
                                Without this, emitted bash doesn't pin
                                transformers/diffusers — same drift risk.

Validation behaviors:
    - Refuses if source config's models block is itself `source: "sample"`
      (would produce a double-wrap that exposes a latent resolve_models bug).
    - --validate recursively walks the models tree and runs sha256 drift
      checks at every level (the inner `from` block of the wrapped sample
      contains the source_sha256 pin; without recursion, validate would PASS
      on stale sha256 and crash at run time).
    - Refuses if source config is missing python_bin or modellib_pins.
    - Refuses if `_derived_from` / `_derived_stage` annotations would collide
      with existing top-level keys.

Skip-to-full guardrail:
    - On successful --run of gate, writes /tmp/derive_sweep_state/<sha8>.json
      recording (config_sha256, gate_passed_at). Same for sample.
    - On --run --stage full, refuses unless gate AND sample have passed for
      the CURRENT source-config sha256 (regenerated state if config changed).
    - Override: --allow-skip-gate (logged loud).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_EXPERIMENT = REPO_ROOT / "tools" / "run_experiment.py"
STATE_DIR = Path(tempfile.gettempdir()) / "derive_sweep_state"

# Stage configuration — the ONLY thing that varies across stages
STAGE_CONFIG = {
    "gate":   {"size": 5,  "name_suffix": "-gate"},
    "sample": {"size": 20, "name_suffix": "-sample"},
    "full":   {"size": None, "name_suffix": ""},  # None = no sub-sampling
}
DEFAULT_SUB_SAMPLE_SEED = 42
RESERVED_ANNOTATIONS = ("_derived_from", "_derived_stage", "_derived_source_sha256")


class DeriveError(Exception):
    """Raised when derive cannot proceed (refusal, not just an underlying tool failure)."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _config_sha8(orig_config_path: Path) -> str:
    """Short sha256 of the source config — used for filename uniqueness + state keying."""
    return _sha256(orig_config_path.read_bytes())[:16]


def _check_source_config_required_fields(orig_config: dict) -> None:
    """Refuse early if the source config is missing fields derive REQUIRES.

    Closes adversary gaps #4, #5: derive cannot emit a complete launch command
    without python_bin (interpreter pin) and modellib_pins (stack pin). Bare
    `python3` + missing modellib flags is the EXACT drift mode this exists
    to prevent.
    """
    settings = orig_config.get("settings", {})
    if not settings.get("python_bin"):
        raise DeriveError(
            "source config is missing settings.python_bin — required by derive.\n"
            "  Add e.g. \"python_bin\": \"~/envs/torch-nightly-cu126/bin/python\" to settings.\n"
            "  Without this the emitted bash uses bare `python3` from PATH (stack-drift risk)."
        )
    pins = settings.get("modellib_pins")
    if not pins:
        raise DeriveError(
            "source config is missing settings.modellib_pins — required by derive.\n"
            "  Add e.g. \"modellib_pins\": {\"transformers\": \"5.6.2\", \"diffusers\": \"0.38.0\"} to settings.\n"
            "  Without this the emitted bash doesn't pin transformers/diffusers (stack-drift risk)."
        )


def _check_source_models_not_already_sampled(orig_config: dict) -> None:
    """Refuse if source config's models block is already a sample (gap #2).

    Double-wrapping (sample of a sample) exposes a latent bug in
    run_experiment.py:resolve_models around lines 416-417 where the `if "status"`
    check overwrites `base_config` and silently drops the inner `from`.
    Cheapest fix: refuse the input.
    """
    src = orig_config.get("models", {}).get("source")
    if src == "sample":
        raise DeriveError(
            "source config's models.source is already 'sample' — derive does not\n"
            "support double-wrapping. Un-wrap the source config to a non-sample\n"
            "models block (e.g., corpus_filter or list) before deriving stages."
        )


def _check_annotation_collisions(orig_config: dict) -> None:
    """Refuse if reserved annotation keys would collide with existing fields (gap #6)."""
    collisions = [k for k in RESERVED_ANNOTATIONS if k in orig_config]
    if collisions:
        raise DeriveError(
            f"source config has reserved annotation key(s) at top level: {collisions}.\n"
            f"  derive_sweep_commands uses these to track origin; rename your fields\n"
            f"  or remove them from the source."
        )


def _walk_sha256_pins(models_block: dict, errors: list) -> None:
    """Recursively walk a models block, checking source_sha256 pins at every level.

    Closes gap #1: validate_config only runs the sha256 check inside
    `corpus_filter`. After derive wraps in sample, the OUTER source becomes
    `sample` and the inner `from`'s source_sha256 is never checked at validate
    time. This walker explicitly recurses into `from` blocks to catch drift.
    """
    if not isinstance(models_block, dict):
        return
    src_path = models_block.get("from")
    pinned = models_block.get("source_sha256")
    # Two cases: 'from' is a path string (corpus_filter) OR a nested models block (sample)
    if isinstance(src_path, str) and pinned:
        full_path = REPO_ROOT / src_path
        if not full_path.exists():
            errors.append(f"models.from path not found: {full_path}")
            return
        actual = _sha256(full_path.read_bytes())
        if actual != pinned:
            errors.append(
                f"source_sha256 drift on {full_path}\n"
                f"      pinned: {pinned}\n"
                f"      actual: {actual}\n"
                f"      The source file has been regenerated since this config was written.\n"
                f"      Update source_sha256 to '{actual}' if regeneration is intentional."
            )
    if isinstance(src_path, dict):
        # Recurse into nested models block (e.g., sample wrapping corpus_filter)
        _walk_sha256_pins(src_path, errors)


def derive_stage_config(orig_config: dict, stage: str, seed: int = DEFAULT_SUB_SAMPLE_SEED,
                        orig_config_sha: str = "") -> dict:
    """Transform an experiment config for a stage (gate / sample / full).

    Only `models` and `name` change. Every other field (description, configs[],
    settings, etc.) is preserved verbatim.
    """
    if stage not in STAGE_CONFIG:
        raise ValueError(f"unknown stage {stage!r}; expected one of {list(STAGE_CONFIG)}")

    _check_source_config_required_fields(orig_config)
    _check_source_models_not_already_sampled(orig_config)
    _check_annotation_collisions(orig_config)

    sc = STAGE_CONFIG[stage]
    new_config = json.loads(json.dumps(orig_config))  # deep copy
    new_config["name"] = orig_config["name"] + sc["name_suffix"]

    if sc["size"] is not None:
        # Sub-sample: wrap original models block in a sample resolver
        new_config["models"] = {
            "_comment": (f"Stage={stage} sub-sample (size={sc['size']}, seed={seed}). "
                         f"Original cohort preserved in 'from'. Generated by "
                         f"tools/derive_sweep_commands.py — DO NOT hand-edit; "
                         f"regenerate from {orig_config['name']} instead."),
            "source": "sample",
            "size": sc["size"],
            "seed": seed,
            "strategy": "random",
            "from": orig_config["models"],
        }
    # else: full stage — models block unchanged

    # Annotate that this is a derived config (not the source of truth)
    new_config["_derived_from"] = orig_config["name"]
    new_config["_derived_stage"] = stage
    if orig_config_sha:
        new_config["_derived_source_sha256"] = orig_config_sha

    return new_config


def emit_bash(transformed_config_path: Path, original_config: dict, stage: str) -> str:
    """Build the bash command that invokes run_experiment.py run on the
    transformed config. Pinned-interpreter + pinned-modellibs from the source
    config (closes gaps #4, #5). validate runs before run.
    """
    settings = original_config["settings"]
    python_bin = settings["python_bin"]  # presence already enforced
    pins = settings["modellib_pins"]      # presence already enforced

    pin_flags = []
    for pkg in ("transformers", "diffusers", "timm"):
        if pkg in pins:
            pin_flags.append(f"--{pkg}")
            pin_flags.append(shlex.quote(pins[pkg]))
    pin_flags_str = " ".join(pin_flags) if pin_flags else ""

    # Note: --transformers / --diffusers are flags on `sweep` subcommand, not `run`.
    # `run` injects via PYTHONPATH. We document both: the canonical run path
    # is via run_experiment.py run, with PYTHONPATH front-loading the modellib trees.
    # Resolve $HOME at python time — shlex.quote single-quotes the result and
    # bash won't expand $HOME inside single quotes.
    home = str(Path.home())
    pythonpath_paths = [
        f"{home}/envs/modellibs/{pkg}-{ver}"
        for pkg, ver in pins.items()
    ]
    pythonpath_str = ":".join(pythonpath_paths)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Stage: {stage}",
        f"# Derived from: {original_config['name']}",
        f"# Transformed config: {transformed_config_path}",
        f"# Pinned interpreter: {python_bin}",
        f"# Pinned modellibs:   {pins}",
        "",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "",
        f"# Validate the pinned interpreter exists at runtime",
        f'[ -x {shlex.quote(python_bin)} ] || {{ echo "ERROR: pinned python_bin not executable: {python_bin}"; exit 2; }}',
        "",
        f"# Validate the transformed config before running (catches sha256 drift, schema errors)",
        f"{shlex.quote(python_bin)} {shlex.quote(str(RUN_EXPERIMENT))} validate {shlex.quote(str(transformed_config_path))}",
        "",
        f"# Front-load modellibs PYTHONPATH (run subcommand doesn't auto-inject like sweep does)",
        f"export PYTHONPATH={shlex.quote(pythonpath_str)}${{PYTHONPATH:+:$PYTHONPATH}}",
        "",
        f"# Run the stage",
        f"{shlex.quote(python_bin)} {shlex.quote(str(RUN_EXPERIMENT))} run {shlex.quote(str(transformed_config_path))}",
    ]
    return "\n".join(lines) + "\n"


def write_transformed_config(transformed: dict, source_sha8: str) -> Path:
    """Write the transformed config to a /tmp path keyed by source-config sha
    (closes gap #7: concurrent derives don't collide; old-config-driven derives
    don't overwrite new-config-driven derives)."""
    name = transformed["name"]
    out = Path(tempfile.gettempdir()) / f"{name}-{source_sha8}.json"
    out.write_text(json.dumps(transformed, indent=2) + "\n")
    return out


# ─── Skip-to-full guardrail (gap #3) ─────────────────────────────────────

def _state_path(orig_config: dict, source_sha8: str) -> Path:
    STATE_DIR.mkdir(exist_ok=True)
    return STATE_DIR / f"{orig_config['name']}-{source_sha8}.json"


def _load_state(orig_config: dict, source_sha8: str) -> dict:
    p = _state_path(orig_config, source_sha8)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _record_stage_passed(orig_config: dict, source_sha8: str, stage: str) -> None:
    state = _load_state(orig_config, source_sha8)
    state[f"{stage}_passed_at"] = time.time()
    state["source_sha256"] = source_sha8  # belt-and-suspenders
    _state_path(orig_config, source_sha8).write_text(json.dumps(state, indent=2))


def _check_skip_to_full_guardrail(orig_config: dict, source_sha8: str,
                                   stage: str, allow_skip_gate: bool) -> None:
    """Refuse --stage full --run if gate or sample hasn't passed for current sha."""
    if stage != "full":
        return
    if allow_skip_gate:
        print("[derive_sweep_commands] WARNING: --allow-skip-gate set; "
              "skipping gate→sample→full sequence enforcement. "
              "Override is logged.", file=sys.stderr)
        return
    state = _load_state(orig_config, source_sha8)
    missing = []
    if "gate_passed_at" not in state:
        missing.append("gate")
    if "sample_passed_at" not in state:
        missing.append("sample")
    if missing:
        raise DeriveError(
            f"--stage full --run refused: prior stages have not passed "
            f"for current source-config sha {source_sha8}.\n"
            f"  missing: {missing}\n"
            f"  expected state file: {_state_path(orig_config, source_sha8)}\n"
            f"  Run gate + sample first (with --run, which records success state),\n"
            f"  OR pass --allow-skip-gate to override (logged)."
        )


# ─── CLI ─────────────────────────────────────────────────────────────────

def validate_transformed_config(config_path: Path, transformed: dict) -> tuple[int, str]:
    """Run run_experiment.py validate on the transformed config + walk
    sha256 pins recursively (gap #1).
    """
    # First: standard validate via run_experiment.py
    r = subprocess.run(
        [sys.executable, str(RUN_EXPERIMENT), "validate", str(config_path)],
        capture_output=True, text=True,
    )
    output = r.stdout + r.stderr
    if r.returncode != 0:
        return r.returncode, output

    # Second: recursive sha256 walk on the transformed models block (catches
    # drift on inner `from` blocks that the standard validate doesn't recurse into)
    sha_errors = []
    _walk_sha256_pins(transformed.get("models", {}), sha_errors)
    if sha_errors:
        return 1, output + "\nDeep sha256 validation FAILED:\n  - " + "\n  - ".join(sha_errors)

    return 0, output


def main() -> int:
    if sys.version_info < (3, 9):
        sys.exit("ERROR: derive_sweep_commands.py requires Python >= 3.9")

    p = argparse.ArgumentParser(
        description="Derive gate/sample/full launch commands from an experiment config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", type=Path, help="Path to experiment config JSON")
    p.add_argument("--stage", required=True, choices=list(STAGE_CONFIG),
                   help="Which stage to derive: gate (5 models), sample (20 models), or full")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true",
                      help="Derive + deep-validate the transformed config; do not emit or run")
    mode.add_argument("--emit", action="store_true",
                      help="Derive + write transformed config + print bash to stdout")
    mode.add_argument("--run", action="store_true",
                      help="Derive + write transformed config + execute the launch")
    p.add_argument("--seed", type=int, default=DEFAULT_SUB_SAMPLE_SEED,
                   help="Sub-sample seed (default: 42; deterministic per stage)")
    p.add_argument("--allow-skip-gate", action="store_true",
                   help="Allow --stage full --run without prior gate+sample passing. Logged loud.")
    args = p.parse_args()

    if not args.config.is_file():
        sys.exit(f"ERROR: config not found: {args.config}")

    orig = json.loads(args.config.read_text())
    if "name" not in orig or "models" not in orig:
        sys.exit(f"ERROR: config missing required fields (name, models)")

    source_sha8 = _config_sha8(args.config)

    try:
        transformed = derive_stage_config(orig, args.stage, seed=args.seed,
                                          orig_config_sha=source_sha8)
    except DeriveError as e:
        print(f"[derive_sweep_commands] REFUSED: {e}", file=sys.stderr)
        return 1

    transformed_path = write_transformed_config(transformed, source_sha8)

    print(f"[derive_sweep_commands] stage={args.stage} → {transformed_path}", file=sys.stderr)
    print(f"[derive_sweep_commands] derived name: {transformed['name']}", file=sys.stderr)
    print(f"[derive_sweep_commands] source-config sha8: {source_sha8}", file=sys.stderr)

    rc, out = validate_transformed_config(transformed_path, transformed)
    if rc != 0:
        print(f"[derive_sweep_commands] VALIDATION FAILED on {transformed_path}", file=sys.stderr)
        print(out, file=sys.stderr)
        return 1
    print(f"[derive_sweep_commands] validated transformed config (incl. recursive sha256)", file=sys.stderr)

    if args.validate:
        return 0

    bash = emit_bash(transformed_path, orig, args.stage)
    if args.emit:
        print(bash)
        return 0

    if args.run:
        # Skip-to-full guardrail (gap #3)
        try:
            _check_skip_to_full_guardrail(orig, source_sha8, args.stage, args.allow_skip_gate)
        except DeriveError as e:
            print(f"[derive_sweep_commands] REFUSED: {e}", file=sys.stderr)
            return 1
        print(f"[derive_sweep_commands] running stage={args.stage}...", file=sys.stderr)
        bash_path = Path(tempfile.gettempdir()) / f"{transformed['name']}-{source_sha8}.sh"
        bash_path.write_text(bash)
        bash_path.chmod(0o755)
        r = subprocess.run([str(bash_path)])
        if r.returncode == 0 and args.stage in ("gate", "sample"):
            _record_stage_passed(orig, source_sha8, args.stage)
            print(f"[derive_sweep_commands] recorded {args.stage} success "
                  f"in {_state_path(orig, source_sha8)}", file=sys.stderr)
        return r.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
