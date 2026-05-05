#!/usr/bin/env python3
"""bootstrap_modellibs.py — provision standalone modellibs trees per sweep/modellibs.json.

For each (pkg, ver) listed in sweep/modellibs.json, ensures
~/envs/modellibs/<pkg>-<ver>/ exists with the package installed via
`pip install --target`. Idempotent — skips if the tree is already present
and importable at the requested version.

USAGE
-----
    python3 tools/bootstrap_modellibs.py
    python3 tools/bootstrap_modellibs.py --dry-run
    python3 tools/bootstrap_modellibs.py --pkg transformers --ver 5.5.3
    python3 tools/bootstrap_modellibs.py --python ~/envs/torch211/bin/python3

WHY
---
Per design/venv-decouple-modellibs.md, model libs (transformers, diffusers,
timm) move out of per-torch-version venvs and into standalone trees so a
single torch venv can be tested against multiple model-lib versions without
duplication.

EXIT CODES
----------
  0  all trees ready (or successfully bootstrapped)
  1  bootstrap failed for at least one (pkg, ver)
  2  bad arguments / config
  3  pip install hit BPF jail (re-read recipe ~/.myclaw-shared/recipes/python-venv-bpf.md)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "sweep" / "modellibs.json"
MODELLIBS_ROOT = Path.home() / "envs" / "modellibs"


def load_config() -> dict:
    with open(CONFIG) as f:
        return json.load(f)


def tree_path(pkg: str, ver: str) -> Path:
    return MODELLIBS_ROOT / f"{pkg}-{ver}"


def tree_is_ready(pkg: str, ver: str, python_bin: str, verbose: bool = False) -> bool:
    """Returns True iff the tree exists AND `import <pkg>` from it returns ver."""
    path = tree_path(pkg, ver)
    if not path.exists():
        return False
    pkg_import_name = pkg.replace("-", "_")
    probe = subprocess.run(
        [python_bin, "-c",
         f"import sys; sys.path.insert(0, '{path}'); "
         f"import {pkg_import_name}; print({pkg_import_name}.__version__)"],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        if verbose:
            print(f"    probe failed: {probe.stderr.strip()}", file=sys.stderr)
        return False
    actual = probe.stdout.strip()
    if actual != ver:
        if verbose:
            print(f"    version mismatch: expected {ver}, got {actual}", file=sys.stderr)
        return False
    return True


def bootstrap_tree(pkg: str, ver: str, python_bin: str, dry_run: bool = False, no_deps: bool = False) -> int:
    """Pip-install pkg==ver into the tree dir. Returns shell exit code."""
    path = tree_path(pkg, ver)
    suffix = " (--no-deps)" if no_deps else ""
    print(f"  bootstrapping {pkg}=={ver} into {path}{suffix}", file=sys.stderr)
    if dry_run:
        print(f"    DRY RUN — skipping pip install", file=sys.stderr)
        return 0
    path.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin, "-m", "pip", "install",
        "--target", str(path),
        f"{pkg}=={ver}",
    ]
    if no_deps:
        cmd.append("--no-deps")
    print(f"    {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        # Detect BPF jail signatures
        combined = result.stdout + result.stderr
        if ("403 Forbidden" in combined or "has not been allowlisted" in combined
                or "agent_id=agent:claude_code" in combined):
            print(f"    BPF JAIL — read ~/.myclaw-shared/recipes/python-venv-bpf.md",
                  file=sys.stderr)
            print(combined[:2000], file=sys.stderr)
            return 3
        print(f"    pip install FAILED (exit {result.returncode}):", file=sys.stderr)
        print(combined[:2000], file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--python", default=sys.executable,
                    help="Python binary used to verify tree importability "
                         "and run pip (default: this script's interpreter)")
    ap.add_argument("--pkg", help="Bootstrap only this package (must match config)")
    ap.add_argument("--ver", help="Bootstrap only this version (requires --pkg)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan-only; print what would be installed but don't pip install")
    ap.add_argument("--force", action="store_true",
                    help="Re-install even if tree appears ready")
    ap.add_argument("--no-deps", action="store_true",
                    help="Pass --no-deps to pip. Required when a package transitively pulls "
                         "cuda-toolkit (BPF-blocked under agent identity). E.g. timm 1.0.26+.")
    args = ap.parse_args()

    if args.ver and not args.pkg:
        print("ERROR: --ver requires --pkg", file=sys.stderr)
        return 2

    if not CONFIG.exists():
        print(f"ERROR: config not found: {CONFIG}", file=sys.stderr)
        return 2

    config = load_config()
    pkgs = {k: v for k, v in config.items() if not k.startswith("_")}
    if args.pkg:
        if args.pkg not in pkgs:
            print(f"ERROR: --pkg {args.pkg!r} not in config", file=sys.stderr)
            return 2
        if args.ver:
            pkgs = {args.pkg: [args.ver]}
        else:
            pkgs = {args.pkg: pkgs[args.pkg]}

    MODELLIBS_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"bootstrap_modellibs.py — config: {CONFIG}", file=sys.stderr)
    print(f"  python: {args.python}", file=sys.stderr)
    print(f"  modellibs root: {MODELLIBS_ROOT}", file=sys.stderr)
    print(file=sys.stderr)

    failures = []
    skipped = 0
    bootstrapped = 0
    bpf = False
    for pkg, vers in pkgs.items():
        for ver in vers:
            print(f"[{pkg}=={ver}]", file=sys.stderr)
            if not args.force and tree_is_ready(pkg, ver, args.python, verbose=True):
                print(f"  ✓ already ready: {tree_path(pkg, ver)}", file=sys.stderr)
                skipped += 1
                continue
            rc = bootstrap_tree(pkg, ver, args.python, dry_run=args.dry_run, no_deps=args.no_deps)
            if rc == 3:
                bpf = True
                failures.append((pkg, ver, "BPF jail"))
                continue
            if rc != 0:
                failures.append((pkg, ver, f"pip exit {rc}"))
                continue
            # Re-verify
            if not args.dry_run and not tree_is_ready(pkg, ver, args.python, verbose=True):
                failures.append((pkg, ver, "post-install probe failed"))
                continue
            bootstrapped += 1
            print(f"  ✓ bootstrapped", file=sys.stderr)

    print(file=sys.stderr)
    print(f"Summary: {bootstrapped} bootstrapped, {skipped} already ready, "
          f"{len(failures)} failed.", file=sys.stderr)
    if failures:
        for pkg, ver, reason in failures:
            print(f"  ✗ {pkg}=={ver}: {reason}", file=sys.stderr)
    if bpf:
        return 3
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
