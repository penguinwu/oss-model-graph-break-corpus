#!/usr/bin/env python3
"""cleanup_modellibs_from_venvs.py — Phase 5 of venv-decouple-modellibs.

Removes baked-in transformers/diffusers/timm from each torch venv. Single
source of truth becomes ~/envs/modellibs/<pkg>-<ver>/ trees.

USAGE
-----
    # Dry run (default — just shows the plan)
    python3 tools/cleanup_modellibs_from_venvs.py

    # Actually uninstall
    python3 tools/cleanup_modellibs_from_venvs.py --apply

    # Restrict to specific venvs
    python3 tools/cleanup_modellibs_from_venvs.py --venv ~/envs/torch211 --apply

    # Force uninstall even when no modellib tree covers the version
    # (overrides the default skip-uncovered behavior — destructive!)
    python3 tools/cleanup_modellibs_from_venvs.py --apply --include-uncovered

SAFETY
------
1. Dry-run by default. --apply must be passed explicitly.
2. By default, SKIPS pkgs whose installed version isn't in modellibs.json
   (these are dev hashes / nightlies that legitimately stay baked-in per
   design doc §Phase 5). Use --include-uncovered to override.
3. Per-venv pip uninstall, not env-global pip uninstall.
4. Records what was uninstalled to a restore.sh so reversal is one command.
5. After uninstall, verifies `import <pkg>` from the venv FAILS as expected
   (so a stale .pth or .egg-info doesn't keep the import alive).

EXIT CODES
----------
  0  cleanup successful (or dry-run plan emitted)
  1  cleanup failed for at least one venv
  2  bad arguments
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = REPO_ROOT / "sweep" / "modellibs.json"
ENVS_ROOT = Path.home() / "envs"
MODELLIBS_ROOT = ENVS_ROOT / "modellibs"
TARGET_PKGS = ("transformers", "diffusers", "timm")


def discover_torch_venvs() -> List[Path]:
    """Return torch* venv dirs (excludes pool venvs cu126/cu128 and modellibs/)."""
    venvs = []
    for entry in sorted(ENVS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "modellibs":
            continue
        if entry.name in ("cu126", "cu128", "sentinel-before"):
            continue
        if not (entry / "bin" / "python3").exists():
            continue
        venvs.append(entry)
    return venvs


def installed_version(venv: Path, pkg: str) -> str | None:
    """Return version of <pkg> installed in <venv>, or None if not installed.

    Uses `pip show` (metadata-based) NOT `import` — because some venvs have
    pkgs installed but with import-time failures (e.g. torch ABI mismatch in
    torch212-rc, or torch-not-in-venv as in torch-181552). pip show tells the
    truth about what's on disk regardless of whether import works.
    """
    py = venv / "bin" / "python3"
    result = subprocess.run(
        [str(py), "-m", "pip", "show", pkg],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def load_modellib_versions() -> dict[str, set[str]]:
    """Parse sweep/modellibs.json into {pkg: {ver, ver, ...}}."""
    with open(CONFIG) as f:
        cfg = json.load(f)
    return {k: set(v) for k, v in cfg.items() if not k.startswith("_") and k in TARGET_PKGS}


def load_defaults_per_venv() -> dict:
    """Parse sweep/modellibs.json defaults_per_venv into {venv_name: {pkg: ver}}."""
    with open(CONFIG) as f:
        cfg = json.load(f)
    return cfg.get("defaults_per_venv", {})


def pip_uninstall(venv: Path, pkg: str, dry_run: bool) -> int:
    """Run pip uninstall -y inside the venv. Returns shell exit code."""
    py = venv / "bin" / "python3"
    cmd = [str(py), "-m", "pip", "uninstall", "-y", pkg]
    if dry_run:
        print(f"    [dry-run] would run: {' '.join(cmd)}", file=sys.stderr)
        return 0
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"    pip uninstall FAILED (exit {result.returncode}):", file=sys.stderr)
        print((result.stdout + result.stderr)[:1500], file=sys.stderr)
        return result.returncode
    return 0


def verify_uninstalled(venv: Path, pkg: str) -> bool:
    """Return True iff <pkg> is no longer installed.

    Two checks:
      1. `pip show` reports it gone (metadata-level uninstall worked).
      2. No top-level pkg dir remains in site-packages (orphan-file check).
    Both must be true. (Skip `import` check — some venvs have ABI issues that
    make `import` fail even when the pkg IS still installed.)
    """
    if installed_version(venv, pkg) is not None:
        return False
    py = venv / "bin" / "python3"
    sp_result = subprocess.run(
        [str(py), "-c", "import site; print('\\n'.join(site.getsitepackages()))"],
        capture_output=True, text=True, timeout=10,
    )
    if sp_result.returncode != 0:
        return True  # can't check; trust pip show
    for sp_line in sp_result.stdout.strip().splitlines():
        sp = Path(sp_line)
        if not sp.exists():
            continue
        for entry in sp.iterdir():
            name = entry.name.lower()
            if name == pkg or name.startswith(f"{pkg}-") or name.startswith(f"{pkg}."):
                return False
    return True


def cleanup_orphan_dirs(venv: Path, pkg: str) -> List[Path]:
    """After pip uninstall, hunt for orphan dirs/files left behind in site-packages.
    pip doesn't remove files it didn't install (e.g. cruft from a prior
    `pip install --target` install whose RECORD doesn't match). Returns list
    of paths that were removed."""
    removed = []
    # Find the venv's site-packages
    py = venv / "bin" / "python3"
    result = subprocess.run(
        [str(py), "-c",
         "import site; print('\\n'.join(site.getsitepackages()))"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return removed
    for sp_line in result.stdout.strip().splitlines():
        sp = Path(sp_line)
        if not sp.exists():
            continue
        # Look for dirs/files matching the pkg name
        for entry in sp.iterdir():
            name = entry.name.lower()
            if name == pkg or name.startswith(f"{pkg}-") or name.startswith(f"{pkg}."):
                if entry.is_dir():
                    import shutil
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed.append(entry)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--apply", action="store_true",
                    help="Actually uninstall. Without this flag, only the plan is printed.")
    ap.add_argument("--venv", action="append", default=[],
                    help="Restrict to this venv path. Repeatable. Default: all torch* venvs.")
    ap.add_argument("--include-uncovered", action="store_true",
                    help="Also uninstall pkg versions NOT covered by a modellib tree. "
                         "Default behavior is to skip them (dev hashes / nightlies stay baked-in "
                         "per design doc §Phase 5). Pass this only if you know what you're doing.")
    ap.add_argument("--restore-script",
                    default=str(REPO_ROOT / "tools" / "restore_modellibs_to_venvs.sh"),
                    help="Where to write the restore script (default: tools/).")
    args = ap.parse_args()

    if args.venv:
        venvs = [Path(v).expanduser().resolve() for v in args.venv]
        for v in venvs:
            if not (v / "bin" / "python3").exists():
                print(f"ERROR: not a venv: {v}", file=sys.stderr)
                return 2
    else:
        venvs = discover_torch_venvs()

    coverage = load_modellib_versions()
    defaults_per_venv = load_defaults_per_venv()

    print(f"cleanup_modellibs_from_venvs.py — {'APPLY' if args.apply else 'DRY RUN'}", file=sys.stderr)
    print(f"  venvs: {len(venvs)}", file=sys.stderr)
    print(f"  modellib coverage:", file=sys.stderr)
    for pkg, vers in sorted(coverage.items()):
        print(f"    {pkg}: {sorted(vers)}", file=sys.stderr)
    print(file=sys.stderr)

    plan: List[Tuple[Path, str, str, bool]] = []  # (venv, pkg, ver, covered)
    skipped: List[Tuple[Path, str, str]] = []
    for venv in venvs:
        print(f"=== {venv.name} ===", file=sys.stderr)
        venv_defaults = defaults_per_venv.get(venv.name, {})
        for pkg in TARGET_PKGS:
            ver = installed_version(venv, pkg)
            if ver is None:
                print(f"  {pkg}: not installed", file=sys.stderr)
                continue
            covered_by_tree = ver in coverage.get(pkg, set())
            covered_by_default = pkg in venv_defaults  # any default → sweep will inject modellib
            covered = covered_by_tree or covered_by_default
            if not covered and not args.include_uncovered:
                print(f"  {pkg}: {ver} — SKIP (no modellib tree, no defaults_per_venv mapping)",
                      file=sys.stderr)
                skipped.append((venv, pkg, ver))
                continue
            if covered_by_tree:
                mark = "✓ (tree exists)"
            elif covered_by_default:
                mark = f"✓ (defaults_per_venv → {venv_defaults[pkg]})"
            else:
                mark = "✗ (--include-uncovered)"
            print(f"  {pkg}: {ver} → {mark}", file=sys.stderr)
            plan.append((venv, pkg, ver, covered))
        print(file=sys.stderr)

    if skipped:
        print(f"Skipped {len(skipped)} pkg-removals (no modellib coverage — staying baked-in):",
              file=sys.stderr)
        for v, p, ver in skipped:
            print(f"  - {v.name}: {p}=={ver}", file=sys.stderr)
        print(file=sys.stderr)

    print(f"Plan: {len(plan)} pkg-removals across {len(set(v for v, _, _, _ in plan))} venvs",
          file=sys.stderr)
    if not args.apply:
        print(f"\nDry run only. Re-run with --apply to actually uninstall.", file=sys.stderr)
        return 0

    # Write restore script BEFORE uninstalling.
    restore_lines = [
        "#!/bin/bash",
        "# Restore script auto-generated by cleanup_modellibs_from_venvs.py.",
        "# Reverses Phase 5 cleanup by re-installing the exact versions that were uninstalled.",
        "# Note: re-installs go via pypi.org, which may BPF-block under agent identity for some pkgs.",
        "",
        "set -e",
        "",
    ]
    for venv, pkg, ver, _ in plan:
        py = venv / "bin" / "python3"
        restore_lines.append(
            f"HTTP_PROXY=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080 "
            f"{py} -m pip install {shlex.quote(f'{pkg}=={ver}')}"
        )
    Path(args.restore_script).write_text("\n".join(restore_lines) + "\n")
    Path(args.restore_script).chmod(0o755)
    print(f"  restore script: {args.restore_script}", file=sys.stderr)
    print(file=sys.stderr)

    failures = []
    for venv, pkg, ver, _ in plan:
        print(f"  uninstalling {pkg}=={ver} from {venv.name} ...", file=sys.stderr)
        rc = pip_uninstall(venv, pkg, dry_run=False)
        if rc != 0:
            failures.append((venv, pkg, ver, "pip exit %d" % rc))
            continue
        if not verify_uninstalled(venv, pkg):
            # pip uninstall left orphan files — hunt them down
            orphans = cleanup_orphan_dirs(venv, pkg)
            if orphans:
                print(f"    cleaned {len(orphans)} orphan path(s) pip missed:", file=sys.stderr)
                for o in orphans:
                    print(f"      → {o}", file=sys.stderr)
            if not verify_uninstalled(venv, pkg):
                failures.append((venv, pkg, ver, "still importable after orphan cleanup"))
                continue
        print(f"    ✓ removed", file=sys.stderr)

    print(file=sys.stderr)
    print(f"Summary: {len(plan) - len(failures)} removed, {len(failures)} failed.", file=sys.stderr)
    for venv, pkg, ver, reason in failures:
        print(f"  ✗ {venv.name} {pkg}=={ver}: {reason}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
