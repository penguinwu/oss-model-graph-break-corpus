"""Housekeeping: clean up leftover per-trial sandbox directories under /tmp/runs/.

The parallel runner (`discovery/run_config.py`) creates `<out>/sandbox/` per
trial and removes it on success. Failed trials (or `--no-cleanup`) leave the
sandbox behind for inspection. Without housekeeping these ~50 MB dirs
accumulate.

Usage:
    # Dry-run (default — shows what would be deleted)
    python -m discovery.clean_sandboxes

    # Actually delete sandboxes older than 3 days
    python -m discovery.clean_sandboxes --apply --older-than-days 3

    # Delete all sandboxes (CAREFUL — wipes inspection state)
    python -m discovery.clean_sandboxes --apply --older-than-days 0
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

DEFAULT_ROOTS = [Path("/tmp/runs"), Path("/tmp/discovery-runs")]


def _human_size(bytes_: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _find_sandboxes(root: Path) -> list[Path]:
    """Find all `sandbox/` directories under root."""
    if not root.exists():
        return []
    return list(root.rglob("sandbox"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", action="append", type=Path,
                   help="root dir to scan (default: /tmp/runs and /tmp/discovery-runs); can pass multiple")
    p.add_argument("--older-than-days", type=float, default=3.0,
                   help="only delete sandboxes whose mtime is older than N days (default 3)")
    p.add_argument("--apply", action="store_true",
                   help="actually delete (default: dry-run, just print)")
    args = p.parse_args()

    roots = args.root if args.root else DEFAULT_ROOTS
    cutoff = time.time() - (args.older_than_days * 86400)

    candidates: list[tuple[Path, float, int]] = []
    for root in roots:
        for sb in _find_sandboxes(root):
            try:
                mtime = sb.stat().st_mtime
            except OSError:
                continue
            if mtime > cutoff:
                continue  # too recent
            size = _dir_size(sb)
            age_days = (time.time() - mtime) / 86400
            candidates.append((sb, age_days, size))

    if not candidates:
        print(f"No sandbox dirs older than {args.older_than_days} days under {[str(r) for r in roots]}")
        return 0

    total_size = sum(s for _, _, s in candidates)
    print(f"Found {len(candidates)} sandbox dir(s) older than {args.older_than_days} days "
          f"({_human_size(total_size)} total):")
    for sb, age, size in sorted(candidates, key=lambda x: -x[2]):
        print(f"  {age:5.1f}d  {_human_size(size):>9}  {sb}")

    if not args.apply:
        print(f"\nDRY RUN — pass --apply to actually delete {len(candidates)} dir(s) "
              f"({_human_size(total_size)})")
        return 0

    n_deleted = 0
    for sb, _, _ in candidates:
        try:
            shutil.rmtree(sb)
            n_deleted += 1
        except OSError as e:
            print(f"  ✗ failed to delete {sb}: {e}", file=sys.stderr)
    print(f"\nDeleted {n_deleted} of {len(candidates)} sandbox dir(s) ({_human_size(total_size)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
