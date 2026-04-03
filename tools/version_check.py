#!/usr/bin/env python3
"""Check installed package versions against corpus metadata.

Compares installed PyTorch, transformers, and diffusers versions against
what the corpus was generated with. Returns CURRENT, NEWER, or OLDER
status for each package.

Usage:
    python tools/version_check.py          # Check all packages
    python tools/version_check.py --json   # Machine-readable output
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus" / "corpus.json"

# Map of corpus metadata key → Python import name
PACKAGES = {
    "pytorch_version": "torch",
    "transformers_version": "transformers",
    "diffusers_version": "diffusers",
}


def parse_version(v):
    """Parse version string into comparable tuple, ignoring build metadata."""
    # Strip build metadata like +cu128
    base = v.split("+")[0] if v else ""
    # Strip pre-release suffixes like .dev20260401
    parts = base.split(".")
    nums = []
    for p in parts:
        # Stop at non-numeric parts (dev, rc, post, etc.)
        if p.isdigit():
            nums.append(int(p))
        else:
            break
    return tuple(nums) if nums else (0,)


def get_installed_version(import_name):
    """Get the installed version of a package, or None if not installed."""
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", None)
    except ImportError:
        return None


def compare_versions(installed, corpus):
    """Compare two version strings. Returns 'CURRENT', 'NEWER', or 'OLDER'."""
    iv = parse_version(installed)
    cv = parse_version(corpus)

    if iv == cv:
        return "CURRENT"
    elif iv > cv:
        return "NEWER"
    else:
        return "OLDER"


def main():
    parser = argparse.ArgumentParser(
        description="Check installed versions against corpus metadata",
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--corpus", default=None,
                        help="Path to corpus.json (default: corpus/corpus.json)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve() if args.corpus else CORPUS_PATH
    if not corpus_path.exists():
        print(f"ERROR: Corpus not found at {corpus_path}", file=sys.stderr)
        sys.exit(1)

    with open(corpus_path) as f:
        metadata = json.load(f).get("metadata", {})

    results = []
    any_mismatch = False

    for meta_key, import_name in PACKAGES.items():
        corpus_ver = metadata.get(meta_key, "")
        installed_ver = get_installed_version(import_name)

        if installed_ver is None:
            status = "NOT_INSTALLED"
            any_mismatch = True
        else:
            status = compare_versions(installed_ver, corpus_ver)
            if status != "CURRENT":
                any_mismatch = True

        results.append({
            "package": import_name,
            "corpus_version": corpus_ver,
            "installed_version": installed_ver,
            "status": status,
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Corpus generated with:")
        print()
        for r in results:
            pkg = r["package"]
            cv = r["corpus_version"]
            iv = r["installed_version"] or "(not installed)"
            status = r["status"]

            # Status indicator
            if status == "CURRENT":
                indicator = "="
            elif status == "NEWER":
                indicator = "^"
            elif status == "OLDER":
                indicator = "!"
            else:
                indicator = "X"

            print(f"  [{indicator}] {pkg:15s}  corpus={cv:12s}  installed={iv:12s}  {status}")

        print()
        if not any_mismatch:
            print("All packages match corpus versions.")
        else:
            has_older = any(r["status"] == "OLDER" for r in results)
            has_missing = any(r["status"] == "NOT_INSTALLED" for r in results)
            if has_older:
                print("WARNING: Some packages are OLDER than corpus versions.")
                print("Results may not match corpus data. Consider updating:")
                for r in results:
                    if r["status"] == "OLDER":
                        print(f"  pip install {r['package']}=={r['corpus_version']}")
            if has_missing:
                print("WARNING: Some packages are not installed.")
                print("Install them with:")
                for r in results:
                    if r["status"] == "NOT_INSTALLED":
                        print(f"  pip install {r['package']}=={r['corpus_version']}")

    sys.exit(0 if not any_mismatch else 1)


if __name__ == "__main__":
    main()
