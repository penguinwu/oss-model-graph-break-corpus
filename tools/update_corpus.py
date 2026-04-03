#!/usr/bin/env python3
"""Merge sweep results into corpus.json with changelog generation.

Usage:
    # Update corpus from a sweep results directory
    python tools/update_corpus.py sweep_results/v2.10/

    # Dry run — show what would change without writing
    python tools/update_corpus.py sweep_results/v2.10/ --dry-run

    # Skip validation after update
    python tools/update_corpus.py sweep_results/v2.10/ --no-validate

The sweep directory should contain:
    - identify_results.json (required)
    - explain_results.json (optional — merges break_reasons, graph counts)
    - versions.json (optional — updates corpus metadata)

Generates a changelog at {sweep_dir}/changelog.md showing all status
transitions, new/removed models, and break count changes.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path


CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "corpus", "corpus.json")

# Fields copied from identify results into corpus model.{mode}
IDENTIFY_FIELDS = [
    "status", "fullgraph_ok", "create_time_s", "eager_time_s",
    "compile_time_s", "gpu_mem_mb", "wall_time_s",
]

# Fields copied from explain results into corpus model.{mode}
EXPLAIN_FIELDS = [
    "graph_break_count", "graph_count", "break_reasons",
]


def load_sweep_results(path):
    """Load results from a JSON file (supports {results: [...]} or [...])."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("results", [])
    return data


def index_by_name_mode(results):
    """Index results by (name, mode) tuple."""
    idx = {}
    for r in results:
        key = (r["name"], r.get("mode", "eval"))
        idx[key] = r
    return idx


def compute_summary(models):
    """Compute the summary block from model data."""
    eval_counts = Counter()
    train_counts = Counter()
    has_graph_break_count = 0

    for m in models:
        e_status = m.get("eval", {}).get("status")
        t_status = m.get("train", {}).get("status")
        if e_status:
            eval_counts[e_status] += 1
        if t_status:
            train_counts[t_status] += 1
        if m.get("has_graph_break"):
            has_graph_break_count += 1

    return {
        "total_models": len(models),
        "eval": dict(eval_counts),
        "train": dict(train_counts),
        "has_graph_break": has_graph_break_count,
        "eval_status": dict(eval_counts),
        "train_status": dict(train_counts),
    }


def compute_has_graph_break(model):
    """Determine if a model has graph breaks in any mode or dynamic variant."""
    for mode_key in ("eval", "train"):
        md = model.get(mode_key, {})
        if md.get("status") == "graph_break":
            return True
        for dyn in ("dynamic_mark", "dynamic_true"):
            if md.get(dyn, {}).get("status") == "graph_break":
                return True
    return False


def merge_identify_into_mode(mode_data, identify_result):
    """Merge identify result fields into a corpus mode entry."""
    for field in IDENTIFY_FIELDS:
        if field in identify_result:
            mode_data[field] = identify_result[field]
        elif field in mode_data and field != "status":
            # Remove fields no longer present (e.g. compile_time_s for eager_error)
            del mode_data[field]

    # Handle error fields
    status = identify_result.get("status", "")
    if status == "graph_break" and "fullgraph_error" in identify_result:
        mode_data["error"] = identify_result["fullgraph_error"]
        mode_data.pop("fullgraph_error", None)  # normalize to "error"
    elif status in ("eager_error", "create_error", "timeout") and "error" in identify_result:
        mode_data["error"] = identify_result["error"]
    elif status == "clean":
        mode_data.pop("error", None)
        mode_data.pop("fullgraph_error", None)

    # Clear explain fields if model is no longer graph_break
    if status != "graph_break":
        for field in EXPLAIN_FIELDS:
            mode_data.pop(field, None)


def merge_explain_into_mode(mode_data, explain_result):
    """Merge explain result fields into a corpus mode entry."""
    if explain_result.get("status") == "explain_error":
        mode_data["explain_error"] = explain_result.get("error", "unknown")
        return

    for field in EXPLAIN_FIELDS:
        if field in explain_result:
            mode_data[field] = explain_result[field]


def update_corpus(corpus, identify_idx, explain_idx, versions):
    """Update corpus with sweep results. Returns (updated_corpus, changelog_entries)."""
    changelog = []
    models_by_name = {}
    for m in corpus["models"]:
        models_by_name[m["name"]] = m

    # Track which models appear in sweep
    sweep_model_names = set()
    for (name, mode) in identify_idx:
        sweep_model_names.add(name)

    # Process each sweep result
    for (name, mode), ident in identify_idx.items():
        if name not in models_by_name:
            # New model
            models_by_name[name] = {
                "name": name,
                "source": ident.get("source", "hf"),
                "has_graph_break": False,
                "eval": {},
                "train": {},
            }
            changelog.append({
                "type": "new_model",
                "name": name,
                "mode": mode,
                "new_status": ident.get("status"),
            })

        model = models_by_name[name]
        mode_data = model.setdefault(mode, {})

        old_status = mode_data.get("status")
        new_status = ident.get("status")

        # Merge identify fields
        merge_identify_into_mode(mode_data, ident)

        # Merge explain fields if available
        explain_key = (name, mode)
        if explain_key in explain_idx:
            merge_explain_into_mode(mode_data, explain_idx[explain_key])

        # Record status change
        if old_status and old_status != new_status:
            changelog.append({
                "type": "status_change",
                "name": name,
                "mode": mode,
                "old_status": old_status,
                "new_status": new_status,
            })

    # Recompute has_graph_break for all affected models
    for name in sweep_model_names:
        if name in models_by_name:
            models_by_name[name]["has_graph_break"] = compute_has_graph_break(models_by_name[name])

    # Update metadata
    if versions:
        meta = corpus.setdefault("metadata", {})
        if "torch" in versions:
            # Parse "2.10.0+cu128" → "2.10.0"
            torch_ver = versions["torch"].split("+")[0]
            meta["pytorch_version"] = torch_ver
        if "transformers" in versions:
            meta["transformers_version"] = versions["transformers"]
        if "diffusers" in versions:
            meta["diffusers_version"] = versions["diffusers"]
        if "python" in versions:
            # Parse "3.12.13+meta (...)" → "3.12.13"
            py_ver = versions["python"].split("+")[0].split(" ")[0]
            meta["python_version"] = py_ver
        meta["last_updated"] = time.strftime("%Y-%m-%d")

    # Rebuild model list (preserve order, put new models at end)
    original_names = [m["name"] for m in corpus["models"]]
    new_names = sorted(set(models_by_name.keys()) - set(original_names))
    ordered = [models_by_name[n] for n in original_names if n in models_by_name]
    ordered.extend([models_by_name[n] for n in new_names])

    corpus["models"] = ordered
    corpus["summary"] = compute_summary(ordered)

    return corpus, changelog


def format_changelog(changelog, sweep_dir, versions, corpus):
    """Format changelog entries into markdown."""
    lines = ["# Corpus Update Changelog\n"]

    # Header
    lines.append(f"**Sweep directory:** `{sweep_dir}`")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}")
    if versions:
        lines.append(f"**PyTorch:** {versions.get('torch', '?')}")
        lines.append(f"**Transformers:** {versions.get('transformers', '?')}")
        lines.append(f"**Diffusers:** {versions.get('diffusers', '?')}")
    lines.append("")

    # Summary
    summary = corpus.get("summary", {})
    lines.append("## Corpus Summary")
    lines.append(f"- Total models: {summary.get('total_models', '?')}")
    lines.append(f"- Models with graph breaks: {summary.get('has_graph_break', '?')}")
    lines.append("")

    if not changelog:
        lines.append("**No changes detected.** Corpus is identical to sweep results.")
        return "\n".join(lines)

    # Categorize changes
    new_models = [c for c in changelog if c["type"] == "new_model"]
    status_changes = [c for c in changelog if c["type"] == "status_change"]

    # Status transitions
    regressions = [c for c in status_changes if c["old_status"] == "clean" and c["new_status"] == "graph_break"]
    fixes = [c for c in status_changes if c["old_status"] == "graph_break" and c["new_status"] == "clean"]
    other_changes = [c for c in status_changes if c not in regressions and c not in fixes]

    lines.append(f"## Changes ({len(changelog)} total)\n")

    if regressions:
        lines.append(f"### Regressions ({len(regressions)}) — clean → graph_break")
        for c in sorted(regressions, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']})")
        lines.append("")

    if fixes:
        lines.append(f"### Fixes ({len(fixes)}) — graph_break → clean")
        for c in sorted(fixes, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']})")
        lines.append("")

    if other_changes:
        lines.append(f"### Other Status Changes ({len(other_changes)})")
        for c in sorted(other_changes, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']}): {c['old_status']} → {c['new_status']}")
        lines.append("")

    if new_models:
        lines.append(f"### New Models ({len(new_models)})")
        for c in sorted(new_models, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']}): {c['new_status']}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Merge sweep results into corpus.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sweep_dir", help="Path to sweep results directory")
    parser.add_argument("--corpus", default=CORPUS_PATH,
                        help="Path to corpus.json (default: corpus/corpus.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip running validate.py after update")
    parser.add_argument("--changelog", default=None,
                        help="Path to write changelog (default: {sweep_dir}/changelog.md)")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    corpus_path = Path(args.corpus).resolve()

    # Load identify results (required)
    identify_path = sweep_dir / "identify_results.json"
    if not identify_path.exists():
        print(f"ERROR: {identify_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading identify results from {identify_path}")
    identify_results = load_sweep_results(identify_path)
    identify_idx = index_by_name_mode(identify_results)
    print(f"  {len(identify_results)} results ({len(set(r['name'] for r in identify_results))} models)")

    # Load explain results (optional)
    explain_path = sweep_dir / "explain_results.json"
    explain_idx = {}
    if explain_path.exists():
        print(f"Loading explain results from {explain_path}")
        explain_results = load_sweep_results(explain_path)
        explain_idx = index_by_name_mode(explain_results)
        print(f"  {len(explain_results)} results")
    else:
        print("No explain results found (skipping break detail merge)")

    # Load versions (optional)
    versions_path = sweep_dir / "versions.json"
    versions = {}
    if versions_path.exists():
        with open(versions_path) as f:
            versions = json.load(f)
        print(f"Loaded versions: torch={versions.get('torch', '?')}, "
              f"transformers={versions.get('transformers', '?')}")

    # Load current corpus
    print(f"\nLoading corpus from {corpus_path}")
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"  {len(corpus['models'])} models")

    # Merge
    print("\nMerging sweep results into corpus...")
    corpus, changelog = update_corpus(corpus, identify_idx, explain_idx, versions)

    # Format changelog
    changelog_text = format_changelog(changelog, str(sweep_dir), versions, corpus)

    # Print summary
    status_changes = [c for c in changelog if c["type"] == "status_change"]
    new_models = [c for c in changelog if c["type"] == "new_model"]
    regressions = [c for c in status_changes if c["old_status"] == "clean" and c["new_status"] == "graph_break"]
    fixes = [c for c in status_changes if c["old_status"] == "graph_break" and c["new_status"] == "clean"]

    print(f"\n  Status changes: {len(status_changes)}")
    print(f"    Regressions (clean→break): {len(regressions)}")
    print(f"    Fixes (break→clean): {len(fixes)}")
    print(f"    Other: {len(status_changes) - len(regressions) - len(fixes)}")
    print(f"  New models: {len(new_models)}")

    if args.dry_run:
        print("\n--- DRY RUN — no files written ---")
        print(changelog_text)
        return

    # Write updated corpus
    print(f"\nWriting corpus to {corpus_path}")
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")

    # Write changelog
    changelog_path = Path(args.changelog) if args.changelog else sweep_dir / "changelog.md"
    print(f"Writing changelog to {changelog_path}")
    with open(changelog_path, "w") as f:
        f.write(changelog_text)

    # Run validate.py
    if not args.no_validate:
        print("\nRunning validate.py...")
        validate_path = Path(__file__).parent / "validate.py"
        result = subprocess.run(
            [sys.executable, str(validate_path), "--skip-tools"],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            print("WARNING: Validation failed! Review the corpus changes.")
            print("Run with --fix to auto-repair: python tools/validate.py --fix")
            sys.exit(1)
    else:
        print("\nSkipping validation (--no-validate)")

    print("Done.")


if __name__ == "__main__":
    main()
