#!/usr/bin/env python3
"""Merge sweep results into corpus.json with changelog generation.

Three modes:
    Overlay (default) — partial sweep results merged on top of existing corpus.
        Models not in the sweep keep their existing entries.
        Version and timestamp safety checks prevent accidental cross-version merges.
        Requires identify_results.json.

    Replace (--replace) — full sweep results become the complete corpus.
        Models not in the sweep are dropped.
        Requires identify_results.json with metadata.versions.

    Explain-only (auto) — when only explain_results.json exists (no identify).
        Overlays break_reasons, graph counts onto existing corpus entries.
        No version/timestamp checks (not changing model status).
        Does not add or remove models.

Usage:
    # Overlay: merge partial sweep into corpus
    python tools/update_corpus.py sweep_results/pt2.10/

    # Replace: full sweep becomes the new corpus
    python tools/update_corpus.py sweep_results/pt2.10/ --replace

    # Explain-only: merge just explain data (auto-detected)
    python tools/update_corpus.py sweep_results/pt2.12_explain/

    # Dry run — show what would change without writing
    python tools/update_corpus.py sweep_results/pt2.10/ --dry-run

    # Skip validation after update
    python tools/update_corpus.py sweep_results/pt2.10/ --no-validate

For overlay/replace, the sweep directory should contain:
    - identify_results.json (required — must include metadata.versions)
    - explain_results.json (optional — merges break_reasons, graph counts)

For explain-only, the directory needs only:
    - explain_results.json

Version info is read from identify_results.json metadata.versions (embedded
by run_sweep.py at sweep time). No separate versions.json needed.

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
    elif status == "full_graph":
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


def update_corpus(corpus, identify_idx, explain_idx, versions, replace=False):
    """Update corpus with sweep results. Returns (updated_corpus, changelog, pre_merge_names, health).

    In overlay mode (default): sweep results are merged on top of existing corpus.
    Models not in the sweep keep their existing entries.

    In replace mode: sweep results become the complete corpus.
    Models not in the sweep are dropped.
    """
    changelog = []
    models_by_name = {}
    for m in corpus["models"]:
        models_by_name[m["name"]] = m

    # Snapshot pre-merge state for diagnostics
    pre_merge_names = set(models_by_name.keys())
    pre_merge_errors = set()
    for m in corpus["models"]:
        for mode_key in ("eval", "train"):
            if m.get(mode_key, {}).get("status") == "create_error":
                pre_merge_errors.add(m["name"])

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

    # Replace mode: drop models not in the sweep
    if replace:
        removed = pre_merge_names - sweep_model_names
        for name in removed:
            del models_by_name[name]
            changelog.append({
                "type": "removed_model",
                "name": name,
            })

    # Update metadata — only overwrite versions in replace mode
    if versions:
        meta = corpus.setdefault("metadata", {})
        if replace:
            if "torch" in versions:
                torch_ver = versions["torch"].split("+")[0]
                meta["pytorch_version"] = torch_ver
            if "transformers" in versions:
                meta["transformers_version"] = versions["transformers"]
            if "diffusers" in versions:
                meta["diffusers_version"] = versions["diffusers"]
            if "python" in versions:
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

    # Post-merge health diagnostics
    health = {}

    # Risk 3: Ghost entries — models with create_error both before and after merge
    post_merge_errors = set()
    for m in ordered:
        for mode_key in ("eval", "train"):
            if m.get(mode_key, {}).get("status") == "create_error":
                post_merge_errors.add(m["name"])
    persistent_errors = pre_merge_errors & post_merge_errors
    if persistent_errors:
        health["persistent_create_errors"] = sorted(persistent_errors)

    # Risk 4: Missing explain data — graph_break models without break_reasons
    missing_explain = []
    for m in ordered:
        for mode_key in ("eval", "train"):
            md = m.get(mode_key, {})
            if md.get("status") == "graph_break" and not md.get("break_reasons"):
                missing_explain.append((m["name"], mode_key))
    if missing_explain:
        health["missing_explain"] = missing_explain

    return corpus, changelog, pre_merge_names, health


# HF model class suffixes, longest first for greedy stripping
_HF_SUFFIXES = sorted([
    "ForCausalLM", "ForConditionalGeneration", "ForSequenceClassification",
    "ForTokenClassification", "ForQuestionAnswering", "ForMaskedLM",
    "ForMultipleChoice", "ForPreTraining", "ForImageClassification",
    "LMHeadModel", "DoubleHeadsModel", "WithLMHeadModel",
    "EncoderModel", "DecoderModel", "TextModel", "VisionModel",
    "AudioModel", "BaseModel", "Model",
], key=len, reverse=True)


def _model_family(name):
    """Extract model family from a HF class name by stripping task suffixes.

    E.g. 'LlamaForCausalLM' → 'Llama', 'Gemma3Model' → 'Gemma3'.
    """
    for suffix in _HF_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[:-len(suffix)]
    return name


def _classify_new_models(new_model_entries, existing_model_names):
    """Split new model entries into new families vs new configurations.

    Returns (new_families, new_configs) where each is a list of changelog entries.
    A 'new family' is a model whose family name doesn't appear among existing models.
    A 'new config' is a variant (e.g. ForCausalLM) of an already-present family.
    """
    existing_families = {_model_family(n) for n in existing_model_names}

    # Deduplicate: a model can appear twice (eval + train); classify once per name
    seen = set()
    new_families = []
    new_configs = []
    for entry in sorted(new_model_entries, key=lambda x: x["name"]):
        name = entry["name"]
        if name in seen:
            continue
        seen.add(name)
        family = _model_family(name)
        if family in existing_families:
            new_configs.append(entry)
        else:
            new_families.append(entry)
            existing_families.add(family)  # subsequent variants of this family → config
    return new_families, new_configs


def format_changelog(changelog, sweep_dir, versions, corpus, pre_merge_names=None,
                     health=None):
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
    removed_models = [c for c in changelog if c["type"] == "removed_model"]
    status_changes = [c for c in changelog if c["type"] == "status_change"]

    # Status transitions
    regressions = [c for c in status_changes if c["old_status"] == "full_graph" and c["new_status"] == "graph_break"]
    fixes = [c for c in status_changes if c["old_status"] == "graph_break" and c["new_status"] == "full_graph"]
    other_changes = [c for c in status_changes if c not in regressions and c not in fixes]

    lines.append(f"## Changes ({len(changelog)} total)\n")

    if regressions:
        lines.append(f"### Regressions ({len(regressions)}) — full_graph → graph_break")
        for c in sorted(regressions, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']})")
        lines.append("")

    if fixes:
        lines.append(f"### Fixes ({len(fixes)}) — graph_break → full_graph")
        for c in sorted(fixes, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']})")
        lines.append("")

    if other_changes:
        lines.append(f"### Other Status Changes ({len(other_changes)})")
        for c in sorted(other_changes, key=lambda x: x["name"]):
            lines.append(f"- **{c['name']}** ({c['mode']}): {c['old_status']} → {c['new_status']}")
        lines.append("")

    if new_models:
        # Classify into new families vs new configurations
        existing_names = pre_merge_names or set()
        new_families, new_configs = _classify_new_models(new_models, existing_names)

        # Deduplicate: each model name listed once (eval entry preferred)
        def _dedup(entries):
            seen = set()
            out = []
            for c in sorted(entries, key=lambda x: (x["name"], x["mode"])):
                if c["name"] not in seen:
                    seen.add(c["name"])
                    out.append(c)
            return out

        new_families_dedup = _dedup(new_families)
        new_configs_dedup = _dedup(new_configs)

        lines.append(f"### New Model Families ({len(new_families_dedup)})")
        lines.append("Entirely new architectures not previously in the corpus.\n")
        if new_families_dedup:
            for c in new_families_dedup:
                lines.append(f"- **{c['name']}** ({c['mode']}): {c['new_status']}")
        else:
            lines.append("_(none)_")
        lines.append("")

        lines.append(f"### New Configurations ({len(new_configs_dedup)})")
        lines.append("Task-head variants (ForCausalLM, ForConditionalGeneration, etc.) "
                      "of model families already in the corpus.\n")
        if new_configs_dedup:
            for c in new_configs_dedup:
                family = _model_family(c["name"])
                lines.append(f"- **{c['name']}** ({c['mode']}): {c['new_status']}  ← {family}")
        else:
            lines.append("_(none)_")
        lines.append("")

    if removed_models:
        removed_names = sorted(set(c["name"] for c in removed_models))
        lines.append(f"### Removed Models ({len(removed_names)})")
        lines.append("Models in the previous corpus but not in this sweep (--replace mode).\n")
        for name in removed_names:
            lines.append(f"- **{name}**")
        lines.append("")

    # Health warnings
    if health:
        lines.append("## Health Warnings\n")

        persistent = health.get("persistent_create_errors", [])
        if persistent:
            lines.append(f"### Persistent Failures ({len(persistent)})")
            lines.append("Models with `create_error` in both the previous and current corpus.")
            lines.append("These may have been removed upstream — consider removing from corpus.\n")
            for name in persistent:
                lines.append(f"- **{name}**")
            lines.append("")

        missing = health.get("missing_explain", [])
        if missing:
            # Deduplicate by model name for readability
            missing_names = sorted(set(name for name, mode in missing))
            lines.append(f"### Missing Explain Data ({len(missing_names)} models)")
            lines.append("Models with `graph_break` status but no `break_reasons`.")
            lines.append("Run an explain pass to fill in break details.\n")
            for name in missing_names:
                modes = [mode for n, mode in missing if n == name]
                lines.append(f"- **{name}** ({', '.join(modes)})")
            lines.append("")

    return "\n".join(lines)


def update_corpus_explain_only(corpus, explain_idx):
    """Overlay explain data onto existing corpus entries.

    Unlike update_corpus(), this does NOT touch identify fields, add/remove models,
    or update version metadata. It only fills in break_reasons, graph counts for
    existing graph_break entries.

    Returns (updated_corpus, changelog, health).
    """
    changelog = []
    models_by_name = {m["name"]: m for m in corpus["models"]}

    updated_count = 0
    skipped_not_found = []

    for (name, mode), explain_result in explain_idx.items():
        if name not in models_by_name:
            skipped_not_found.append((name, mode))
            continue

        model = models_by_name[name]
        mode_data = model.get(mode, {})

        if not mode_data:
            skipped_not_found.append((name, mode))
            continue

        merge_explain_into_mode(mode_data, explain_result)
        updated_count += 1
        changelog.append({
            "type": "explain_updated",
            "name": name,
            "mode": mode,
        })

    # Recompute summary (graph counts may have changed)
    corpus["summary"] = compute_summary(corpus["models"])

    # Health: remaining missing explain
    missing_explain = []
    for m in corpus["models"]:
        for mode_key in ("eval", "train"):
            md = m.get(mode_key, {})
            if md.get("status") == "graph_break" and not md.get("break_reasons"):
                missing_explain.append((m["name"], mode_key))

    health = {}
    if missing_explain:
        health["missing_explain"] = missing_explain
    if skipped_not_found:
        health["explain_skipped_not_in_corpus"] = skipped_not_found

    return corpus, changelog, health


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
    parser.add_argument("--replace", action="store_true",
                        help="Full replacement: sweep results become the entire corpus. "
                             "Models not in the sweep are dropped.")
    parser.add_argument("--force", action="store_true",
                        help="Allow merging across different PyTorch versions")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    corpus_path = Path(args.corpus).resolve()

    identify_path = sweep_dir / "identify_results.json"
    explain_path = sweep_dir / "explain_results.json"

    # Detect mode: explain-only when no identify_results.json
    explain_only = not identify_path.exists() and explain_path.exists()

    if explain_only:
        if args.replace:
            print("ERROR: --replace requires identify_results.json", file=sys.stderr)
            sys.exit(1)

        print(f"Mode: EXPLAIN-ONLY (no identify_results.json found)")
        print(f"Loading explain results from {explain_path}")
        explain_results = load_sweep_results(explain_path)
        explain_idx = index_by_name_mode(explain_results)
        explain_models = set(name for name, mode in explain_idx)
        print(f"  {len(explain_results)} results ({len(explain_models)} models)")

        # Load corpus
        print(f"\nLoading corpus from {corpus_path}")
        with open(corpus_path) as f:
            corpus = json.load(f)
        print(f"  {len(corpus['models'])} models")

        # Merge explain only
        print("\nOverlaying explain data onto corpus...")
        corpus, changelog, health = update_corpus_explain_only(corpus, explain_idx)

        # Print summary
        updated = [c for c in changelog if c["type"] == "explain_updated"]
        updated_models = set(c["name"] for c in updated)
        print(f"  Updated: {len(updated)} entries ({len(updated_models)} models)")

        skipped = health.get("explain_skipped_not_in_corpus", [])
        if skipped:
            skipped_models = set(name for name, mode in skipped)
            print(f"  Skipped: {len(skipped_models)} models not in corpus")

        missing = health.get("missing_explain", [])
        if missing:
            missing_names = sorted(set(name for name, mode in missing))
            print(f"\n  ⚠ Still missing explain data: {len(missing_names)} graph_break models")

        if args.dry_run:
            print("\n--- DRY RUN — no files written ---")
            return

        # Write
        print(f"\nWriting corpus to {corpus_path}")
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=2)
            f.write("\n")

        changelog_path = Path(args.changelog) if args.changelog else sweep_dir / "changelog.md"
        print(f"Writing changelog to {changelog_path}")
        lines = [
            "# Explain-Only Merge Changelog\n",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
            f"**Source:** `{sweep_dir}`",
            f"**Models updated:** {len(updated_models)}",
            f"**Entries updated:** {len(updated)}",
            "",
        ]
        if missing:
            missing_names = sorted(set(name for name, mode in missing))
            lines.append(f"## Still Missing Explain Data ({len(missing_names)} models)\n")
            for name in missing_names:
                modes = [mode for n, mode in missing if n == name]
                lines.append(f"- **{name}** ({', '.join(modes)})")
        with open(changelog_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        # Validate
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
                sys.exit(1)

        print("Done.")
        return

    # --- Full merge (identify + optional explain) ---

    if not identify_path.exists():
        print(f"ERROR: {identify_path} not found", file=sys.stderr)
        print(f"  Expected identify_results.json or explain_results.json in {sweep_dir}")
        sys.exit(1)

    print(f"Loading identify results from {identify_path}")
    with open(identify_path) as f:
        identify_data = json.load(f)
    identify_meta = identify_data.get("metadata", {}) if isinstance(identify_data, dict) else {}
    identify_results = identify_data.get("results", identify_data) if isinstance(identify_data, dict) else identify_data
    identify_idx = index_by_name_mode(identify_results)
    print(f"  {len(identify_results)} results ({len(set(r['name'] for r in identify_results))} models)")

    # Load explain results (optional)
    explain_idx = {}
    if explain_path.exists():
        print(f"Loading explain results from {explain_path}")
        explain_results = load_sweep_results(explain_path)
        explain_idx = index_by_name_mode(explain_results)
        print(f"  {len(explain_results)} results")
    else:
        print("No explain results found (skipping break detail merge)")

    # Load versions from results metadata
    versions = identify_meta.get("versions", {})
    if versions:
        print(f"Versions: torch={versions.get('torch', '?')}, "
              f"transformers={versions.get('transformers', '?')}")
    else:
        if args.replace:
            print(f"\nERROR: --replace requires version info in results metadata.",
                  file=sys.stderr)
            print(f"  A full corpus replacement must record which versions it was built from.")
            sys.exit(1)
        print("\nWARNING: No version info in results metadata — cannot verify version match.")

    # Load current corpus
    print(f"\nLoading corpus from {corpus_path}")
    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"  {len(corpus['models'])} models")

    if args.replace:
        print(f"\n  Mode: REPLACE (full sweep → complete corpus replacement)")
    else:
        # Version safety check (overlay mode only — replace doesn't mix versions)
        meta = corpus.get("metadata", {})
        version_checks = [
            ("PyTorch", meta.get("pytorch_version", ""),
             versions.get("torch", "").split("+")[0]),
            ("transformers", meta.get("transformers_version", ""),
             versions.get("transformers", "")),
            ("diffusers", meta.get("diffusers_version", ""),
             versions.get("diffusers", "")),
        ]
        mismatches = [(name, corp, sweep)
                      for name, corp, sweep in version_checks
                      if corp and sweep and corp != sweep]
        if mismatches:
            print(f"\nERROR: Version mismatch detected!")
            for name, corp, sweep in mismatches:
                print(f"  {name}:  corpus={corp}  sweep={sweep}")
            print(f"\nMerging results from different versions produces a mixed corpus")
            print(f"where skipped models reflect old versions but metadata claims new.")
            print(f"\nTo update versions, run a full sweep first (with --replace).")
            print(f"To force overlay anyway: --force")
            if not args.force:
                sys.exit(1)
            print(f"\n--force specified, proceeding despite version mismatch...")

        # Timestamp staleness check (overlay mode only)
        corpus_updated = meta.get("last_updated", "")
        sweep_timestamp = identify_meta.get("timestamp", "")
        if corpus_updated and sweep_timestamp:
            sweep_date = sweep_timestamp[:10]  # "2026-04-15T..." → "2026-04-15"
            if sweep_date < corpus_updated:
                print(f"\nWARNING: Sweep results ({sweep_date}) are older than "
                      f"corpus ({corpus_updated}).")
                print("  You may be overwriting newer data with older results.")
                if not args.force:
                    print("  Use --force to proceed anyway.")
                    sys.exit(1)

    # Merge
    print("\nMerging sweep results into corpus...")
    corpus, changelog, pre_merge_names, health = update_corpus(
        corpus, identify_idx, explain_idx, versions, replace=args.replace)

    # Format changelog
    changelog_text = format_changelog(
        changelog, str(sweep_dir), versions, corpus, pre_merge_names, health)

    # Print summary
    status_changes = [c for c in changelog if c["type"] == "status_change"]
    new_models = [c for c in changelog if c["type"] == "new_model"]
    regressions = [c for c in status_changes if c["old_status"] == "full_graph" and c["new_status"] == "graph_break"]
    fixes = [c for c in status_changes if c["old_status"] == "graph_break" and c["new_status"] == "full_graph"]

    print(f"\n  Status changes: {len(status_changes)}")
    print(f"    Regressions (full_graph→break): {len(regressions)}")
    print(f"    Fixes (break→full_graph): {len(fixes)}")
    print(f"    Other: {len(status_changes) - len(regressions) - len(fixes)}")
    removed_models = [c for c in changelog if c["type"] == "removed_model"]
    new_fam, new_cfg = _classify_new_models(new_models, pre_merge_names)
    print(f"  New entries: {len(new_models)}")
    print(f"    New model families: {len(new_fam)}")
    print(f"    New configurations: {len(new_cfg)}")
    if removed_models:
        print(f"  Removed: {len(removed_models)} models (not in sweep, --replace mode)")

    # Print health warnings
    if health:
        persistent = health.get("persistent_create_errors", [])
        if persistent:
            print(f"\n  ⚠ Persistent create_error: {len(persistent)} models")
            print(f"    May be removed upstream. Review: {', '.join(persistent[:5])}")
            if len(persistent) > 5:
                print(f"    ... and {len(persistent) - 5} more (see changelog)")
        missing = health.get("missing_explain", [])
        if missing:
            missing_names = sorted(set(name for name, mode in missing))
            print(f"\n  ⚠ Missing explain data: {len(missing_names)} graph_break models")
            print(f"    Run explain pass to fill in break details.")

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
