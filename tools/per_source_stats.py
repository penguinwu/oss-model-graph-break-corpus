#!/usr/bin/env python3
"""Per-source breakdown for a sweep result file.

Usage:
    python3 tools/per_source_stats.py <identify_results.json>
    python3 tools/per_source_stats.py sweep_results/baseline/pt2.11/identify_results.json

Reports work-item and unique-model counts per source (hf/diffusers/custom/timm),
with status breakdown. Useful for divide-and-conquer triage: focus on one source
at a time when investigating gated failures.

Backfills the `source` field for older sweeps that didn't tag it (heuristically
by model-name pattern).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# Heuristic: name patterns that strongly indicate diffusers source. Used only
# to backfill the `source` field on older sweeps that didn't tag it explicitly.
DIFFUSERS_NAME_PATTERNS = (
    "Autoencoder", "Transformer3D", "Transformer2D", "UNet", "ControlNet",
    "CogView", "AuraFlow", "Allegro", "Cosmos", "AudioLDM", "StableCascade",
    "OmniGen", "Hunyuan", "Sana", "Flux", "PixArt", "Latte", "Mochi", "Lumina",
    "Kandinsky", "I2VGen", "SD3", "OvisImageTransformer", "ChronoEditTransformer",
    "AmusedTransformer", "PriorTransformer", "ChromaTransformer",
    "BriaFiboTransformer", "BriaTransformer", "WanTransformer", "EasyAnimate",
    "MotionAdapter", "ConsistencyDecoder", "CLIPImageProjection",
    "AsymmetricAutoencoderKL", "SkyReels", "HiDream", "PRXTransformer",
    "GlmImageTransformer", "ErnieImageTransformer", "AutoencoderRAE",
    "AutoencoderOobleck", "AutoencoderDC",
)


def infer_source(name: str) -> str:
    """Heuristic source backfill — used only when the row didn't tag itself."""
    if any(p in name for p in DIFFUSERS_NAME_PATTERNS):
        return "diffusers"
    return "hf"


def report(path: Path) -> None:
    # Route through canonical loader so amendments are merged
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sweep.results_loader import load_results_list
    results = load_results_list(path)

    # Backfill source on rows that don't have it
    backfilled = 0
    for r in results:
        if not r.get("source"):
            r["source"] = infer_source(r["name"])
            backfilled += 1

    by_src_status = defaultdict(Counter)
    by_src_models = defaultdict(set)
    for r in results:
        src = r["source"]
        by_src_status[src][r.get("status", "unknown")] += 1
        by_src_models[src].add(r["name"])

    print(f"Sweep: {path}")
    print(f"Total work items: {len(results)}"
          + (f"  ({backfilled} sources backfilled)" if backfilled else ""))
    print()
    print(f"{'source':10s}  {'work':>6s}  {'models':>7s}  {'full_graph':>11s}  "
          f"{'graph_break':>12s}  {'gated_fail':>11s}  {'other':>6s}")
    print("-" * 76)
    for src in sorted(by_src_status):
        counts = by_src_status[src]
        total = sum(counts.values())
        models = len(by_src_models[src])
        full = counts.get("full_graph", 0)
        gb = counts.get("graph_break", 0)
        gated = counts.get("create_error", 0) + counts.get("eager_error", 0)
        other = total - full - gb - gated
        print(f"{src:10s}  {total:6d}  {models:7d}  {full:11d}  {gb:12d}  "
              f"{gated:11d}  {other:6d}")

    # Detailed status breakdown per source
    print()
    print("Detailed status per source:")
    for src in sorted(by_src_status):
        statuses = by_src_status[src]
        breakdown = ", ".join(f"{k}: {v}" for k, v in sorted(statuses.items()))
        print(f"  {src:10s}: {breakdown}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, help="Path to identify_results.json or correctness_results.json")
    args = p.parse_args()
    if not args.path.exists():
        sys.exit(f"ERROR: file not found: {args.path}")
    report(args.path)


if __name__ == "__main__":
    main()
