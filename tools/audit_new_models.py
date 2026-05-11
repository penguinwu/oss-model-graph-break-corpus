#!/usr/bin/env python3
"""audit_new_models.py — Step 2b of the weekly sweep workflow.

For NEW models in current cohort (not in baseline), propose tier classification
+ per-row triage. For REMOVED models, classify by source.

Surfaces only — never auto-writes config files.

Design: sweep/AUDIT_NEW_MODELS_DESIGN.md (rev 2).
Adversary review: subagents/adversary-review/invocations/adv-2026-05-10-150000-audit-new-models-design.md

Usage:
    python3 tools/audit_new_models.py <sweep_dir>

Exit codes:
    0 — report written
    1 — input parse error
    3 — compare-vs-baseline.json absent (degraded mode)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from results_loader import load_effective_results  # noqa: E402

ERROR_STATUSES = {"eager_error", "create_error", "worker_error", "timeout"}

# Tier thresholds — wall-clock only per adversary case adv-2026-05-10-150000 gap #4.
# CHANGING THESE: update tools/test_audit_new_models.py::test_threshold_constants_pinned
# AND sweep/WEEKLY_SWEEP_WORKFLOW.md AND invoke adversary-review.
WALL_LARGE_S = 60
WALL_VERY_LARGE_S = 300

LARGE_MODELS_FILE = REPO_ROOT / "sweep" / "large_models.json"
SKIP_MODELS_FILE = REPO_ROOT / "sweep" / "skip_models.json"
KNOWN_ERRORS_FILE = REPO_ROOT / "sweep" / "known_errors.json"


def classify_tier(wall_time_s: float | None) -> str:
    """Tier from wall-clock. Boundary: ≤60s regular, ≤300s large, else very_large."""
    if wall_time_s is None:
        raise ValueError("wall_time_s is None on a non-error row — unexpected")
    if wall_time_s <= WALL_LARGE_S:
        return "regular"
    if wall_time_s <= WALL_VERY_LARGE_S:
        return "large"
    return "very_large"


def classify_timeout_tier(phase_at_timeout: str | None) -> str:
    """Timeout rows can't measure wall_time_s; infer from phase."""
    return "very_large" if phase_at_timeout == "create" else "large"


def load_known_errors_models(active_torch_major_minor: str | None) -> set:
    """Return set of model names covered by known_errors.json for active torch."""
    if not KNOWN_ERRORS_FILE.exists():
        return set()
    with open(KNOWN_ERRORS_FILE) as f:
        data = json.load(f)
    out = set()
    for entry in data.get("entries", []):
        applies = entry.get("applies_to_versions")
        if applies is None:
            print(
                f"FATAL: known_errors.json entry for {entry.get('model')!r} "
                f"missing 'applies_to_versions'; rejected per audit_new_errors design rev 2.",
                file=sys.stderr,
            )
            sys.exit(2)
        if active_torch_major_minor is None or active_torch_major_minor in applies:
            out.add(entry["model"])
    return out


def load_skip_models() -> set:
    """Delegates to sweep.skip_models_loader (handles legacy + dict formats)."""
    from skip_models_loader import load_skip_models as _load
    return _load(SKIP_MODELS_FILE)


def torch_major_minor(sweep_dir: Path) -> str | None:
    state = sweep_dir / "sweep_state.json"
    if not state.exists():
        return None
    try:
        with state.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    ver = (data.get("versions") or {}).get("torch", "")
    parts = ver.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{parts[0]}.{parts[1]}"
    return None


def load_compare(sweep_dir: Path) -> dict | None:
    p = sweep_dir / "compare-vs-baseline.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_audit_new_errors_sidecar(sweep_dir: Path) -> dict | None:
    """Return {(name, mode): triage_class} from audit-new-errors.json if present."""
    p = sweep_dir / "audit-new-errors.json"
    if not p.exists():
        return None
    with p.open() as f:
        data = json.load(f)
    out = {}
    for c in data.get("candidates", []):
        out[(c.get("name"), c.get("mode"))] = c.get("triage_class")
    return out


def collect_new_models(compare: dict, current_rows: dict,
                       fixture_fix_map: dict | None) -> list:
    """Per-name rollup of cat4 entries.

    Returns list of dicts, one per unique model name in cat4.
    """
    cat4 = compare.get("cat4", [])
    by_name: dict[str, list] = defaultdict(list)
    for entry in cat4:
        key = entry.get("key")
        if isinstance(key, list):
            key = tuple(key)
        if not key:
            continue
        name, mode = key
        by_name[name].append((mode, entry))

    out = []
    for name, mode_entries in sorted(by_name.items()):
        modes = [m for m, _ in mode_entries]
        # Look up wall_time per mode from current rows
        per_mode_walls = []
        any_error = False
        any_timeout = False
        timeout_phase_at = None
        fixture_fix_classes = []
        for mode, entry in mode_entries:
            row = current_rows.get((name, mode), {})
            status = row.get("status")
            if status in ERROR_STATUSES:
                any_error = True
                if status == "timeout":
                    any_timeout = True
                    timeout_phase_at = row.get("phase_at_timeout")
                if fixture_fix_map is not None:
                    cls = fixture_fix_map.get((name, mode))
                    if cls:
                        fixture_fix_classes.append(f"{mode}={cls}")
                continue
            # Non-error row: assert wall_time_s present per gap #5
            wall = row.get("wall_time_s")
            if wall is None:
                # Defensive: real data always has this; if missing, surface loudly.
                per_mode_walls.append(None)
            else:
                per_mode_walls.append(float(wall))

        valid_walls = [w for w in per_mode_walls if w is not None]
        max_wall = max(valid_walls) if valid_walls else None

        if any_timeout:
            proposed_tier = classify_timeout_tier(timeout_phase_at)
        elif any_error:
            proposed_tier = None  # fixture-fix needed first
        elif max_wall is not None:
            proposed_tier = classify_tier(max_wall)
        else:
            proposed_tier = None

        out.append({
            "name": name,
            "modes": modes,
            "max_wall_s": max_wall,
            "any_error": any_error,
            "any_timeout": any_timeout,
            "proposed_tier": proposed_tier,
            "fixture_fix_classes": fixture_fix_classes,
            "fixture_fix_link_available": fixture_fix_map is not None,
        })
    return out


def collect_removed_models(compare: dict, skip_set: set,
                           known_error_models: set) -> list:
    """Three sources of REMOVED:
    1. skip_listed (only_in baseline) → intentional-skip
    2. cat5 + name in known_errors → known-error-evolution
    3. cat5 otherwise → unexpected-removal
    """
    out = []
    seen_names = set()

    # Source 1: skip_listed entries that were only in baseline
    for entry in compare.get("skip_listed", []):
        key = entry.get("key")
        if isinstance(key, list):
            key = tuple(key)
        if not key:
            continue
        name = key[0]
        # only_in field tells us baseline vs current side
        only_in = entry.get("only_in")
        if only_in == "baseline" or (only_in is None and name in skip_set):
            if name in seen_names:
                continue
            seen_names.add(name)
            out.append({
                "name": name,
                "classification": "intentional-skip",
                "evidence": f"in skip_models.json; was in baseline only",
            })

    # Source 2 + 3: cat5 entries (unexpected removals at the cat-partition level)
    for entry in compare.get("cat5", []):
        key = entry.get("key")
        if isinstance(key, list):
            key = tuple(key)
        if not key:
            continue
        name = key[0]
        if name in seen_names:
            continue
        seen_names.add(name)
        if name in known_error_models:
            classification = "known-error-evolution"
            evidence = "in known_errors.json for active torch — entry may have evolved"
        else:
            classification = "unexpected-removal"
            evidence = "not in skip_models.json and not in known_errors.json — likely transformers refactor"
        out.append({
            "name": name,
            "classification": classification,
            "evidence": evidence,
        })

    return sorted(out, key=lambda x: (x["classification"], x["name"]))


def render_report(sweep_dir: Path, new_models: list, removed_models: list,
                  fixture_fix_available: bool, versions: dict,
                  compare: dict | None) -> str:
    lines = []
    em = lines.append
    em(f"# New / removed models triage — sweep {sweep_dir.name}")
    em("")
    em(f"torch: {versions.get('torch', '?')}  "
       f"transformers: {versions.get('transformers', '?')}")
    em("")

    new_names = len(new_models)
    new_pairs = sum(len(m["modes"]) for m in new_models)
    em(f"## Cohort delta")
    em(f"- NEW: {new_names} unique models ({new_pairs} (name, mode) pairs)")
    em(f"- REMOVED: {len(removed_models)} models")
    em("")

    # NEW
    em("## NEW models (cat 4)")
    em("")
    if not new_models:
        em("_None._")
    else:
        em("| Model | Modes | max wall (s) | Proposed tier | Notes |")
        em("|---|---|---|---|---|")
        for m in new_models:
            modes_str = ",".join(m["modes"])
            tier = m["proposed_tier"] or "(deferred — fixture-fix needed)"
            notes = []
            if m["any_error"]:
                if fixture_fix_available and m["fixture_fix_classes"]:
                    notes.append(f"fixture-fix needed; sister-tool says: {', '.join(m['fixture_fix_classes'])}")
                else:
                    notes.append("fixture-fix needed; run audit_new_errors first for triage")
            if m["any_timeout"]:
                notes.append(f"TIMEOUT — propose {tier}")
            wall_str = f"{m['max_wall_s']:.1f}" if m["max_wall_s"] is not None else "-"
            em(f"| {m['name']} | {modes_str} | {wall_str} | {tier} | {'; '.join(notes) if notes else 'tier-classify only'} |")
    em("")

    # REMOVED
    em("## REMOVED models (cat 5 + skip_listed)")
    em("")
    if not removed_models:
        em("_None._")
    else:
        by_class: dict[str, list] = defaultdict(list)
        for m in removed_models:
            by_class[m["classification"]].append(m)
        for cls in ["intentional-skip", "known-error-evolution", "unexpected-removal"]:
            entries = by_class.get(cls, [])
            em(f"### {cls} ({len(entries)})")
            em("")
            if not entries:
                em("_None._")
            else:
                em("| Model | Evidence |")
                em("|---|---|")
                for m in entries:
                    em(f"| {m['name']} | {m['evidence']} |")
            em("")

    # Tier upgrade proposals (per-name, with evidence)
    upgrades = [m for m in new_models if m["proposed_tier"] in {"large", "very_large"}]
    em("## Proposed tier upgrades (large_models.json edits)")
    em("")
    if upgrades:
        em("Each entry below is a PROPOSAL — reviewer approves before commit. "
           "Format matches existing `sweep/large_models.json` schema.")
        em("```json")
        for m in upgrades:
            ts = m["proposed_tier"]
            em(f'  "{m["name"]}": {{')
            em(f'    "discovered": "{sweep_dir.name}",')
            em(f'    "source": "hf",')
            em(f'    "timeout_tier": "{ts}",')
            if m["max_wall_s"] is not None:
                em(f'    "wall_time_s": {m["max_wall_s"]:.2f}')
            em("  },")
        em("```")
    else:
        em("_No tier upgrade proposals._")
    em("")

    # skip_models.json — surface text only (schema doesn't support TEMPORARY today)
    em("## Skip-list proposals (text-only — schema doesn't support TEMPORARY today)")
    em("")
    skip_proposals = [m for m in new_models if m["any_error"] and not m["any_timeout"]]
    if skip_proposals:
        em("These NEW models have unfixed fixture errors. Per Peng directive, "
           "skip-list adds are TEMPORARY but `sweep/skip_models.json` is a flat "
           "string array today (no metadata). Reviewer must hand-decide; tool does "
           "NOT emit JSON edits. Schema upgrade is a separate WS1 task.")
        em("")
        for m in skip_proposals:
            cls = ', '.join(m['fixture_fix_classes']) if m['fixture_fix_classes'] else '?'
            em(f"- {m['name']} — fixture-fix triage: {cls}")
    else:
        em("_No skip-list candidates._")
    em("")

    em("## Action checklist for Peng review")
    em("- [ ] Approve `large_models.json` tier-upgrade proposals (each: model + tier + evidence)")
    em("- [ ] Triage `unexpected-removal` cases — likely transformers refactor; investigate")
    em("- [ ] Review `known-error-evolution` cases — known_errors.json entry may need update")
    em("- [ ] Triage NEW-model fixture errors — these block tier classification")
    if not fixture_fix_available:
        em("- [ ] **Run `tools/audit_new_errors.py` first** — fixture-fix linkage was unavailable")
    em("")
    return "\n".join(lines) + "\n"


def run_audit(sweep_dir: Path) -> int:
    if not (sweep_dir / "identify_results.json").exists():
        print(f"FATAL: {sweep_dir}/identify_results.json not found", file=sys.stderr)
        return 1
    rows = load_effective_results(sweep_dir)

    compare = load_compare(sweep_dir)
    fixture_fix_map = load_audit_new_errors_sidecar(sweep_dir)
    skip_set = load_skip_models()
    tmm = torch_major_minor(sweep_dir)
    known_error_models = load_known_errors_models(tmm)

    versions = {}
    state = sweep_dir / "sweep_state.json"
    if state.exists():
        try:
            versions = json.loads(state.read_text()).get("versions") or {}
        except json.JSONDecodeError:
            pass

    if compare is None:
        # Degraded mode — emit minimal report
        report = (
            f"# Audit new models — sweep {sweep_dir.name}\n\n"
            f"**DEGRADED MODE — no compare-vs-baseline.json present.**\n\n"
            f"Run sweep_compare first: `python3 tools/sweep_compare.py "
            f"--baseline <prior> --current {sweep_dir} --json {sweep_dir}/compare-vs-baseline.json`\n"
        )
        (sweep_dir / "audit-new-models.md").write_text(report)
        (sweep_dir / "audit-new-models.json").write_text(json.dumps(
            {"sweep_date": sweep_dir.name, "degraded": True}, indent=2))
        print(f"Wrote: {sweep_dir / 'audit-new-models.md'} (degraded mode)")
        return 3

    new_models = collect_new_models(compare, rows, fixture_fix_map)
    removed_models = collect_removed_models(compare, skip_set, known_error_models)

    md = render_report(sweep_dir, new_models, removed_models,
                       fixture_fix_map is not None, versions, compare)
    (sweep_dir / "audit-new-models.md").write_text(md)

    cohort_delta = {
        "new": len(new_models),
        "new_pairs": sum(len(m["modes"]) for m in new_models),
        "removed": len(removed_models),
    }
    json_payload = {
        "sweep_date": sweep_dir.name,
        "torch_version": versions.get("torch"),
        "cohort_delta": cohort_delta,
        "fixture_fix_link_available": fixture_fix_map is not None,
        "new_models": new_models,
        "removed_models": removed_models,
    }
    (sweep_dir / "audit-new-models.json").write_text(json.dumps(json_payload, indent=2))

    print(f"Wrote: {sweep_dir / 'audit-new-models.md'}")
    print(f"Wrote: {sweep_dir / 'audit-new-models.json'}")
    print(f"Cohort delta: NEW={cohort_delta['new']} ({cohort_delta['new_pairs']} pairs), REMOVED={cohort_delta['removed']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()
    return run_audit(args.sweep_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
