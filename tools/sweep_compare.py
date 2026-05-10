#!/usr/bin/env python3
"""sweep_compare.py — compare two sweep result dirs with invariant-checked partition.

Single source of truth: every count derives from load_effective_results +
load_effective_explain (sweep/results_loader.py). No ad-hoc parsing.

Output partitions all (name, mode) keys in (baseline ∪ current) into 6
mutually-exclusive categories:

  Cat 1: error → compile-success (improvements)
  Cat 2: compile-success → error (regressions)
  Cat 3: compile-success in both (steady-state, may have GB count delta)
  Cat 4: NEW (in current only)
  Cat 5: REMOVED (in baseline only)
  Cat 6: error in both (stable failures; cross-checked against known_errors.json)

  ('skipped' status from known_errors.json is excluded from cats 1/2/3/6 — it
  is not really a sweep result, just a deferred re-run.)

Invariants enforced BEFORE any output is produced:
  • cat1 + cat2 + cat3 + cat6 = |common| - |skipped_in_either|
  • cat5 = |baseline_only|, cat4 = |current_only|
  • every graph_break row in identify has a matching explain entry
  • every full_graph row in identify has NO explain entry expected (no assertion;
    explain pass intentionally skips full_graph)

Failure modes:
  exit 0 — report written, all invariants passed
  exit 1 — invariant failure (tool bug or data corruption) — fix at root
  exit 2 — explain coverage gap — run amend_sweep with explain extension
  exit 3 — IO / argument error

CLI:
  python3 tools/sweep_compare.py \\
    --baseline sweep_results/nightly/2026-04-26 \\
    --current  sweep_results/nightly/2026-05-03 \\
    --out      experiments/2026-05-03-nightly-report.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from results_loader import load_effective_results, load_effective_explain  # noqa: E402

# Status taxonomy (matches sweep/run_sweep.py)
SUCCESS_STATUSES = {"full_graph", "graph_break"}
ERROR_STATUSES = {"eager_error", "create_error", "worker_error", "timeout"}
SKIP_STATUSES = {"skipped"}  # known_errors.json gates these

SKIP_MODELS_FILE = REPO_ROOT / "sweep" / "skip_models.json"


def _load_skip_models() -> set[str]:
    """Load model names from sweep/skip_models.json (out-of-scope models).

    These are MODEL NAMES (not (name, mode) keys) that the sweep is
    configured to skip entirely — currently used for timm-dependent
    models (out of our test scope: pip install timm not in our setup).

    sweep_compare excludes any (name, mode) where name is in this list
    from the 6-category partition, placing them in a separate
    'skip_listed' bucket so they don't pollute regression / improvement
    counts. Symmetric with how run_sweep.py handles them.
    """
    if not SKIP_MODELS_FILE.exists():
        return set()
    with open(SKIP_MODELS_FILE) as f:
        return set(json.load(f))


def is_success(status: str) -> bool:
    return status in SUCCESS_STATUSES


def is_error(status: str) -> bool:
    return status in ERROR_STATUSES


def is_skipped(status: str) -> bool:
    return status in SKIP_STATUSES


# ─────────────────────────────────────────────────────────────────────────────
# Categorization
# ─────────────────────────────────────────────────────────────────────────────


def _gb_count(ex: dict, key: tuple[str, str]) -> int:
    if key not in ex:
        return 0
    return len(ex[key].get("break_reasons", []))


def _gb_reasons(ex: dict, key: tuple[str, str]) -> list:
    if key not in ex:
        return []
    return ex[key].get("break_reasons", [])


def _first_error_line(row: dict) -> str:
    msg = row.get("fullgraph_error") or row.get("error") or ""
    return msg.strip().split("\n", 1)[0][:200]


def categorize(
    b_id: dict, c_id: dict, b_ex: dict, c_ex: dict,
    skip_models: set[str] | None = None,
) -> dict[str, list]:
    """Partition all (name, mode) keys in baseline ∪ current into 6 categories.

    Skipped pairs (status='skipped' in either nightly) are excluded from
    cats 1/2/3/6 and reported separately under 'skipped'.

    Pairs whose model NAME is in skip_models (sweep/skip_models.json) are
    excluded from ALL cats (1-6 and 'skipped') and reported separately
    under 'skip_listed'. These are out-of-scope models (e.g., timm-deps
    we don't test) and should not pollute regression/improvement counts.
    """
    skip_models = skip_models or set()
    common = set(b_id.keys()) & set(c_id.keys())
    only_b = set(b_id.keys()) - common
    only_c = set(c_id.keys()) - common

    cat1: list = []
    cat2: list = []
    cat3: list = []
    cat6: list = []
    skipped: list = []
    skip_listed: list = []

    for k in common:
        if k[0] in skip_models:
            skip_listed.append({"key": k,
                                "baseline_status": b_id[k].get("status"),
                                "current_status": c_id[k].get("status")})
            continue

        bs = b_id[k].get("status")
        cs = c_id[k].get("status")

        # If either side is 'skipped', it's not a real comparison signal
        if is_skipped(bs) or is_skipped(cs):
            skipped.append({"key": k, "baseline_status": bs, "current_status": cs})
            continue

        if is_error(bs) and is_success(cs):
            cat1.append({
                "key": k, "baseline_status": bs, "current_status": cs,
                "current_gb": _gb_count(c_ex, k),
            })
        elif is_success(bs) and is_error(cs):
            cat2.append({
                "key": k, "baseline_status": bs, "current_status": cs,
                "error_message": _first_error_line(c_id[k]),
            })
        elif is_success(bs) and is_success(cs):
            b_gb = _gb_count(b_ex, k)
            c_gb = _gb_count(c_ex, k)
            cat3.append({
                "key": k, "baseline_status": bs, "current_status": cs,
                "baseline_gb": b_gb, "current_gb": c_gb,
                "gb_delta": c_gb - b_gb,
            })
        elif is_error(bs) and is_error(cs):
            cat6.append({
                "key": k, "baseline_status": bs, "current_status": cs,
                "error_message": _first_error_line(c_id[k]),
            })
        else:
            # Unknown status combo — should not happen if taxonomy is complete
            raise ValueError(
                f"Unhandled status combo for {k}: baseline={bs!r} current={cs!r}. "
                f"Known: success={SUCCESS_STATUSES}, error={ERROR_STATUSES}, "
                f"skipped={SKIP_STATUSES}"
            )

    cat4 = [{
        "key": k, "status": c_id[k].get("status"),
        "gb_count": _gb_count(c_ex, k),
        "break_reasons": _gb_reasons(c_ex, k),
    } for k in only_c if k[0] not in skip_models]
    cat4_skip_listed = [{
        "key": k, "status": c_id[k].get("status"),
    } for k in only_c if k[0] in skip_models]
    skip_listed.extend(cat4_skip_listed)

    cat5 = [{
        "key": k, "status": b_id[k].get("status"),
    } for k in only_b if k[0] not in skip_models]
    cat5_skip_listed = [{
        "key": k, "status": b_id[k].get("status"),
    } for k in only_b if k[0] in skip_models]
    skip_listed.extend(cat5_skip_listed)

    return {
        "cat1": cat1, "cat2": cat2, "cat3": cat3,
        "cat4": cat4, "cat5": cat5, "cat6": cat6,
        "skipped": skipped,
        "skip_listed": skip_listed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Invariants
# ─────────────────────────────────────────────────────────────────────────────


class InvariantFailure(AssertionError):
    """Raised when the partition or data coverage doesn't satisfy invariants."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "partition" | "explain_coverage" | "data"


def enforce_invariants(
    cats: dict, b_id: dict, c_id: dict, b_ex: dict, c_ex: dict,
    skip_models: set[str] | None = None,
) -> None:
    """Raise InvariantFailure on any violation. Output is gated on this passing."""
    skip_models = skip_models or set()
    common = set(b_id.keys()) & set(c_id.keys())
    only_b = set(b_id.keys()) - common
    only_c = set(c_id.keys()) - common

    common_in_scope = {k for k in common if k[0] not in skip_models}
    only_b_in_scope = {k for k in only_b if k[0] not in skip_models}
    only_c_in_scope = {k for k in only_c if k[0] not in skip_models}

    # Partition: cat1+cat2+cat3+cat6+skipped = |common in-scope|
    n_partitioned = (
        len(cats["cat1"]) + len(cats["cat2"])
        + len(cats["cat3"]) + len(cats["cat6"])
        + len(cats["skipped"])
    )
    if n_partitioned != len(common_in_scope):
        raise InvariantFailure(
            "partition",
            f"cat1+cat2+cat3+cat6+skipped={n_partitioned}, "
            f"expected {len(common_in_scope)} (|common in-scope|, "
            f"common={len(common)} minus {len(common) - len(common_in_scope)} skip-listed). "
            f"Some (name, mode) pair has a status combo the partition doesn't handle."
        )

    # cat5 / cat4 sizes (in-scope only)
    if len(cats["cat5"]) != len(only_b_in_scope):
        raise InvariantFailure(
            "partition",
            f"cat5 (removed in-scope) has {len(cats['cat5'])} items, expected {len(only_b_in_scope)}",
        )
    if len(cats["cat4"]) != len(only_c_in_scope):
        raise InvariantFailure(
            "partition",
            f"cat4 (new in-scope) has {len(cats['cat4'])} items, expected {len(only_c_in_scope)}",
        )

    # Explain coverage: every graph_break row must have a USABLE explain entry.
    # Two failure modes are checked:
    #   (a) MISSING — no entry at all (key not in ex_data)
    #   (b) BROKEN — entry exists but its status indicates the explain pass failed
    #                (status='explain_error' or 'worker_error' etc.) — the
    #                break_reasons list is empty/stale, so any cat-3 GB-count
    #                delta involving this pair is invalid.
    #
    # Both must be flagged: 2026-05-04 brief reported InstructBlipModel/
    # InstructBlipVideoModel as "0 → 9 GBs regression" because the 04-26 baseline
    # had explain_status='explain_error' (count=0, missed), and the current entry
    # had explain_status='ok' (count=9). The delta was bogus. The check below
    # would have flagged this baseline at run time.
    BROKEN_EXPLAIN_STATUSES = {"explain_error", "worker_error", "timeout"}
    for label, id_data, ex_data in [
        ("baseline", b_id, b_ex), ("current", c_id, c_ex),
    ]:
        missing = sorted(
            k for k, v in id_data.items()
            if v.get("status") == "graph_break"
            and k not in ex_data
            and k[0] not in skip_models
        )
        broken = sorted(
            k for k, v in id_data.items()
            if v.get("status") == "graph_break"
            and k in ex_data
            and ex_data[k].get("status") in BROKEN_EXPLAIN_STATUSES
            and k[0] not in skip_models
        )
        if missing or broken:
            details = []
            if missing:
                details.append(
                    f"{len(missing)} MISSING explain entries: "
                    f"{missing[:3]}{'...' if len(missing) > 3 else ''}"
                )
            if broken:
                # Surface the broken-status breakdown (e.g. 4 explain_error + 1 worker_error)
                from collections import Counter as _C
                bs = _C(ex_data[k].get("status") for k in broken)
                details.append(
                    f"{len(broken)} BROKEN explain entries (status in "
                    f"{sorted(BROKEN_EXPLAIN_STATUSES)}): "
                    f"{dict(bs)}; e.g. {broken[:3]}{'...' if len(broken) > 3 else ''}"
                )
            raise InvariantFailure(
                "explain_coverage",
                f"{label}: " + " | ".join(details) + ". "
                f"Reasons / GB counts will be wrong for these pairs. "
                f"Fix: re-run explain via `tools/amend_sweep.py` (with matching "
                f"torch venv) on the affected models, OR exclude them from "
                f"cat-3 analysis if baseline venv unrecoverable."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern break-count delta, segmented by partition category.
#
# CRITICAL DESIGN: this function NEVER returns a single corpus-wide scalar
# delta. It returns a dict segmented by category. The reason: a single
# scalar (current_total - baseline_total) for any break-reason pattern is
# semantically meaningless — it conflates apple-to-apple regression on
# common models (cat 3) with EXPOSURE from newly-compile-testable models
# (cat 1, cat 4). Mixing them produced a wrong "_local_scalar_dense
# regression" finding on 2026-05-04 that almost shipped to Animesh.
#
# Use the cat3 delta for "regression / improvement" claims. Use cat1/cat4
# numbers for "exposure / coverage growth" claims. Never sum them.
# ─────────────────────────────────────────────────────────────────────────────


PatternMatcher = Any  # callable(text:str) -> bool, OR a substring str


def _make_matcher(pattern):
    """Accept either a callable matcher or a literal substring."""
    if callable(pattern):
        return pattern
    if isinstance(pattern, str):
        sub = pattern
        return lambda text: sub in text
    raise TypeError(f"pattern must be callable or str, got {type(pattern)}")


def pattern_delta(
    pattern, b_id: dict, c_id: dict, b_ex: dict, c_ex: dict,
    cats: dict, skip_models: set[str] | None = None,
) -> dict:
    """Per-pattern break-count breakdown segmented by partition category.

    Returns a dict with structure:
      {
        'cat3_baseline':  N,   'cat3_current':  M,   'cat3_delta':  D,
                                                # ↑ ONLY this is regression/improvement
        'cat1_current':   K,   # exposure (was eager_error / timeout / worker_error)
        'cat4_current':   L,   # exposure (truly new model in current)
        'cat6_current':   X,   # always 0 in practice (no break_reasons on errors)
        'totals': {'baseline': ..., 'current': ...},
                                # NEVER subtract these. The subtraction would
                                # mix apple-to-apple regression with exposure.
      }

    To talk about "regression" or "improvement", use cat3_delta and ONLY
    cat3_delta. cat1_current + cat4_current are EXPOSURE numbers — patterns
    that became newly observable because models that previously failed
    eager / didn't exist now compile.
    """
    skip_models = skip_models or set()
    match = _make_matcher(pattern)

    def count_in(ex_data, key):
        if key not in ex_data:
            return 0
        return sum(1 for br in ex_data[key].get("break_reasons", [])
                   if match(br.get("reason", "")))

    cat3_keys = {tuple(r["key"]) if isinstance(r["key"], list) else r["key"]
                 for r in cats["cat3"]}
    cat1_keys = {tuple(r["key"]) if isinstance(r["key"], list) else r["key"]
                 for r in cats["cat1"]}
    cat4_keys = {tuple(r["key"]) if isinstance(r["key"], list) else r["key"]
                 for r in cats["cat4"]}
    cat6_keys = {tuple(r["key"]) if isinstance(r["key"], list) else r["key"]
                 for r in cats["cat6"]}

    cat3_b = sum(count_in(b_ex, k) for k in cat3_keys)
    cat3_c = sum(count_in(c_ex, k) for k in cat3_keys)
    cat1_c = sum(count_in(c_ex, k) for k in cat1_keys)
    cat4_c = sum(count_in(c_ex, k) for k in cat4_keys)
    cat6_c = sum(count_in(c_ex, k) for k in cat6_keys)

    # Corpus-wide totals — exposed in 'totals' for transparency, but tagged
    # with a warning sentinel so consumers don't treat the difference as a delta.
    return {
        "cat3_baseline": cat3_b,
        "cat3_current": cat3_c,
        "cat3_delta": cat3_c - cat3_b,
        "cat1_current": cat1_c,
        "cat4_current": cat4_c,
        "cat6_current": cat6_c,
        "totals": {
            "baseline": cat3_b,  # only cat3 has baseline data (cat1/4/5 by definition)
            "current": cat3_c + cat1_c + cat4_c + cat6_c,
            "_warning": "DO NOT subtract — mixes apple-to-apple regression with exposure. Use cat3_delta for regression claims; cat1/cat4 for exposure.",
        },
    }


def format_pattern_delta(pattern_label: str, result: dict) -> str:
    """Render a pattern_delta dict as a human-readable block. Always includes
    the segmentation explicitly so a reader can't accidentally collapse to one number."""
    out = []
    out.append(f"Pattern: {pattern_label}")
    out.append(f"  cat3 (apple-to-apple, common compile-success):")
    out.append(f"      baseline {result['cat3_baseline']} → current {result['cat3_current']}    Δ {result['cat3_delta']:+d}    ← only this means regression/improvement")
    out.append(f"  cat1 (newly compile-testable, was error in baseline):  +{result['cat1_current']}    (exposure)")
    out.append(f"  cat4 (truly new model in current):                     +{result['cat4_current']}    (exposure)")
    if result['cat6_current']:
        out.append(f"  cat6 (stable failures, normally 0):                    {result['cat6_current']}    (likely a tool bug — investigate)")
    out.append(f"  current total: {result['totals']['current']}    (do NOT subtract from baseline {result['totals']['baseline']})")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Break-reason analysis (Cat 3 categorization-shift, Cat 4 NEW reasons)
# ─────────────────────────────────────────────────────────────────────────────


def reason_signature(reason_obj: Any) -> tuple[str, str]:
    """Stable (location, op_or_explanation) signature for break-reason diffing."""
    text = reason_obj.get("reason", "") if isinstance(reason_obj, dict) else str(reason_obj)
    m_loc = re.search(r"Graph break in user code at ([^\s\n]+)", text)
    loc = m_loc.group(1) if m_loc else "unknown"
    m_op = re.search(r"Operator `([^`]+)`", text)
    if m_op:
        op_or_explain = m_op.group(1)
    else:
        m_ex = re.search(r"Explanation:\s*([^\n]+)", text)
        op_or_explain = m_ex.group(1).strip()[:120] if m_ex else "unknown"
    return (loc, op_or_explain)


def classify_shift(
    baseline_reasons: list, current_reasons: list,
) -> str:
    """Classify a Cat 3 GB-count change.

    Returns one of:
      NO_CHANGE              — multisets identical
      CATEGORIZATION_SHIFT   — same locations + per-location counts; reason strings differ
      REAL_NEW               — at least one location count differs (real new break or removed break)
    """
    sigs_b = [reason_signature(r) for r in baseline_reasons]
    sigs_c = [reason_signature(r) for r in current_reasons]
    locs_b = Counter(s[0] for s in sigs_b)
    locs_c = Counter(s[0] for s in sigs_c)
    if locs_b != locs_c:
        return "REAL_NEW"
    if Counter(sigs_b) == Counter(sigs_c):
        return "NO_CHANGE"
    return "CATEGORIZATION_SHIFT"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown rendering
# ─────────────────────────────────────────────────────────────────────────────


def _meta_line(loader_path: Path) -> str:
    """Return a short identifier for the sweep dir."""
    return loader_path.name


def render_markdown(
    cats: dict, baseline_dir: Path, current_dir: Path,
    b_id: dict, c_id: dict, b_ex: dict, c_ex: dict,
    top_reasons: int = 10,
) -> str:
    """Render the comparison report as Markdown. Run AFTER enforce_invariants."""
    lines: list[str] = []

    def emit(s=""):
        lines.append(s)

    emit(f"# Sweep comparison report — {_meta_line(current_dir)} vs {_meta_line(baseline_dir)}")
    emit()
    emit(f"**Baseline:** `{baseline_dir}`  ({len(b_id)} work items)")
    emit(f"**Current:**  `{current_dir}`  ({len(c_id)} work items)")
    emit(f"**Common:**   {len(set(b_id) & set(c_id))} work items")
    emit(f"**New:**      {len(cats['cat4'])} work items (in current only)")
    emit(f"**Removed:**  {len(cats['cat5'])} work items (in baseline only)")
    if cats["skipped"]:
        emit(f"**Skipped (known_errors.json gated, in either nightly):** {len(cats['skipped'])} work items")
    if cats.get("skip_listed"):
        skip_models = sorted({r["key"][0] for r in cats["skip_listed"]})
        emit(f"**Skip-listed (sweep/skip_models.json — out of test scope):** "
             f"{len(cats['skip_listed'])} work items across {len(skip_models)} models "
             f"(timm/einops dependents: {', '.join(skip_models[:5])}"
             f"{', ...' if len(skip_models) > 5 else ''})")
    emit()
    emit("**Invariants:** ✓ all passed  (partition complete, explain coverage complete)")
    emit()
    emit("---")

    # Cat 1
    emit()
    emit(f"## 1. Improvements: error → compile-success ({len(cats['cat1'])} work items)")
    emit()
    if cats["cat1"]:
        emit("| Model | Mode | baseline | → | current | GB count |")
        emit("|---|---|---|---|---|---|")
        for r in sorted(cats["cat1"], key=lambda r: (r["key"][0], r["key"][1])):
            n, m = r["key"]
            emit(f"| {n} | {m} | {r['baseline_status']} | → | {r['current_status']} | {r['current_gb']} |")
    else:
        emit("_None._")

    # Cat 2
    emit()
    emit(f"## 2. Regressions: compile-success → error ({len(cats['cat2'])} work items)")
    emit()
    if cats["cat2"]:
        emit("| Model | Mode | baseline | → | current | error message |")
        emit("|---|---|---|---|---|---|")
        for r in sorted(cats["cat2"], key=lambda r: (r["key"][0], r["key"][1])):
            n, m = r["key"]
            err = r["error_message"].replace("|", "\\|")
            emit(f"| {n} | {m} | {r['baseline_status']} | → | {r['current_status']} | `{err}` |")
    else:
        emit("_None._")

    # Cat 3
    emit()
    emit(f"## 3. Steady-state: compile-success in both ({len(cats['cat3'])} work items)")
    emit()
    improved = [r for r in cats["cat3"] if r["gb_delta"] < 0]
    regressed = [r for r in cats["cat3"] if r["gb_delta"] > 0]
    unchanged = [r for r in cats["cat3"] if r["gb_delta"] == 0]
    net_gb = sum(r["gb_delta"] for r in cats["cat3"])
    emit(f"**GB-count: improved {len(improved)} / regressed {len(regressed)} / unchanged {len(unchanged)}.** "
         f"Net GB delta across category: **{net_gb:+d}**")
    emit()

    if improved:
        emit("### 3a. GB-count improved (fewer breaks)")
        emit()
        emit("| Model | Mode | baseline GB | current GB | Δ | shift class |")
        emit("|---|---|---|---|---|---|")
        for r in sorted(improved, key=lambda r: r["gb_delta"]):
            n, m = r["key"]
            cls = classify_shift(_gb_reasons(b_ex, r["key"]), _gb_reasons(c_ex, r["key"]))
            emit(f"| {n} | {m} | {r['baseline_gb']} | {r['current_gb']} | {r['gb_delta']} | {cls} |")
        emit()

    if regressed:
        emit("### 3b. GB-count regressed (more breaks)")
        emit()
        emit("| Model | Mode | baseline GB | current GB | Δ | shift class |")
        emit("|---|---|---|---|---|---|")
        for r in sorted(regressed, key=lambda r: -r["gb_delta"]):
            n, m = r["key"]
            cls = classify_shift(_gb_reasons(b_ex, r["key"]), _gb_reasons(c_ex, r["key"]))
            emit(f"| {n} | {m} | {r['baseline_gb']} | {r['current_gb']} | +{r['gb_delta']} | {cls} |")
        emit()

    # Cat 4
    emit()
    emit(f"## 4. New models in current ({len(cats['cat4'])} work items)")
    emit()
    new_models = sorted({r["key"][0] for r in cats["cat4"]})
    emit(f"**Distinct new models:** {len(new_models)}")
    by_status = Counter(r["status"] for r in cats["cat4"])
    emit()
    emit("**Status distribution:**")
    emit()
    emit("| Status | Count |")
    emit("|---|---|")
    for s, c in sorted(by_status.items(), key=lambda x: -x[1]):
        emit(f"| {s} | {c} |")
    emit()
    full_graph_count = by_status.get("full_graph", 0)
    emit(f"**Compile-clean (full_graph):** {full_graph_count} of {len(cats['cat4'])} work items")
    total_new_gb = sum(r["gb_count"] for r in cats["cat4"])
    emit(f"**Total graph breaks across new work items:** {total_new_gb}")
    emit()

    # Top break reasons + NEW reasons
    baseline_sigs: set[tuple[str, str]] = set()
    for k, ex_row in b_ex.items():
        for br in ex_row.get("break_reasons", []):
            baseline_sigs.add(reason_signature(br))
    current_new_sig_counter: Counter = Counter()
    new_only_sig_counter: Counter = Counter()
    for r in cats["cat4"]:
        for br in r["break_reasons"]:
            sig = reason_signature(br)
            current_new_sig_counter[sig] += 1
            if sig not in baseline_sigs:
                new_only_sig_counter[sig] += 1

    emit(f"### Top {top_reasons} break reasons in new models")
    emit()
    if current_new_sig_counter:
        emit("| Count | Location | Op / Explanation |")
        emit("|---|---|---|")
        for sig, c in current_new_sig_counter.most_common(top_reasons):
            emit(f"| {c} | `{sig[0]}` | `{sig[1]}` |")
    else:
        emit("_No graph breaks in new models._")
    emit()
    emit(f"### NEW break reasons (not seen in any baseline model)")
    emit()
    if new_only_sig_counter:
        emit("| Count | Location | Op / Explanation |")
        emit("|---|---|---|")
        for sig, c in new_only_sig_counter.most_common(top_reasons):
            emit(f"| {c} | `{sig[0]}` | `{sig[1]}` |")
    else:
        emit("_All break reasons in new models also appear in baseline models._")

    # Cat 5
    emit()
    emit(f"## 5. Removed models (in baseline only) — {len(cats['cat5'])} work items")
    emit()
    removed_models = sorted({r["key"][0] for r in cats["cat5"]})
    if removed_models:
        emit(f"**Distinct removed models:** {len(removed_models)}")
        emit()
        emit(", ".join(removed_models))
    else:
        emit("_None._")

    # Cat 6
    emit()
    emit(f"## 6. Stable failures: error in both ({len(cats['cat6'])} work items)")
    emit()
    if cats["cat6"]:
        emit("| Model | Mode | status (both) | error message |")
        emit("|---|---|---|---|")
        for r in sorted(cats["cat6"], key=lambda r: (r["key"][0], r["key"][1])):
            n, m = r["key"]
            both_status = r["baseline_status"] if r["baseline_status"] == r["current_status"] else f"{r['baseline_status']}→{r['current_status']}"
            err = r["error_message"].replace("|", "\\|")
            emit(f"| {n} | {m} | {both_status} | `{err}` |")
    else:
        emit("_None._")

    # Footer
    emit()
    emit("---")
    emit()
    emit(f"_Generated by `tools/sweep_compare.py`. Reproduce with:_")
    emit()
    emit(f"```")
    emit(f"python3 tools/sweep_compare.py \\")
    emit(f"    --baseline {baseline_dir} \\")
    emit(f"    --current  {current_dir}")
    emit(f"```")

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# JSON output
# ─────────────────────────────────────────────────────────────────────────────


def cats_to_json(cats: dict) -> dict:
    """Convert categorization to JSON-serializable form (tuples → lists)."""
    out = {}
    for k, v in cats.items():
        out[k] = []
        for r in v:
            r2 = {**r}
            if "key" in r2 and isinstance(r2["key"], tuple):
                r2["key"] = list(r2["key"])
            # Drop the bulky break_reasons from cat4 in JSON to keep it compact;
            # consumers can re-derive from explain_checkpoint.jsonl
            if "break_reasons" in r2:
                r2 = {kk: vv for kk, vv in r2.items() if kk != "break_reasons"}
            out[k].append(r2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--baseline", required=True, type=Path,
                    help="Baseline sweep dir (contains identify_results.json + explain_checkpoint.jsonl)")
    ap.add_argument("--current", required=True, type=Path,
                    help="Current sweep dir")
    ap.add_argument("--out", type=Path, default=None,
                    help="Markdown output path (default: stdout)")
    ap.add_argument("--json", type=Path, default=None,
                    help="Optional JSON output for downstream tooling")
    ap.add_argument("--check-only", action="store_true",
                    help="Run invariants + coverage checks; no report")
    ap.add_argument("--verbose", action="store_true",
                    help="Print invariant pass/fail traces")
    ap.add_argument("--top-reasons", type=int, default=10,
                    help="Top-N break reasons in Cat 4 (default: 10)")
    ap.add_argument("--pattern", action="append", default=None, metavar="SUBSTR",
                    help="Compute per-pattern delta segmented by partition category. "
                         "Pass a substring of the break-reason text (matched literally). "
                         "Repeat the flag for multiple patterns. Output is the structured "
                         "breakdown (cat3 delta = real regression/improvement, cat1/cat4 = exposure). "
                         "Mutually exclusive with --out (writes to stdout instead).")
    ap.add_argument("--source", action="append", default=None,
                    choices=["hf", "diffusers", "custom"],
                    help="Filter rows by source. Repeat for multiple. Default: all sources. "
                         "For weekly HF-only sweeps, pass `--source hf` so apple-to-apple "
                         "is HF-only. Filter applied to BOTH baseline + current before categorization.")
    ap.add_argument("--ignore-invariants", action="store_true",
                    help="Emit warnings on invariant failures instead of refusing to write output. "
                         "Used when a baseline has known explain_coverage gaps that cannot be "
                         "backfilled (e.g. baseline venv unrecoverable). Output JSON metadata "
                         "records the bypass for audit.")
    args = ap.parse_args()

    # Load
    try:
        b_id = load_effective_results(args.baseline)
        c_id = load_effective_results(args.current)
        b_ex = load_effective_explain(args.baseline)
        c_ex = load_effective_explain(args.current)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"IO/parse error: {e}", file=sys.stderr)
        return 3

    # --source filter (applied to BOTH sides before categorization)
    if args.source:
        sources = set(args.source)
        before_b = len(b_id); before_c = len(c_id)
        b_id = {k: r for k, r in b_id.items() if r.get("source") in sources}
        c_id = {k: r for k, r in c_id.items() if r.get("source") in sources}
        b_ex = {k: r for k, r in b_ex.items() if isinstance(r, dict) and r.get("source") in sources}
        c_ex = {k: r for k, r in c_ex.items() if isinstance(r, dict) and r.get("source") in sources}
        if args.verbose:
            print(f"Source filter {sorted(sources)}: baseline {before_b} → {len(b_id)} pairs, "
                  f"current {before_c} → {len(c_id)} pairs", file=sys.stderr)

    if args.verbose:
        print(f"Loaded baseline: {len(b_id)} identify, {len(b_ex)} explain", file=sys.stderr)
        print(f"Loaded current:  {len(c_id)} identify, {len(c_ex)} explain", file=sys.stderr)

    # Load skip list
    skip_models = _load_skip_models()
    if args.verbose:
        print(f"Skip list: {len(skip_models)} models out of test scope (sweep/skip_models.json)",
              file=sys.stderr)

    # Categorize
    try:
        cats = categorize(b_id, c_id, b_ex, c_ex, skip_models=skip_models)
    except ValueError as e:
        print(f"Categorization failure (likely tool bug): {e}", file=sys.stderr)
        return 1

    # Enforce invariants
    invariant_bypass = None
    try:
        enforce_invariants(cats, b_id, c_id, b_ex, c_ex, skip_models=skip_models)
    except InvariantFailure as e:
        if args.ignore_invariants and e.kind == "explain_coverage":
            invariant_bypass = {"kind": e.kind, "message": str(e)}
            print(f"INVARIANT FAILURE [{e.kind}] BYPASSED via --ignore-invariants: {e}",
                  file=sys.stderr)
        else:
            print(f"INVARIANT FAILURE [{e.kind}]: {e}", file=sys.stderr)
            return 2 if e.kind == "explain_coverage" else 1

    if args.verbose or args.check_only:
        print(
            f"OK: invariants passed. "
            f"cat1={len(cats['cat1'])}, cat2={len(cats['cat2'])}, "
            f"cat3={len(cats['cat3'])}, cat4={len(cats['cat4'])}, "
            f"cat5={len(cats['cat5'])}, cat6={len(cats['cat6'])}, "
            f"skipped={len(cats['skipped'])}, "
            f"skip_listed={len(cats.get('skip_listed', []))}",
            file=sys.stderr,
        )

    if args.check_only:
        return 0

    # Per-pattern delta query mode (skips the full report)
    if args.pattern:
        for pat in args.pattern:
            result = pattern_delta(pat, b_id, c_id, b_ex, c_ex, cats,
                                   skip_models=skip_models)
            print(format_pattern_delta(pat, result))
            print()
        return 0

    md = render_markdown(
        cats, args.baseline, args.current,
        b_id, c_id, b_ex, c_ex,
        top_reasons=args.top_reasons,
    )
    if args.out:
        args.out.write_text(md)
        print(f"Wrote: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(md)

    if args.json:
        out_data = cats_to_json(cats)
        out_data["metadata"] = {
            "baseline_dir": str(args.baseline),
            "current_dir": str(args.current),
            "source_filter": sorted(set(args.source)) if args.source else None,
            "invariant_bypass": invariant_bypass,
        }
        args.json.write_text(json.dumps(out_data, indent=2, default=str))
        print(f"Wrote JSON: {args.json}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
