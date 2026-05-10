#!/usr/bin/env python3
"""audit_new_errors.py — Step 2a of the weekly sweep workflow.

For each ERROR row in current sweep that's NEW this week (not in baseline at
the same status, not covered by known_errors.json, not in skip_models.json),
classify via heuristic and emit a markdown report + JSON sidecar.

Surfaces only — never auto-writes config files.

Design: sweep/AUDIT_NEW_ERRORS_DESIGN.md (rev 2).
Adversary review: subagents/adversary-review/invocations/adv-2026-05-10-145000-audit-new-errors-design.md

Usage:
    python3 tools/audit_new_errors.py <sweep_dir>

Exit codes:
    0 — report written
    1 — input parse error or no identify_results.json
    2 — known_errors.json entry missing applies_to_versions (fail-loud)
    3 — compare-vs-baseline.json absent (degraded mode; report still written, tagged PARTIAL)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

from results_loader import load_effective_results  # noqa: E402

ERROR_STATUSES = {"eager_error", "create_error", "worker_error", "timeout"}
KNOWN_ERRORS_FILE = REPO_ROOT / "sweep" / "known_errors.json"
SKIP_MODELS_FILE = REPO_ROOT / "sweep" / "skip_models.json"


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic table — rules evaluated top-to-bottom; first match wins.
# Each rule returns (triage_class, suggested_action) or None on no match.
# case_id attribution per rule (initial set: adv-2026-05-10-145000).
# ─────────────────────────────────────────────────────────────────────────────


def _err(row: dict) -> str:
    return (row.get("error") or "").strip()


def _rule_venv_bootstrap_broken(row: dict):
    if row.get("status") != "worker_error":
        return None
    err = _err(row)
    patterns = ("Unable to load", "Cannot load symbol", "libcudnn",
                "403 Forbidden", "has not been allowlisted")
    if any(p in err for p in patterns):
        return ("venv-bootstrap-broken",
                "STOP triage — fix venv first. See ~/.myclaw-shared/recipes/python-venv-bpf.md.")
    return None


def _rule_gpu_contention(row: dict):
    if row.get("error_type") == "OutOfMemoryError":
        return ("gpu-contention",
                "Wait for auto-retry serial pass; if still OOMs, propose tier upgrade.")
    if row.get("status") == "worker_error" and row.get("returncode") == -9:
        return ("gpu-contention",
                "Wait for auto-retry serial pass; SIGKILL likely OOM-killer.")
    return None


def _rule_cuda_context_pollution(row: dict):
    if row.get("error_type") == "AcceleratorError":
        return ("cuda-context-pollution",
                "Wait for auto-retry; if still fails serially, file as upstream bug.")
    if "device-side assert" in _err(row):
        return ("cuda-context-pollution",
                "Wait for auto-retry; if still fails serially, file as upstream bug.")
    return None


def _rule_subprocess_crash(row: dict):
    if row.get("status") == "worker_error" and row.get("returncode") in {-6, -11}:
        signal = "SIGABRT" if row.get("returncode") == -6 else "SIGSEGV"
        action = (f"Wait for auto-retry. If retry_note=='confirmed_error', file as upstream bug. "
                  f"({signal} suggests assert/segfault.)")
        return ("subprocess-crash", action)
    return None


_FIXTURE_BUG_SUBSTRINGS = (
    "Audio must be mono",
    "Image features and image tokens do not match",
    "expected sequence of length",
    "Sizes of tensors must match",
)


def _rule_fixture_bug(row: dict):
    if row.get("error_type") != "RuntimeError":
        return None
    if row.get("phase") != "eager":
        return None
    err = _err(row)
    if any(s in err for s in _FIXTURE_BUG_SUBSTRINGS):
        return ("fixture-bug",
                "Fix sweep/worker.py input synthesis. PR-blocker — must land before Step 2c.")
    return None


def _rule_tier_upgrade(row: dict):
    if row.get("status") != "timeout":
        return None
    pat = row.get("phase_at_timeout")
    tier = "very_large" if pat == "create" else "large"
    return ("tier-upgrade",
            f"Propose large_models.json entry with timeout_tier={tier!r}; "
            f"defer actual proposal to audit_new_models.py.")


def _rule_upstream_bug(row: dict):
    if row.get("error_type") == "AttributeError":
        return ("upstream-bug",
                "Propose new GitHub issue via subagents/file-issue (Step 2c).")
    err = _err(row)
    # transformers/torch API references in the error text
    if re.search(r"\btorch\.\w+", err) or "transformers" in err.lower():
        return ("upstream-bug",
                "Propose new GitHub issue via subagents/file-issue (Step 2c).")
    return None


# RULE TABLE — order matters; first match wins.
# Each entry: (rule_function, case_id_attribution)
RULES = [
    (_rule_venv_bootstrap_broken, "adv-2026-05-10-145000"),
    (_rule_gpu_contention,        "adv-2026-05-10-145000"),
    (_rule_cuda_context_pollution,"adv-2026-05-10-145000"),
    (_rule_subprocess_crash,      "adv-2026-05-10-145000"),
    (_rule_fixture_bug,           "adv-2026-05-10-145000"),
    (_rule_tier_upgrade,          "adv-2026-05-10-145000"),
    (_rule_upstream_bug,          "adv-2026-05-10-145000"),
]


def classify(row: dict) -> dict:
    """Apply RULES in order; return dict with triage_class + suggested_action + matched_rule."""
    for rule_fn, case_id in RULES:
        out = rule_fn(row)
        if out is not None:
            triage_class, suggested_action = out
            return {
                "triage_class": triage_class,
                "suggested_action": suggested_action,
                "matched_rule": case_id,
            }
    return {
        "triage_class": "unknown",
        "suggested_action": "Surface to human reviewer for manual triage.",
        "matched_rule": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────


def load_known_errors_filtered(active_torch_major_minor: str | None) -> set:
    """Return set of (model, mode) covered by known_errors.json for active torch.

    Per rev 2: missing applies_to_versions field → exit 2 (fail-loud).
    """
    if not KNOWN_ERRORS_FILE.exists():
        return set()
    with open(KNOWN_ERRORS_FILE) as f:
        data = json.load(f)
    covered = set()
    for entry in data.get("entries", []):
        applies = entry.get("applies_to_versions")
        if applies is None:
            print(
                f"FATAL: known_errors.json entry for {entry.get('model')!r} "
                f"is missing 'applies_to_versions'. Per design rev 2, missing field "
                f"is no longer allowed (was 'discouraged'). Add the version list explicitly.",
                file=sys.stderr,
            )
            sys.exit(2)
        if active_torch_major_minor is None or active_torch_major_minor in applies:
            for mode in entry.get("modes", []):
                covered.add((entry["model"], mode))
    return covered


def load_skip_models() -> set:
    if not SKIP_MODELS_FILE.exists():
        return set()
    with open(SKIP_MODELS_FILE) as f:
        return set(json.load(f))


def load_compare(sweep_dir: Path) -> dict | None:
    p = sweep_dir / "compare-vs-baseline.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_streaming_first_seen(sweep_dir: Path) -> dict:
    """Return {(name, mode): first_error_text} from identify_streaming.jsonl.

    Used for diagnostic context only — NOT for classification (per disposition #10).
    """
    p = sweep_dir / "identify_streaming.jsonl"
    if not p.exists():
        return {}
    out = {}
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_record_type") == "metadata":
                continue
            key = (row.get("name"), row.get("mode"))
            if key in out:
                continue  # first occurrence wins
            err = (row.get("error") or "").strip()
            if err:
                out[key] = err
    return out


def torch_major_minor(sweep_dir: Path) -> str | None:
    """Read sweep_state.json → versions.torch → 'major.minor'."""
    state = sweep_dir / "sweep_state.json"
    if not state.exists():
        return None
    try:
        with state.open() as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    ver = (data.get("versions") or {}).get("torch")
    if not ver:
        return None
    m = re.match(r"(\d+)\.(\d+)", ver)
    return f"{m.group(1)}.{m.group(2)}" if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Audit core
# ─────────────────────────────────────────────────────────────────────────────


def select_candidates(rows: dict, compare: dict | None,
                      known_covered: set, skip_set: set) -> list:
    """Return list of (key, row, baseline_status) for candidates."""
    cat6_keys = set()
    if compare is not None:
        for entry in compare.get("cat6", []):
            k = entry.get("key")
            if isinstance(k, list):
                k = tuple(k)
            cat6_keys.add(k)

    candidates = []
    for key, row in rows.items():
        status = row.get("status")
        if status not in ERROR_STATUSES:
            continue
        name = row.get("name")
        if name in skip_set:
            continue
        if (name, row.get("mode")) in known_covered:
            continue
        if compare is not None and key in cat6_keys:
            # Stable failure — sweep_compare reports it; we don't re-surface.
            continue
        baseline_status = None
        if compare is not None:
            for entry in compare.get("cat2", []):
                k = entry.get("key")
                if isinstance(k, list):
                    k = tuple(k)
                if k == key:
                    baseline_status = entry.get("baseline_status")
                    break
        candidates.append((key, row, baseline_status))
    return candidates


def emit_rerun_marker(sweep_dir: Path, fixture_bug_keys: list) -> bool:
    """Write .audit-rerun-required if fixture-bug candidates exist. Returns True if written."""
    marker = sweep_dir / ".audit-rerun-required"
    if not fixture_bug_keys:
        # Idempotent: also remove a stale marker if present.
        if marker.exists():
            marker.unlink()
        return False
    lines = ["# Models needing re-run after fixture-bug fix; one (model, mode) per line."]
    for name, mode in sorted(fixture_bug_keys):
        lines.append(f"{name}|{mode}")
    marker.write_text("\n".join(lines) + "\n")
    return True


def render_report(sweep_dir: Path, candidates: list, compare: dict | None,
                  known_covered: set, all_error_rows: int,
                  cat6_count: int, rerun_marker_emitted: bool,
                  versions: dict) -> str:
    by_class: dict[str, list] = {}
    for key, row, baseline_status, classification, first_seen in candidates:
        by_class.setdefault(classification["triage_class"], []).append(
            (key, row, baseline_status, classification, first_seen)
        )

    lines = []
    em = lines.append
    sweep_date = sweep_dir.name
    baseline_label = "<none — degraded mode>"
    if compare is not None:
        baseline_label = compare.get("baseline_dir", compare.get("metadata", {}).get("baseline_dir", "<unknown>"))
    em(f"# New errors triage — sweep {sweep_date} vs baseline {baseline_label}")
    em("")
    em(f"torch: {versions.get('torch', '?')}  "
       f"transformers: {versions.get('transformers', '?')}  "
       f"diffusers: {versions.get('diffusers', '?')}")
    em("")
    em("## Summary")
    em(f"- {all_error_rows} total ERROR rows in current sweep (incl. worker_error + timeout)")
    em(f"- {len(known_covered)} (model, mode) pairs covered by known_errors.json for active torch")
    em(f"- {cat6_count} stable failures (cat 6; reported by sweep_compare, not here)")
    em(f"- {len(candidates)} candidates surfaced below")
    em("")

    em("## Candidates (in priority order)")
    em("")
    priority = ["venv-bootstrap-broken", "gpu-contention", "cuda-context-pollution",
                "subprocess-crash", "fixture-bug", "tier-upgrade", "upstream-bug", "unknown"]
    for cls in priority:
        rows = by_class.get(cls, [])
        em(f"### {cls} ({len(rows)})")
        em("")
        if not rows:
            em("_None._")
            em("")
            continue
        em("| Model | Mode | Status | error_type | phase | First-line | Baseline status | Action |")
        em("|---|---|---|---|---|---|---|---|")
        for key, row, baseline_status, classification, _first_seen in rows:
            name, mode = key
            err_first = (row.get("error") or "").split("\n", 1)[0][:120].replace("|", "\\|")
            em(f"| {name} | {mode} | {row.get('status')} | "
               f"{row.get('error_type', '')} | {row.get('phase', '')} | "
               f"`{err_first}` | {baseline_status or '?'} | "
               f"{classification['suggested_action']} |")
        em("")

    em("## Re-run-required marker")
    if rerun_marker_emitted:
        em(f"`{sweep_dir / '.audit-rerun-required'}` was emitted. "
           f"Step 2c tools (file_issues.py, close-mode) must check this marker before running.")
    else:
        em("_No fixture-bug candidates → no marker emitted._")
    em("")

    em("## Action checklist for Peng review")
    em("- [ ] STOP if any `venv-bootstrap-broken` — fix venv before re-running anything")
    em("- [ ] Approve `fixture-bug` fixes; re-run affected models BEFORE Step 2c")
    em("- [ ] Approve `upstream-bug` filings → file-issue subagent")
    em("- [ ] Triage `unknown` candidates manually")
    em("- [ ] Approve `tier-upgrade` proposals → audit_new_models.py picks them up")
    em("")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_audit(sweep_dir: Path) -> int:
    if not (sweep_dir / "identify_results.json").exists():
        print(f"FATAL: {sweep_dir}/identify_results.json not found", file=sys.stderr)
        return 1

    rows = load_effective_results(sweep_dir)
    compare = load_compare(sweep_dir)
    streaming_first_seen = load_streaming_first_seen(sweep_dir)
    skip_set = load_skip_models()
    tmm = torch_major_minor(sweep_dir)
    known_covered = load_known_errors_filtered(tmm)  # may sys.exit(2)

    # Versions for header
    versions = {}
    state = sweep_dir / "sweep_state.json"
    if state.exists():
        try:
            with state.open() as f:
                versions = (json.load(f).get("versions") or {})
        except json.JSONDecodeError:
            pass

    # Total error count BEFORE filtering
    all_error_rows = sum(1 for r in rows.values() if r.get("status") in ERROR_STATUSES)
    cat6_count = len(compare.get("cat6", [])) if compare is not None else 0

    candidates_raw = select_candidates(rows, compare, known_covered, skip_set)

    # Classify each + attach first-seen-error
    candidates = []
    fixture_bug_keys = []
    for key, row, baseline_status in candidates_raw:
        classification = classify(row)
        first_seen = streaming_first_seen.get(key)
        candidates.append((key, row, baseline_status, classification, first_seen))
        if classification["triage_class"] == "fixture-bug":
            fixture_bug_keys.append(key)

    rerun_marker_emitted = emit_rerun_marker(sweep_dir, fixture_bug_keys)

    md = render_report(sweep_dir, candidates, compare, known_covered,
                       all_error_rows, cat6_count, rerun_marker_emitted, versions)
    md_path = sweep_dir / "audit-new-errors.md"
    md_path.write_text(md)

    json_payload = {
        "sweep_date": sweep_dir.name,
        "baseline_date": compare.get("metadata", {}).get("baseline_dir") if compare else None,
        "torch_version": versions.get("torch"),
        "summary": {
            "total_errors": all_error_rows,
            "in_known_errors": len(known_covered),
            "stable_cat6": cat6_count,
            "candidates": len(candidates),
        },
        "candidates": [
            {
                "name": key[0], "mode": key[1], "status": row.get("status"),
                "error_type": row.get("error_type"),
                "phase": row.get("phase"),
                "returncode": row.get("returncode"),
                "retry_note": row.get("retry_note"),
                "error_first_line": (row.get("error") or "").split("\n", 1)[0],
                "triage_class": classification["triage_class"],
                "matched_rule": classification["matched_rule"],
                "suggested_action": classification["suggested_action"],
                "baseline_status": baseline_status,
                "first_seen_error": first_seen,
            }
            for key, row, baseline_status, classification, first_seen in candidates
        ],
        "rerun_marker_emitted": rerun_marker_emitted,
    }
    json_path = sweep_dir / "audit-new-errors.json"
    json_path.write_text(json.dumps(json_payload, indent=2))

    print(f"Wrote: {md_path}")
    print(f"Wrote: {json_path}")
    if rerun_marker_emitted:
        print(f"Wrote: {sweep_dir / '.audit-rerun-required'}")
    print(f"Candidates: {len(candidates)} ({sum(1 for _,_,_,c,_ in candidates if c['triage_class']=='unknown')} unknown)")

    return 0 if compare is not None else 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path)
    args = ap.parse_args()
    return run_audit(args.sweep_dir.resolve())


if __name__ == "__main__":
    sys.exit(main())
