#!/usr/bin/env python3
"""Mechanical doc-vs-impl consistency checker.

Encodes a small set of cross-document invariants that adversary-review case
2026-05-07-190947-doc-vs-impl found Otter's audit kept missing: the audit
walks the *touched* files but doesn't grep the rest of docs/+skills/ for
references that should be synced.

Each rule below targets one specific failure mode that has shipped before.
Run before docs commits to catch drift; failure is exit code 1.

Run: python3 tools/check_doc_consistency.py
     python3 tools/check_doc_consistency.py --list
     python3 tools/check_doc_consistency.py --rules cohort_codes,apply_modes
     python3 tools/check_doc_consistency.py --explain   # show rule docs

Requires Python 3.9+ (uses dict[str, list[str]] type hints).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Iterable

if sys.version_info < (3, 9):
    sys.exit("ERROR: check_doc_consistency.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each rule returns a list of violation strings. Empty list = pass.
Violation = str
RuleFn = Callable[[], list[Violation]]


# Helpers ─────────────────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _grep_files(glob: str, pattern: str) -> list[tuple[Path, int, str]]:
    """Return (path, line_no, line_text) for every match across glob's files."""
    rx = re.compile(pattern)
    hits: list[tuple[Path, int, str]] = []
    for path in REPO_ROOT.glob(glob):
        if not path.is_file():
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if rx.search(line):
                hits.append((path, i, line))
    return hits


# Rules ───────────────────────────────────────────────────────────────────────

def rule_cohort_codes() -> list[Violation]:
    """Every doc/skill/README that lists CohortValidationError codes must list
    ALL of them, OR include an explicit "see X for canonical list" pointer.

    Adversary review case 2026-05-07-190947-doc-vs-impl gap #2: 3 docs gave
    3 different counts (6, 8, 9). The canonical list lives in
    sweep/cohort_validator.py (extracted at runtime).
    """
    src = _read(REPO_ROOT / "sweep" / "cohort_validator.py")
    canonical = sorted(set(re.findall(
        r"CohortValidationError\(\s*[\"\']([A-Z_]+)[\"\']", src)))
    if not canonical:
        return ["could not extract canonical cohort code list from "
                "sweep/cohort_validator.py — regex out of date?"]

    # Files in scope: docs/, skills/, sweep/README.md, tools/README.md, README.md
    targets: list[Path] = []
    targets.extend(REPO_ROOT.glob("docs/*.md"))
    targets.extend(REPO_ROOT.glob("skills/*.md"))
    for special in ("README.md", "sweep/README.md", "tools/README.md"):
        p = REPO_ROOT / special
        if p.exists():
            targets.append(p)

    violations: list[Violation] = []
    for f in targets:
        text = _read(f)
        # Codes mentioned in this doc
        mentioned = set(re.findall(
            r"\b(BARE_LIST_REJECTED|EMPTY_SOURCE_VERSIONS|"
            r"PARTIAL_SOURCE_VERSIONS|MISSING_METADATA_KEY|VERSION_MISMATCH|"
            r"STALE_COHORT|INVALID_MODELS_LIST|FILE_NOT_FOUND|INVALID_JSON)\b",
            text))
        if not mentioned:
            continue  # doc doesn't list any — fine
        # Doc lists at least one. Either it lists ALL canonical codes,
        # OR includes an explicit canonical-list pointer.
        canonical_set = set(canonical)
        missing = canonical_set - mentioned
        if not missing:
            continue  # complete list
        # Look for an explicit pointer
        if re.search(
            r"see\s+`?(?:sweep/)?cohort_validator\.py`?|"
            r"see\s+`?(?:sweep/)?README\.md`?\s+for\s+(?:the\s+)?(?:canonical|full)|"
            r"canonical (?:9-code |code )?list",
            text,
            re.IGNORECASE,
        ):
            continue
        violations.append(
            f"{f.relative_to(REPO_ROOT)}: lists {sorted(mentioned)} but "
            f"missing {sorted(missing)} and no canonical-list pointer "
            f"(`sweep/cohort_validator.py` is canonical; pointer phrases: "
            f"\"see sweep/README.md for the canonical list\")"
        )
    return violations


def rule_apply_modes() -> list[Violation]:
    """APPLY-X references in any skill/doc must resolve to a defined mode in
    skills/sweep_sanity_check.md.

    Adversary review case 2026-05-07-190947-doc-vs-impl gap #1: skills/sweep.md
    cited APPLY-A/C/D after sweep_sanity_check.md v3 deleted them.
    """
    sanity = _read(REPO_ROOT / "skills" / "sweep_sanity_check.md")
    # Find currently-defined APPLY-X labels (a header or a definition line)
    defined = set(re.findall(r"\bAPPLY-([A-Z])\b", sanity))

    violations: list[Violation] = []
    for f in list(REPO_ROOT.glob("skills/*.md")) + list(REPO_ROOT.glob("docs/*.md")):
        if f.name == "sweep_sanity_check.md":
            continue  # the source of truth itself
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            for label in re.findall(r"\bAPPLY-([A-Z])\b", line):
                # Allow references inside revision-log entries (which describe
                # historical state) — those lines have a date prefix or live
                # under "## Revision Log".
                if "Revision" in line or re.search(r"\d{4}-\d{2}-\d{2}", line):
                    continue
                if label not in defined:
                    violations.append(
                        f"{f.relative_to(REPO_ROOT)}:{i}: cites APPLY-{label} "
                        f"but skills/sweep_sanity_check.md does not define it "
                        f"(currently defined: {sorted(defined) or 'none'}). "
                        f"v3 mode names are 'Pre-launch sample', 'Mid-sweep peek', "
                        f"'Post-completion'."
                    )
    return violations


def rule_results_jsonl_field_name() -> list[Violation]:
    """results.jsonl examples in docs must use `"name":` not `"model":` for the
    model identifier field — `tools/run_experiment.py run` writes `name`.

    Adversary review case 2026-05-07-190947-doc-vs-impl gap #3.
    """
    violations: list[Violation] = []
    # Look only at docs/ and skills/ — not git history or fixtures
    for f in list(REPO_ROOT.glob("docs/*.md")) + list(REPO_ROOT.glob("skills/*.md")):
        text = f.read_text()
        # Find code blocks (json/jsonl/text); match results.jsonl-shaped JSON lines
        for i, line in enumerate(text.splitlines(), start=1):
            # A results.jsonl row is an actual JSON object — the line looks
            # like `{"key": "val", ...}`. Heuristic: starts with `{` (after
            # optional whitespace) AND ends with `}` AND has all three keys.
            stripped = line.strip()
            if not (stripped.startswith("{") and stripped.endswith("}")):
                continue
            if (re.search(r'"model"\s*:\s*"', line)
                    and '"config"' in line
                    and '"mode"' in line):
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}:{i}: results.jsonl example "
                    f"uses `\"model\":` but `run` subcommand writes `\"name\":`. "
                    f"corpus_filter+from reads r[\"name\"] — using \"model\" "
                    f"silently breaks downstream tooling."
                )
    return violations


def rule_python_bin_precedence() -> list[Violation]:
    """Any doc that describes the python_bin/SWEEP_PYTHON precedence relationship
    must state that SWEEP_PYTHON wins (or not assert precedence at all).

    Adversary review case 2026-05-07-190947-doc-vs-impl gap #4 + gap #7:
    docs and template comment said "SWEEP_PYTHON env var as fallback" — but
    code at run_experiment.py uses env as PRIMARY: settings.python_bin is
    actually the fallback.
    """
    violations: list[Violation] = []
    # Verify the code claim is still true; refuse to enforce a stale rule.
    src = _read(REPO_ROOT / "tools" / "run_experiment.py")
    if not re.search(
        r"os\.environ\.get\(\s*[\"']SWEEP_PYTHON[\"']\s*,\s*"
        r"settings\.get\(\s*[\"']python_bin[\"']",
        src,
    ):
        return ["could not verify the python_bin precedence in "
                "run_experiment.py — code path may have changed; update this "
                "rule to match."]

    # Bad framing: "SWEEP_PYTHON env var as fallback" or
    # "settings.python_bin takes precedence over SWEEP_PYTHON" etc.
    bad_patterns = [
        # "SWEEP_PYTHON ... fallback" (env-as-fallback claim)
        re.compile(r"SWEEP_PYTHON[^.\n]*fallback", re.IGNORECASE),
        # "uses SWEEP_PYTHON env var as fallback"
        re.compile(r"uses\s+`?SWEEP_PYTHON`?", re.IGNORECASE),
    ]
    # Allow if the same line/block clarifies the actual order
    allow_patterns = [
        re.compile(r"SWEEP_PYTHON[^.\n]*(takes precedence|wins)",
                   re.IGNORECASE),
        re.compile(r"resolution order[^.\n]*SWEEP_PYTHON", re.IGNORECASE),
    ]

    targets: list[Path] = []
    targets.extend(REPO_ROOT.glob("docs/*.md"))
    targets.extend(REPO_ROOT.glob("skills/*.md"))
    targets.append(REPO_ROOT / "tools" / "run_experiment.py")

    for f in targets:
        text = f.read_text()
        for i, line in enumerate(text.splitlines(), start=1):
            if any(rx.search(line) for rx in bad_patterns):
                # Check if any allow pattern is in the same line OR within
                # ±2 lines (a clarifying sentence)
                window_lines = text.splitlines()[max(0, i - 3):i + 2]
                window = "\n".join(window_lines)
                if any(rx.search(window) for rx in allow_patterns):
                    continue
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}:{i}: framing implies "
                    f"SWEEP_PYTHON is fallback to settings.python_bin — but "
                    f"code uses SWEEP_PYTHON as PRIMARY (settings.python_bin "
                    f"is the fallback). Use phrasing like \"SWEEP_PYTHON env "
                    f"var takes precedence if set; otherwise settings.python_bin "
                    f"is used\"."
                )
    return violations


def rule_d1_threshold_notation() -> list[Violation]:
    """D1 catastrophic-divergence threshold uses strict > 1e-3 in
    check_cohort_invariants.py. Docs that give the threshold must use
    strict-greater notation, not >= / ≥.

    Adversary review case 2026-05-07-190947-doc-vs-impl gap #6.
    """
    src = _read(REPO_ROOT / "tools" / "check_cohort_invariants.py")
    if not re.search(r"D1_THRESHOLD\s*=\s*1e-?3", src):
        return ["could not verify D1_THRESHOLD in check_cohort_invariants.py — "
                "code may have changed; update this rule to match."]
    if not re.search(r"diff\s*>\s*D1_THRESHOLD", src):
        return ["could not verify strict-> comparison for D1 in "
                "check_cohort_invariants.py — code may have changed."]

    # Look for ≥1e-3 / >=1e-3 patterns in docs/skills near the word D1
    violations: list[Violation] = []
    bad_rx = re.compile(r"D1[^\n]*(≥|>=)\s*1e-?3", re.IGNORECASE)
    for f in list(REPO_ROOT.glob("docs/*.md")) + list(REPO_ROOT.glob("skills/*.md")):
        for i, line in enumerate(f.read_text().splitlines(), start=1):
            if bad_rx.search(line):
                violations.append(
                    f"{f.relative_to(REPO_ROOT)}:{i}: D1 threshold notation "
                    f"uses `≥` or `>=` but code is strict `>`. A row with "
                    f"max_diff == 1e-3 is FLAG (D2), not STRICT_FAIL (D1). "
                    f"Use \"max_diff > 1e-3\" instead."
                )
    return violations


# Registry ────────────────────────────────────────────────────────────────────

def rule_subagent_required_fields() -> list[Violation]:
    """Every subagents/<name>/SKILL.md must have YAML frontmatter with `name:`
    and `description:` fields. Every subagents/<name>/ dir must have an
    `invocations/` subdirectory.

    Adversary review case adv-2026-05-08-153427-file-issue-design suggested-test #1.
    """
    violations: list[Violation] = []
    subagents_dir = REPO_ROOT / "subagents"
    if not subagents_dir.is_dir():
        return []  # subagents not yet introduced
    for sub in subagents_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        skill = sub / "SKILL.md"
        if not skill.is_file():
            violations.append(f"subagents/{sub.name}/: missing SKILL.md")
            continue
        text = skill.read_text()
        if not text.startswith("---\n"):
            violations.append(f"subagents/{sub.name}/SKILL.md: missing YAML frontmatter")
            continue
        end = text.find("\n---\n", 4)
        if end < 0:
            violations.append(f"subagents/{sub.name}/SKILL.md: frontmatter not closed")
            continue
        fm = text[4:end]
        for required in ("name", "description"):
            if not re.search(rf"^{required}:", fm, re.MULTILINE):
                violations.append(
                    f"subagents/{sub.name}/SKILL.md: missing required frontmatter "
                    f"field `{required}:`"
                )
        if not (sub / "invocations").is_dir():
            violations.append(
                f"subagents/{sub.name}/: missing `invocations/` subdir "
                f"(per-case-file convention)"
            )
        persona = sub / "persona.md"
        if not persona.is_file():
            violations.append(f"subagents/{sub.name}/: missing persona.md")
    return violations


def rule_subagent_paths_migrated() -> list[Violation]:
    """No stale `skills/adversary-review/` or `skills/file-issue/` references
    should remain after the 2026-05-08 migration.

    Adversary review case adv-2026-05-08-153427-file-issue-design gap #1
    + suggested-test #2.
    """
    violations: list[Violation] = []
    bad_patterns = [
        re.compile(r"\bskills/adversary-review\b"),
        re.compile(r"\bskills/file-issue\b"),
    ]
    # Walk repo .md and .py files, EXCLUDING:
    # - the migration doc itself (intentional historical refs)
    # - per-case files (they preserve the pre-migration provenance note)
    # - this rule's own source (we mention the patterns to detect them)
    exclude_paths = (
        "subagents/MIGRATION.md",
        "subagents/README.md",  # historical "what moved here" context
        "subagents/adversary-review/invocations/",
        "subagents/adversary-review/RETROSPECTIVE.md",  # historical
        "tools/check_doc_consistency.py",
        "tools/test_file_issues.py",  # docstring references the case_id format
    )
    for ext in ("*.md", "*.py"):
        for path in REPO_ROOT.rglob(ext):
            rel = path.relative_to(REPO_ROOT)
            if any(str(rel).startswith(e) for e in exclude_paths):
                continue
            # Skip generated invocations_log.md (it's regeneratable)
            if rel.name == "invocations_log.md":
                continue
            text = path.read_text(errors="ignore")
            for i, line in enumerate(text.splitlines(), start=1):
                for rx in bad_patterns:
                    if rx.search(line):
                        violations.append(
                            f"{rel}:{i}: stale reference to pre-migration "
                            f"`{rx.pattern}` path. Update to `subagents/...`."
                        )
                        break  # don't double-report same line
    return violations


RULES: dict[str, RuleFn] = {
    "cohort_codes": rule_cohort_codes,
    "apply_modes": rule_apply_modes,
    "results_jsonl_field_name": rule_results_jsonl_field_name,
    "python_bin_precedence": rule_python_bin_precedence,
    "d1_threshold_notation": rule_d1_threshold_notation,
    "subagent_required_fields": rule_subagent_required_fields,
    "subagent_paths_migrated": rule_subagent_paths_migrated,
}


# CLI ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true",
                        help="List available rules and exit")
    parser.add_argument("--explain", action="store_true",
                        help="Show each rule's docstring and exit")
    parser.add_argument("--rules", default=None,
                        help="Comma-separated subset of rules to run "
                             "(default: all)")
    args = parser.parse_args()

    if args.list:
        for name in RULES:
            print(name)
        return 0

    if args.explain:
        for name, fn in RULES.items():
            print(f"### {name}")
            print((fn.__doc__ or "").strip())
            print()
        return 0

    selected = list(RULES.keys())
    if args.rules:
        requested = [r.strip() for r in args.rules.split(",") if r.strip()]
        unknown = [r for r in requested if r not in RULES]
        if unknown:
            print(f"ERROR: unknown rule(s): {unknown}. "
                  f"Available: {list(RULES)}", file=sys.stderr)
            return 2
        selected = requested

    total_violations = 0
    for name in selected:
        violations = RULES[name]()
        if violations:
            print(f"❌ {name}: {len(violations)} violation(s)")
            for v in violations:
                print(f"    {v}")
            total_violations += len(violations)
        else:
            print(f"✅ {name}")

    print()
    if total_violations:
        print(f"FAIL: {total_violations} violation(s) across {len(selected)} rule(s)")
        return 1
    print(f"PASS: all {len(selected)} rule(s) clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
