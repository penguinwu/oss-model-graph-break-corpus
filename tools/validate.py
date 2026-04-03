#!/usr/bin/env python3
"""Corpus validation — checks corpus and tool integrity.

Every change (manual, agent-driven, or sweep-generated) must pass
validation before shipping.

Usage:
    python3 tools/validate.py           # Run all checks
    python3 tools/validate.py --fix     # Auto-fix what it can
    python3 tools/validate.py --verbose # Show passing checks too
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus" / "corpus.json"
GOLDEN_SET_PATH = REPO_ROOT / "corpus" / "golden_set.json"

# Valid statuses per mode
VALID_STATUSES = {"full_graph", "graph_break", "eager_error", "create_error",
                   "compile_error", "explain_error", "timeout", "worker_error"}
# Required top-level fields per model
REQUIRED_MODEL_FIELDS = {"name", "source", "has_graph_break", "eval", "train"}
# Required fields in eval/train when status is present
REQUIRED_MODE_FIELDS = {"status"}
# Fields expected in --json output
EXPECTED_JSON_FIELDS = {"name", "source", "status"}


class ValidationResult:
    """Collects pass/fail results with messages."""

    def __init__(self):
        self.checks = []  # (name, passed, message)
        self.fixes = []   # (description, before, after)

    def check(self, name, passed, message=""):
        self.checks.append((name, passed, message))

    def fix(self, description, before, after):
        self.fixes.append((description, before, after))

    @property
    def passed(self):
        return all(p for _, p, _ in self.checks)

    @property
    def failures(self):
        return [(n, m) for n, p, m in self.checks if not p]

    def report(self, verbose=False):
        lines = []
        for name, passed, message in self.checks:
            if passed and not verbose:
                continue
            mark = "PASS" if passed else "FAIL"
            msg = f"  {mark}: {name}"
            if message:
                msg += f" — {message}"
            lines.append(msg)

        if self.fixes:
            lines.append("")
            lines.append("Fixes applied:")
            for desc, before, after in self.fixes:
                lines.append(f"  - {desc}: {before} → {after}")

        return "\n".join(lines)


def load_corpus():
    """Load corpus.json and return the parsed dict."""
    with open(CORPUS_PATH) as f:
        return json.load(f)


def save_corpus(corpus):
    """Write corpus.json back."""
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2)
        f.write("\n")


# ─── Golden Set Checks ───────────────────────────────────────────────


def check_golden_set(corpus, result):
    """Check golden set models against corpus."""
    if not GOLDEN_SET_PATH.exists():
        result.check("golden_set_exists", False, f"Missing {GOLDEN_SET_PATH}")
        return

    with open(GOLDEN_SET_PATH) as f:
        golden = json.load(f)

    golden_models = golden.get("models", [])
    result.check("golden_set_nonempty", len(golden_models) > 0,
                 f"{len(golden_models)} models in golden set")

    # Index corpus by name
    corpus_index = {m["name"]: m for m in corpus["models"]}

    for gm in golden_models:
        name = gm["name"]
        if name not in corpus_index:
            result.check(f"golden_{name}_exists", False,
                         f"Golden model {name} not found in corpus")
            continue

        cm = corpus_index[name]

        # Check expected statuses
        for mode in ("eval", "train"):
            expected = gm.get(mode, {}).get("status")
            actual = cm.get(mode, {}).get("status")
            if expected and expected != actual:
                result.check(f"golden_{name}_{mode}_status", False,
                             f"Expected {expected}, got {actual}")
            elif expected:
                result.check(f"golden_{name}_{mode}_status", True)

        # Check expected break reasons (if specified)
        for mode in ("eval", "train"):
            expected_reason = gm.get(mode, {}).get("expected_break_reason")
            if not expected_reason:
                continue
            actual_reasons = cm.get(mode, {}).get("break_reasons", [])
            reason_texts = [r.get("reason", "")[:80] if isinstance(r, dict)
                           else str(r)[:80] for r in actual_reasons]
            found = any(expected_reason in rt for rt in reason_texts)
            result.check(f"golden_{name}_{mode}_reason", found,
                         f"Expected reason containing '{expected_reason}'" +
                         ("" if found else f", got {len(reason_texts)} reasons"))


# ─── Structural Checks ───────────────────────────────────────────────


def check_summary_counts(corpus, result, fix=False):
    """Verify summary block matches actual model counts."""
    models = corpus["models"]
    summary = corpus.get("summary", {})

    # Count actual statuses
    for mode in ("eval", "train"):
        actual = Counter(m[mode]["status"] for m in models if mode in m)
        expected = summary.get(mode, {})

        for status in VALID_STATUSES:
            a = actual.get(status, 0)
            e = expected.get(status, 0)
            ok = a == e
            result.check(f"summary_{mode}_{status}", ok,
                         f"summary.{mode}.{status}: expected {e}, actual {a}" if not ok else "")
            if not ok and fix:
                if mode not in summary:
                    summary[mode] = {}
                summary[mode][status] = a
                result.fix(f"summary.{mode}.{status}", e, a)

    # Total model count
    actual_total = len(models)
    expected_total = summary.get("total_models", 0)
    ok = actual_total == expected_total
    result.check("summary_total_models", ok,
                 f"summary.total_models: expected {expected_total}, actual {actual_total}" if not ok else "")
    if not ok and fix:
        summary["total_models"] = actual_total
        result.fix("summary.total_models", expected_total, actual_total)


def check_has_graph_break_flags(corpus, result, fix=False):
    """Verify has_graph_break flag consistency."""
    models = corpus["models"]

    for m in models:
        name = m["name"]
        has_break = m.get("has_graph_break", False)

        # True if any mode has graph_break status
        actual = any(
            m.get(mode, {}).get("status") == "graph_break"
            for mode in ("eval", "train", "dynamic_mark", "dynamic_true")
        ) or any(
            m.get(mode, {}).get("dynamic_mark", {}).get("status") == "graph_break"
            for mode in ("eval", "train")
        ) or any(
            m.get(mode, {}).get("dynamic_true", {}).get("status") == "graph_break"
            for mode in ("eval", "train")
        )

        if has_break != actual:
            result.check(f"has_graph_break_{name}", False,
                         f"has_graph_break={has_break}, but actual={actual}")
            if fix:
                m["has_graph_break"] = actual
                result.fix(f"{name}.has_graph_break", has_break, actual)
        else:
            result.check(f"has_graph_break_{name}", True)


def check_no_duplicate_names(corpus, result):
    """Check for duplicate model names."""
    names = [m["name"] for m in corpus["models"]]
    dupes = [n for n, c in Counter(names).items() if c > 1]
    ok = len(dupes) == 0
    result.check("no_duplicate_names", ok,
                 f"Duplicate names: {dupes}" if not ok else "")


def check_required_fields(corpus, result):
    """Check all required fields present in every model."""
    for m in corpus["models"]:
        name = m.get("name", "<unnamed>")
        missing = REQUIRED_MODEL_FIELDS - set(m.keys())
        ok = len(missing) == 0
        result.check(f"required_fields_{name}", ok,
                     f"Missing: {missing}" if not ok else "")

        # Check mode-level required fields
        for mode in ("eval", "train"):
            if mode in m and isinstance(m[mode], dict):
                mode_missing = REQUIRED_MODE_FIELDS - set(m[mode].keys())
                ok = len(mode_missing) == 0
                result.check(f"required_{mode}_fields_{name}", ok,
                             f"Missing in {mode}: {mode_missing}" if not ok else "")


def check_metadata_versions(corpus, result):
    """Check metadata versions are valid semver-ish strings."""
    metadata = corpus.get("metadata", {})
    semver_re = re.compile(r"^\d+\.\d+\.\d+")

    for key in ("pytorch_version", "transformers_version", "diffusers_version"):
        val = metadata.get(key, "")
        # Strip build metadata like +cu128
        base = val.split("+")[0] if val else ""
        ok = bool(semver_re.match(base)) if base else False
        result.check(f"metadata_{key}", ok,
                     f"{key}='{val}'" + (" (invalid)" if not ok else ""))


def check_error_field_consistency(corpus, result):
    """Check error field consistency for graph_break models."""
    for m in corpus["models"]:
        name = m["name"]
        for mode in ("eval", "train"):
            mode_data = m.get(mode, {})
            status = mode_data.get("status")
            if status == "graph_break":
                has_error = ("fullgraph_error" in mode_data or
                            "error" in mode_data or
                            "break_reasons" in mode_data)
                result.check(f"error_field_{name}_{mode}", has_error,
                             f"graph_break but no error/fullgraph_error/break_reasons" if not has_error else "")


def check_dynamic_results(corpus, result):
    """Check dynamic results present for all models."""
    missing_mark = []
    missing_true = []
    for m in corpus["models"]:
        name = m["name"]
        eval_data = m.get("eval", {})
        if not eval_data.get("dynamic_mark"):
            missing_mark.append(name)
        if not eval_data.get("dynamic_true"):
            missing_true.append(name)

    ok_mark = len(missing_mark) == 0
    ok_true = len(missing_true) == 0
    result.check("dynamic_mark_present", ok_mark,
                 f"{len(missing_mark)} models missing dynamic_mark" if not ok_mark else "")
    result.check("dynamic_true_present", ok_true,
                 f"{len(missing_true)} models missing dynamic_true" if not ok_true else "")


# ─── Tool Output Checks ──────────────────────────────────────────────


def run_tool(cmd, cwd=None):
    """Run a tool and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=cwd or str(REPO_ROOT),
        )
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def check_tool_outputs(result):
    """Check query.py and compare.py tool outputs."""
    python = sys.executable

    # query.py with no args
    rc, out, err = run_tool([python, "tools/query.py"])
    result.check("query_default_exit0", rc == 0,
                 f"exit {rc}: {err[:100]}" if rc != 0 else "")
    result.check("query_default_nonempty", len(out.strip()) > 0,
                 "empty output" if not out.strip() else "")

    # query.py --json
    rc, out, err = run_tool([python, "tools/query.py", "--json"])
    if rc == 0:
        try:
            data = json.loads(out)
            result.check("query_json_valid", True)
            if isinstance(data, list) and len(data) > 0:
                fields = set(data[0].keys())
                missing = EXPECTED_JSON_FIELDS - fields
                result.check("query_json_fields", len(missing) == 0,
                             f"Missing fields: {missing}" if missing else "")
            else:
                result.check("query_json_fields", False, "Empty or non-list JSON")
        except json.JSONDecodeError as e:
            result.check("query_json_valid", False, f"Invalid JSON: {e}")
    else:
        result.check("query_json_valid", False, f"exit {rc}")

    # query.py --status with each valid status
    for status in sorted(VALID_STATUSES):
        rc, out, err = run_tool([python, "tools/query.py", "--status", status])
        result.check(f"query_status_{status}", rc == 0,
                     f"exit {rc}" if rc != 0 else "")

    # compare.py --corpus-dynamic
    rc, out, err = run_tool([python, "tools/compare.py", "--corpus-dynamic"])
    result.check("compare_corpus_dynamic", rc == 0,
                 f"exit {rc}: {err[:100]}" if rc != 0 else "")


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Corpus validation checks")
    parser.add_argument("--fix", action="store_true",
                        help="Auto-fix what it can (summary, has_graph_break)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show passing checks too")
    parser.add_argument("--skip-tools", action="store_true",
                        help="Skip tool output checks (faster)")
    parser.add_argument("--corpus", default=None,
                        help="Path to corpus.json (default: corpus/corpus.json)")
    args = parser.parse_args()

    global CORPUS_PATH
    if args.corpus:
        CORPUS_PATH = Path(args.corpus).resolve()

    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    corpus = load_corpus()
    result = ValidationResult()

    print(f"Validating corpus: {len(corpus['models'])} models")
    print()

    # Golden set
    print("Golden set checks...")
    check_golden_set(corpus, result)

    # Structural checks
    print("Structural checks...")
    check_summary_counts(corpus, result, fix=args.fix)
    check_has_graph_break_flags(corpus, result, fix=args.fix)
    check_no_duplicate_names(corpus, result)
    check_required_fields(corpus, result)
    check_metadata_versions(corpus, result)
    check_error_field_consistency(corpus, result)
    check_dynamic_results(corpus, result)

    # Tool output checks
    if not args.skip_tools:
        print("Tool output checks...")
        check_tool_outputs(result)

    # Apply fixes if requested
    if args.fix and result.fixes:
        save_corpus(corpus)

    # Report
    print()
    print(result.report(verbose=args.verbose))
    print()

    failures = result.failures
    total = len(result.checks)
    passed = total - len(failures)

    if failures:
        print(f"FAILED: {len(failures)}/{total} checks failed")
        sys.exit(1)
    else:
        print(f"OK: {passed}/{total} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
