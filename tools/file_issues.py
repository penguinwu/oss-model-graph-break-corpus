#!/usr/bin/env python3
"""Scan corpus for errors and file GitHub issues.

Scans corpus.json for error categories, groups by root cause, and generates
or files GitHub issues with the correct labels and templates.

Subcommands:
  scan    Scan corpus and show draft issues (no side effects)
  file    Create issues on GitHub (requires token)

Usage:
  # See what issues would be filed
  python tools/file_issues.py scan

  # File issues for real
  python tools/file_issues.py file

  # Only scan explain_errors
  python tools/file_issues.py scan --category explain_error

  # Validate against nightly before filing
  python tools/file_issues.py scan --validate /path/to/nightly/results.jsonl

  # Show existing issues for context
  python tools/file_issues.py scan --show-existing
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = REPO_ROOT / "corpus" / "corpus.json"
REPO_SLUG = "penguinwu/oss-model-graph-break-corpus"

# ── Label taxonomy ────────────────────────────────────────────────────────
# These rules determine which label a group of errors gets.
# Order matters: first match wins.
LABEL_RULES = {
    "for:dynamo-team": [
        "compile_error", "zombie", "explain_error_dynamo",
    ],
    "corpus-infra": [
        "create_error", "timeout", "explain_error_model",
    ],
}

# ── Error classifiers ─────────────────────────────────────────────────────
# Functions that take a list of (model_name, mode, error_text) and return
# groups of {cause_key: [(model, mode, error_text), ...]}.

def _classify_explain_errors(entries):
    """Group explain_error entries by root cause."""
    groups = defaultdict(list)
    for name, mode, err in entries:
        if "PendingUnbackedSymbolNotFound" in err:
            groups["PendingUnbackedSymbolNotFound"].append((name, mode, err))
        elif "Could not guard on data-dependent expression" in err:
            groups["data-dependent-guard"].append((name, mode, err))
        elif "mark_static_address" in err:
            groups["mark_static_address"].append((name, mode, err))
        elif "fake tensors" in err:
            groups["fake-tensor-failure"].append((name, mode, err))
        elif "Attention mask" in err or "attention mask" in err:
            groups["attention-mask-shape"].append((name, mode, err))
        else:
            groups["other-explain-error"].append((name, mode, err))
    return groups


def _classify_create_errors(entries):
    """Group create_error entries by root cause."""
    groups = defaultdict(list)
    for name, mode, err in entries:
        if "No module named" in err:
            groups["missing-dependency"].append((name, mode, err))
        elif "does not appear to have" in err or "not found" in err.lower():
            groups["model-not-found"].append((name, mode, err))
        else:
            groups["other-create-error"].append((name, mode, err))
    return groups


def _classify_compile_errors(entries):
    """Group compile_error entries."""
    groups = defaultdict(list)
    for name, mode, err in entries:
        groups["compile-error"].append((name, mode, err))
    return groups


def _classify_timeout(entries):
    """Group timeout entries."""
    groups = defaultdict(list)
    for name, mode, err in entries:
        groups["timeout"].append((name, mode, err))
    return groups


def _classify_eager_errors(entries):
    """Group eager_error entries by root cause."""
    groups = defaultdict(list)
    for name, mode, err in entries:
        if "CUDA" in err or "cuda" in err:
            groups["cuda-error"].append((name, mode, err))
        elif "out of memory" in err.lower() or "OOM" in err:
            groups["oom"].append((name, mode, err))
        else:
            groups["other-eager-error"].append((name, mode, err))
    return groups


CLASSIFIERS = {
    "explain_error": _classify_explain_errors,
    "create_error": _classify_create_errors,
    "compile_error": _classify_compile_errors,
    "timeout": _classify_timeout,
    "eager_error": _classify_eager_errors,
}

# ── Label assignment ──────────────────────────────────────────────────────

# Causes that are Dynamo team issues (compiler bugs)
DYNAMO_CAUSES = {
    "PendingUnbackedSymbolNotFound", "data-dependent-guard",
    "mark_static_address", "compile-error", "zombie",
}

# Causes that are our infrastructure issues
INFRA_CAUSES = {
    "missing-dependency", "model-not-found", "other-create-error",
    "timeout", "fake-tensor-failure", "attention-mask-shape",
    "other-explain-error", "cuda-error", "oom", "other-eager-error",
}


def assign_labels(category, cause_key):
    """Assign GitHub labels based on error category and root cause."""
    labels = []

    if cause_key in DYNAMO_CAUSES:
        labels.append("for:dynamo-team")
    else:
        labels.append("corpus-infra")

    if category == "explain_error":
        labels.append("explain-error")

    return labels


def assign_title_prefix(labels):
    """Generate title prefix from labels."""
    prefixes = []
    if "corpus-infra" in labels:
        prefixes.append("[corpus-infra]")
    if "explain-error" in labels:
        prefixes.append("[explain_error]")
    return " ".join(prefixes)


# ── Corpus scanner ────────────────────────────────────────────────────────

def scan_corpus(corpus_path=None, categories=None):
    """Scan corpus for all error categories. Returns dict of category → entries."""
    corpus_path = corpus_path or CORPUS_PATH
    with open(corpus_path) as f:
        corpus = json.load(f)

    all_categories = categories or list(CLASSIFIERS.keys()) + ["zombie"]

    entries_by_category = defaultdict(list)

    for m in corpus["models"]:
        for mode in ("eval", "train"):
            md = m.get(mode, {})
            status = md.get("status", "")

            if status in all_categories:
                err = md.get("error", "")
                entries_by_category[status].append((m["name"], mode, err))

            # explain_error is a separate field, not a status
            if "explain_error" in all_categories and md.get("explain_error"):
                entries_by_category["explain_error"].append(
                    (m["name"], mode, md["explain_error"]))

    return dict(entries_by_category)


def group_by_cause(entries_by_category):
    """Apply classifiers to group entries by root cause.

    Returns list of issue drafts:
    [{"category", "cause", "labels", "models", "sample_error", "model_count"}, ...]
    """
    drafts = []

    for category, entries in entries_by_category.items():
        classifier = CLASSIFIERS.get(category)
        if not classifier:
            # Unclassified category — one big group
            models = sorted(set(name for name, mode, _ in entries))
            sample = entries[0][2][:200] if entries else ""
            labels = ["corpus-infra"]
            drafts.append({
                "category": category,
                "cause": category,
                "labels": labels,
                "models": _models_with_modes(entries),
                "model_count": len(models),
                "sample_error": sample,
            })
            continue

        groups = classifier(entries)
        for cause_key, cause_entries in groups.items():
            models = sorted(set(name for name, mode, _ in cause_entries))
            sample = cause_entries[0][2][:200] if cause_entries else ""
            labels = assign_labels(category, cause_key)

            drafts.append({
                "category": category,
                "cause": cause_key,
                "labels": labels,
                "models": _models_with_modes(cause_entries),
                "model_count": len(models),
                "sample_error": sample,
            })

    return drafts


def _models_with_modes(entries):
    """Deduplicate entries to {model_name: [modes]}."""
    by_model = defaultdict(set)
    for name, mode, _ in entries:
        by_model[name].add(mode)
    return {name: sorted(modes) for name, modes in sorted(by_model.items())}


# ── Nightly validation ────────────────────────────────────────────────────

def validate_against_results(drafts, results_path):
    """Check which issues are already fixed in a validation run.

    Reads results.jsonl and marks drafts as fixed/still-broken.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"WARNING: Validation results not found: {results_path}")
        return drafts

    # Load validation results
    val_results = {}
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                name = r.get("name", r.get("model"))
                mode = r.get("mode", "eval")
                val_results[(name, mode)] = r
            except json.JSONDecodeError:
                continue

    for draft in drafts:
        fixed = []
        still_broken = []
        not_tested = []

        for model_name, modes in draft["models"].items():
            for mode in modes:
                key = (model_name, mode)
                if key not in val_results:
                    not_tested.append((model_name, mode))
                    continue

                val_r = val_results[key]
                val_status = val_r.get("status", "")

                if draft["category"] == "explain_error":
                    # For explain_error: "ok" means the explain pass now works
                    if val_status == "ok":
                        fixed.append((model_name, mode))
                    else:
                        still_broken.append((model_name, mode))
                else:
                    # For other categories: check if status improved
                    if val_status in ("full_graph", "graph_break", "ok"):
                        fixed.append((model_name, mode))
                    else:
                        still_broken.append((model_name, mode))

        draft["validation"] = {
            "fixed": fixed,
            "still_broken": still_broken,
            "not_tested": not_tested,
            "all_fixed": len(still_broken) == 0 and len(not_tested) == 0,
        }

    return drafts


# ── Issue body generation ─────────────────────────────────────────────────

def format_issue_body(draft, corpus_meta):
    """Generate markdown issue body from a draft."""
    category = draft["category"]
    cause = draft["cause"]
    models = draft["models"]
    sample = draft["sample_error"]

    pytorch_ver = corpus_meta.get("pytorch_version", "?")
    transformers_ver = corpus_meta.get("transformers_version", "?")

    lines = ["## Summary\n"]

    lines.append(
        f"{draft['model_count']} models with `{category}` "
        f"(root cause: `{cause}`).\n"
    )
    lines.append(f"**PyTorch:** {pytorch_ver}")
    lines.append(f"**Transformers:** {transformers_ver}")
    lines.append(f"**Impact:** {draft['model_count']} models\n")

    # Affected models table
    lines.append("## Affected Models\n")
    lines.append("| Model | Modes |")
    lines.append("|-------|-------|")
    for model_name, modes in models.items():
        lines.append(f"| {model_name} | {', '.join(modes)} |")
    lines.append("")

    # Sample error
    if sample:
        lines.append("## Error\n")
        lines.append("```")
        lines.append(sample)
        lines.append("```\n")

    # Validation results
    val = draft.get("validation")
    if val:
        lines.append("## Nightly Validation\n")
        if val["all_fixed"]:
            lines.append(
                "**All models pass on nightly.** "
                "This issue is already fixed in development.\n"
            )
        else:
            if val["fixed"]:
                lines.append(
                    f"**Fixed on nightly ({len(val['fixed'])} entries):** "
                    f"{', '.join(n for n, m in val['fixed'][:5])}"
                    f"{'...' if len(val['fixed']) > 5 else ''}\n"
                )
            if val["still_broken"]:
                lines.append(
                    f"**Still broken on nightly ({len(val['still_broken'])} entries):** "
                    f"{', '.join(n for n, m in val['still_broken'][:5])}"
                    f"{'...' if len(val['still_broken']) > 5 else ''}\n"
                )

    return "\n".join(lines)


def format_issue_title(draft):
    """Generate issue title from a draft."""
    prefix = assign_title_prefix(draft["labels"])
    cause = draft["cause"].replace("-", " ").replace("_", " ")
    count = draft["model_count"]
    title = f"{prefix} {cause} ({count} models)".strip()
    return title


# ── GitHub API ────────────────────────────────────────────────────────────

def _get_github_token():
    """Read GitHub token from gh CLI config."""
    gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
    if not gh_config.exists():
        return None
    with open(gh_config) as f:
        for line in f:
            if "oauth_token" in line:
                return line.split(":")[-1].strip()
    return None


def _github_api(method, endpoint, data=None):
    """Make a GitHub API call via sudo+fwdproxy."""
    token = _get_github_token()
    if not token:
        print("ERROR: No GitHub token found in ~/.config/gh/hosts.yml")
        sys.exit(1)

    url = f"https://api.github.com{endpoint}"
    cmd = (
        f"curl -s -x http://fwdproxy:8080 '{url}' "
        f"-X {method} "
        f"-H 'Authorization: Bearer {token}' "
        f"-H 'Accept: application/vnd.github+json'"
    )
    if data:
        cmd += f" -d '{json.dumps(data)}'"

    result = subprocess.run(
        ["sudo", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"ERROR: GitHub API call failed: {result.stderr}")
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON response: {result.stdout[:200]}")
        return None


def fetch_existing_issues():
    """Fetch all open issues from the repo."""
    issues = []
    page = 1
    while True:
        data = _github_api(
            "GET",
            f"/repos/{REPO_SLUG}/issues?state=open&per_page=100&page={page}",
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        issues.extend(data)
        if len(data) < 100:
            break
        page += 1
    return issues


def check_duplicates(drafts, existing_issues):
    """Check which drafts might duplicate existing issues."""
    existing_titles = [i["title"].lower() for i in existing_issues]

    for draft in drafts:
        cause = draft["cause"].lower().replace("-", " ").replace("_", " ")
        category = draft["category"].lower()

        potential_dupes = []
        for i, title in enumerate(existing_titles):
            # Check if the cause keyword appears in any existing title
            if cause in title or category in title:
                potential_dupes.append(existing_issues[i])

        draft["potential_duplicates"] = potential_dupes

    return drafts


def create_issue(title, body, labels):
    """Create a GitHub issue. Returns issue data or None."""
    # Ensure labels exist
    for label in labels:
        _github_api("POST", f"/repos/{REPO_SLUG}/labels",
                    {"name": label, "color": "ededed"})

    data = {
        "title": title,
        "body": body,
        "labels": labels,
    }
    return _github_api("POST", f"/repos/{REPO_SLUG}/issues", data)


# ── CLI ───────────────────────────────────────────────────────────────────

def cmd_scan(args):
    """Scan corpus and display draft issues."""
    categories = [args.category] if args.category else None
    entries = scan_corpus(categories=categories)

    if not entries:
        print("No errors found in corpus.")
        return

    drafts = group_by_cause(entries)

    # Validation
    if args.validate:
        print(f"Validating against: {args.validate}")
        drafts = validate_against_results(drafts, args.validate)

    # Dedup check
    if args.show_existing:
        print("Fetching existing issues...")
        existing = fetch_existing_issues()
        drafts = check_duplicates(drafts, existing)
        print(f"Found {len(existing)} open issues\n")

    # Load corpus metadata for body generation
    with open(CORPUS_PATH) as f:
        corpus_meta = json.load(f).get("metadata", {})

    # Display drafts
    print(f"{'=' * 70}")
    print(f"ISSUE SCAN: {len(drafts)} issue groups found")
    print(f"{'=' * 70}\n")

    for i, draft in enumerate(drafts, 1):
        title = format_issue_title(draft)
        labels_str = ", ".join(draft["labels"])

        print(f"--- Draft #{i} ---")
        print(f"  Title:  {title}")
        print(f"  Labels: {labels_str}")
        print(f"  Models: {draft['model_count']}")

        if draft.get("validation"):
            val = draft["validation"]
            if val["all_fixed"]:
                print(f"  Nightly: ✓ ALL FIXED — skip filing")
            else:
                fixed = len(val["fixed"])
                broken = len(val["still_broken"])
                untested = len(val["not_tested"])
                print(f"  Nightly: {fixed} fixed, {broken} still broken, "
                      f"{untested} not tested")

        dupes = draft.get("potential_duplicates", [])
        if dupes:
            print(f"  ⚠ Possible duplicates:")
            for d in dupes[:3]:
                print(f"    #{d['number']}: {d['title']}")

        if args.verbose:
            print(f"  Sample: {draft['sample_error'][:100]}")
            print(f"  Models: {', '.join(list(draft['models'].keys())[:5])}"
                  f"{'...' if draft['model_count'] > 5 else ''}")

        print()

    # Summary
    dynamo = [d for d in drafts if "for:dynamo-team" in d["labels"]]
    infra = [d for d in drafts if "corpus-infra" in d["labels"]]
    validated_fixed = [d for d in drafts
                       if d.get("validation", {}).get("all_fixed")]

    print(f"Summary: {len(dynamo)} for:dynamo-team, {len(infra)} corpus-infra")
    if validated_fixed:
        print(f"  {len(validated_fixed)} already fixed on nightly (skip filing)")


def cmd_file(args):
    """File issues on GitHub."""
    categories = [args.category] if args.category else None
    entries = scan_corpus(categories=categories)

    if not entries:
        print("No errors found in corpus.")
        return

    drafts = group_by_cause(entries)

    # Validation
    if args.validate:
        print(f"Validating against: {args.validate}")
        drafts = validate_against_results(drafts, args.validate)

    # Dedup check
    print("Fetching existing issues for dedup check...")
    existing = fetch_existing_issues()
    drafts = check_duplicates(drafts, existing)

    # Load corpus metadata
    with open(CORPUS_PATH) as f:
        corpus_meta = json.load(f).get("metadata", {})

    # Filter out validated-fixed issues
    to_file = []
    skipped_fixed = []
    skipped_dupe = []

    for draft in drafts:
        if draft.get("validation", {}).get("all_fixed"):
            skipped_fixed.append(draft)
            continue
        if draft.get("potential_duplicates") and not args.force:
            skipped_dupe.append(draft)
            continue
        to_file.append(draft)

    if skipped_fixed:
        print(f"\nSkipping {len(skipped_fixed)} issues fixed on nightly:")
        for d in skipped_fixed:
            print(f"  - {format_issue_title(d)}")

    if skipped_dupe:
        print(f"\nSkipping {len(skipped_dupe)} potential duplicates "
              f"(use --force to file anyway):")
        for d in skipped_dupe:
            title = format_issue_title(d)
            dupes = d["potential_duplicates"]
            print(f"  - {title}")
            for dup in dupes[:2]:
                print(f"    ↳ existing #{dup['number']}: {dup['title']}")

    if not to_file:
        print("\nNo new issues to file.")
        return

    print(f"\n{'=' * 70}")
    print(f"FILING {len(to_file)} issues:")
    print(f"{'=' * 70}")

    for draft in to_file:
        title = format_issue_title(draft)
        body = format_issue_body(draft, corpus_meta)
        labels = draft["labels"]

        print(f"\n  {title}")
        print(f"  Labels: {', '.join(labels)}")

        if args.dry_run:
            print(f"  [DRY RUN — not filed]")
            continue

        result = create_issue(title, body, labels)
        if result and "number" in result:
            print(f"  → Created #{result['number']}")
        else:
            print(f"  → FAILED: {result}")

    if args.dry_run:
        print(f"\n--- DRY RUN — {len(to_file)} issues would be filed ---")


def main():
    parser = argparse.ArgumentParser(
        description="Scan corpus for errors and file GitHub issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── scan ──
    sub_scan = subparsers.add_parser("scan", help="Scan corpus and show draft issues")
    sub_scan.add_argument("--category",
                          choices=list(CLASSIFIERS.keys()),
                          help="Only scan this error category")
    sub_scan.add_argument("--validate", metavar="RESULTS_JSONL",
                          help="Validate against results from another PyTorch version")
    sub_scan.add_argument("--show-existing", action="store_true",
                          help="Fetch and show existing issues for dedup check")
    sub_scan.add_argument("--verbose", "-v", action="store_true",
                          help="Show sample errors and model names")

    # ── file ──
    sub_file = subparsers.add_parser("file", help="File issues on GitHub")
    sub_file.add_argument("--category",
                          choices=list(CLASSIFIERS.keys()),
                          help="Only file issues for this error category")
    sub_file.add_argument("--validate", metavar="RESULTS_JSONL",
                          help="Validate against nightly results before filing")
    sub_file.add_argument("--dry-run", action="store_true",
                          help="Show what would be filed without creating issues")
    sub_file.add_argument("--force", action="store_true",
                          help="File even if potential duplicates exist")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "file":
        cmd_file(args)


if __name__ == "__main__":
    main()
