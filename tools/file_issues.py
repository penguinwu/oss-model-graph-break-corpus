#!/usr/bin/env python3
"""Scan corpus for errors and manage GitHub issues.

Scans corpus.json for error categories, groups by root cause, and generates
or files GitHub issues with the correct labels and templates.

Subcommands:
  scan        Scan corpus and show draft issues (no side effects)
  file        Create issues on GitHub (requires token)
  reconcile   Compare sweep results against machine-filed issues.
              Generates a plan with per-model evidence; --apply executes it.

Usage:
  # See what issues would be filed
  python tools/file_issues.py scan

  # File issues for real
  python tools/file_issues.py file

  # Generate a reconcile plan (no side effects)
  python tools/file_issues.py reconcile --results sweep_results/nightly/2026-04-19/identify_results.json

  # Review the plan, then apply it
  python tools/file_issues.py reconcile --apply sweep_results/nightly/2026-04-19/reconcile-plan.json

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
import re
import subprocess
import sys
import time
import urllib.request
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

    lines.append("\n<!-- filed-by: otter/file_issues.py -->")

    return "\n".join(lines)


ISSUE_MARKER = "<!-- filed-by: otter/file_issues.py -->"


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
    stdin_data = None
    if data:
        cmd += " -d @-"
        stdin_data = json.dumps(data)

    result = subprocess.run(
        ["sudo", "bash", "-c", cmd],
        capture_output=True, text=True, timeout=30,
        input=stdin_data,
    )
    if result.returncode != 0:
        print(f"ERROR: GitHub API call failed: {result.stderr}")
        return None

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON response: {result.stdout[:200]}")
        return None


PROXY_URL = "http://localhost:7824/fetch"


def _proxy_api(endpoint, method="GET", body=None):
    """Make a GitHub API call via the local web proxy.

    Unlike _github_api (which uses sudo+fwdproxy), this works under
    the agent:claude_code identity by routing through localhost:7824.
    """
    token = _get_github_token()
    if not token:
        print("ERROR: No GitHub token found in ~/.config/gh/hosts.yml")
        sys.exit(1)

    url = f"https://api.github.com{endpoint}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    payload = {"url": url, "method": method, "headers": headers}
    if body is not None:
        payload["body"] = body  # must be dict, NOT json.dumps(dict)

    req = urllib.request.Request(
        PROXY_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print(f"ERROR: Proxy request failed: {e}")
        return None

    if not resp.get("ok"):
        status = resp.get("status", "?")
        error = resp.get("error", "unknown")
        print(f"ERROR: GitHub API {method} {endpoint} → {status}: {error}")
        return None

    content = resp.get("content", "")
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


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


# ── Reconcile helpers ────────────────────────────────────────────────────

ERROR_STATUSES = frozenset({
    "create_error", "compile_error", "timeout", "eager_error", "worker_error",
})


def load_sweep_results(results_path):
    """Load identify_results.json → {(name, mode): result_dict}."""
    with open(results_path) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    by_key = {}
    for r in results:
        by_key[(r["name"], r.get("mode", "eval"))] = r
    return by_key


def load_sweep_metadata(results_path):
    """Extract sweep metadata (version, date) from results file."""
    with open(results_path) as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    versions = meta.get("versions", {})
    return {
        "pytorch_version": versions.get("torch", meta.get("pytorch_version", "unknown")),
        "timestamp": meta.get("timestamp", "unknown"),
    }


def parse_our_affected_models(body):
    """Parse model names from our own issue body format.

    Only handles the format produced by format_issue_body():
      ## Affected Models
      | Model | Modes |
      |-------|-------|
      | ModelName | eval, train |
    """
    if not body:
        return {}
    models = {}
    in_table = False
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped == "## Affected Models":
            in_table = True
            continue
        if in_table:
            if not stripped:
                if not models:
                    continue
                break
            if stripped.startswith("|") and "---" in stripped:
                continue
            if stripped.startswith("| Model"):
                continue
            if stripped.startswith("|"):
                parts = [p.strip() for p in stripped.split("|")]
                if len(parts) >= 3 and parts[1]:
                    model = parts[1]
                    modes = re.findall(r'\b(eval|train)\b', parts[2])
                    if modes:
                        models[model] = sorted(set(modes))
                    else:
                        models[model] = ["eval", "train"]
            else:
                break
    return models


def fetch_open_issues_proxy():
    """Fetch all open issues via web proxy. Filters out pull requests."""
    issues = []
    page = 1
    while True:
        data = _proxy_api(
            f"/repos/{REPO_SLUG}/issues?state=open&per_page=100&page={page}")
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        for issue in data:
            if "pull_request" not in issue:
                issues.append(issue)
        if len(data) < 100:
            break
        page += 1
    return issues


def is_our_issue(issue):
    """Check if an issue was filed by file_issues.py."""
    body = issue.get("body", "") or ""
    return ISSUE_MARKER in body


def categorize_issue(issue, sweep_results):
    """Determine action for a machine-filed issue: CLOSE, UPDATE, or KEEP."""
    models = parse_our_affected_models(issue.get("body", ""))
    if not models:
        return {"action": "KEEP", "reason": "no models parsed", "issue": issue}

    label_names = {l["name"] for l in issue.get("labels", [])}
    is_graph_break_issue = "graph-break" in label_names
    fixed_for_this = frozenset({"full_graph"}) if is_graph_break_issue else frozenset({"full_graph", "graph_break"})

    evidence = []
    still_broken = []
    not_found = []

    for model_name, modes in models.items():
        for mode in modes:
            key = (model_name, mode)
            if key not in sweep_results:
                not_found.append((model_name, mode))
                continue
            status = sweep_results[key].get("status", "")
            if status in fixed_for_this:
                evidence.append({
                    "model": model_name,
                    "mode": mode,
                    "old_status": "broken",
                    "new_status": status,
                })
            else:
                still_broken.append({
                    "model": model_name,
                    "mode": mode,
                    "status": status,
                })

    old_count = len(models)
    new_broken_models = sorted(set(e["model"] for e in still_broken))
    new_count = len(new_broken_models)

    if not still_broken and evidence:
        action = "CLOSE"
    elif not still_broken and not evidence:
        action = "KEEP"
    elif new_count < old_count:
        action = "UPDATE"
    else:
        action = "KEEP"

    return {
        "action": action,
        "issue": {"number": issue["number"], "title": issue["title"],
                  "url": issue.get("html_url", "")},
        "old_count": old_count,
        "new_count": new_count,
        "evidence": evidence,
        "still_broken": still_broken,
        "not_found": [(m, mode) for m, mode in not_found],
    }


def build_reconcile_plan(results_path, sweep_results, open_issues):
    """Build plan: categorize our issues, find new errors."""
    sweep_meta = load_sweep_metadata(results_path)
    our_issues = [i for i in open_issues if is_our_issue(i)]
    skipped = [i for i in open_issues if not is_our_issue(i)]

    actions = []
    for issue in our_issues:
        result = categorize_issue(issue, sweep_results)
        actions.append(result)

    plan = {
        "sweep": sweep_meta,
        "results_path": str(results_path),
        "our_issues": len(our_issues),
        "skipped_issues": len(skipped),
        "skipped_issue_numbers": [i["number"] for i in skipped],
        "actions": actions,
    }
    return plan


def print_reconcile_plan(plan):
    """Print plan with per-model evidence for closes."""
    sweep = plan["sweep"]
    actions = plan["actions"]
    closes = [a for a in actions if a["action"] == "CLOSE"]
    updates = [a for a in actions if a["action"] == "UPDATE"]
    keeps = [a for a in actions if a["action"] == "KEEP"]

    print(f"\n{'=' * 70}")
    print(f"RECONCILE PLAN")
    print(f"  Sweep: PyTorch {sweep['pytorch_version']} ({sweep['timestamp']})")
    print(f"  Scope: {plan['our_issues']} machine-filed issues "
          f"({plan['skipped_issues']} manually-filed skipped)")
    print(f"{'=' * 70}")

    if closes:
        print(f"\n--- PROPOSE CLOSE ({len(closes)} issues) ---\n")
        for a in sorted(closes, key=lambda x: x["issue"]["number"]):
            iss = a["issue"]
            print(f"  #{iss['number']}: {iss['title']}")
            print(f"  {iss.get('url', '')}")
            print(f"  Evidence ({len(a['evidence'])} model/mode pairs fixed):")
            for e in sorted(a["evidence"], key=lambda x: (x["model"], x["mode"])):
                print(f"    {e['model']:40s} {e['mode']:5s}  "
                      f"{e['old_status']} → {e['new_status']}")
            if a["not_found"]:
                print(f"  Not in sweep ({len(a['not_found'])} — removed/skipped):")
                for m, mode in sorted(a["not_found"]):
                    print(f"    {m:40s} {mode}")
            print()

    if updates:
        print(f"\n--- UPDATE ({len(updates)} issues) ---\n")
        for a in sorted(updates, key=lambda x: x["issue"]["number"]):
            iss = a["issue"]
            print(f"  #{iss['number']}: {iss['title']}")
            print(f"       {a['old_count']} models → {a['new_count']} models")
            if a["evidence"]:
                names = sorted(set(e["model"] for e in a["evidence"]))
                print(f"       Fixed: {', '.join(names[:8])}"
                      f"{'...' if len(names) > 8 else ''}")
            if a["still_broken"]:
                names = sorted(set(e["model"] for e in a["still_broken"]))
                print(f"       Still broken: {', '.join(names[:8])}"
                      f"{'...' if len(names) > 8 else ''}")
            print()

    if keeps:
        print(f"\n--- KEEP ({len(keeps)} issues) ---\n")
        for a in sorted(keeps, key=lambda x: x["issue"]["number"]):
            iss = a["issue"]
            print(f"  #{iss['number']}: {iss['title']}")
        print()

    if plan["skipped_issues"] > 0:
        nums = plan["skipped_issue_numbers"]
        print(f"--- SKIPPED ({plan['skipped_issues']} manually-filed) ---")
        print(f"  Issues: {', '.join(f'#{n}' for n in sorted(nums))}")
        print()

    print(f"{'=' * 70}")
    print(f"SUMMARY: {len(closes)} close, {len(updates)} update, {len(keeps)} keep")
    print(f"{'=' * 70}\n")


def apply_reconcile_plan(plan_path):
    """Execute a previously-generated reconcile plan."""
    with open(plan_path) as f:
        plan = json.load(f)

    sweep = plan["sweep"]
    actions_taken = 0

    for a in plan["actions"]:
        if a["action"] == "CLOSE":
            num = a["issue"]["number"]
            evidence_lines = []
            for e in sorted(a["evidence"], key=lambda x: (x["model"], x["mode"])):
                evidence_lines.append(
                    f"| {e['model']} | {e['mode']} | {e['new_status']} |")
            evidence_table = (
                "| Model | Mode | Status |\n|-------|------|--------|\n"
                + "\n".join(evidence_lines)
            )
            comment = (
                f"**Verified fixed** on PyTorch {sweep['pytorch_version']} "
                f"({sweep['timestamp']}).\n\n"
                f"All tracked models now pass:\n\n{evidence_table}\n\n"
                f"Closing."
            )
            print(f"  Closing #{num}...")
            _proxy_api(f"/repos/{REPO_SLUG}/issues/{num}/comments",
                       method="POST", body={"body": comment})
            _proxy_api(f"/repos/{REPO_SLUG}/issues/{num}",
                       method="PATCH",
                       body={"state": "closed", "state_reason": "completed"})
            actions_taken += 1
            time.sleep(1)

        elif a["action"] == "UPDATE":
            num = a["issue"]["number"]
            fixed_names = sorted(set(e["model"] for e in a["evidence"]))
            broken_names = sorted(set(e["model"] for e in a["still_broken"]))
            comment = (
                f"**Sweep update** — PyTorch {sweep['pytorch_version']} "
                f"({sweep['timestamp']}).\n\n"
                f"{a['old_count']} → {a['new_count']} models.\n\n"
                f"**Fixed ({len(fixed_names)}):** {', '.join(fixed_names)}\n\n"
                f"**Still broken ({len(broken_names)}):** "
                f"{', '.join(broken_names)}"
            )
            print(f"  Updating #{num}...")
            _proxy_api(f"/repos/{REPO_SLUG}/issues/{num}/comments",
                       method="POST", body={"body": comment})
            actions_taken += 1
            time.sleep(1)

    print(f"\n  Done: {actions_taken} actions executed.")


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


def cmd_reconcile(args):
    """Reconcile open issues against sweep results."""
    if not args.results and not args.apply:
        print("ERROR: provide --results to generate a plan, "
              "or --apply to execute one.")
        sys.exit(1)

    if args.apply:
        plan_path = Path(args.apply)
        if not plan_path.exists():
            print(f"ERROR: Plan file not found: {plan_path}")
            sys.exit(1)
        print(f"Applying plan from {plan_path}...\n")
        apply_reconcile_plan(plan_path)
        return

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading sweep results from {results_path}...")
    sweep_results = load_sweep_results(results_path)
    print(f"  {len(sweep_results)} model/mode pairs loaded")

    print("Fetching open issues...")
    open_issues = fetch_open_issues_proxy()
    print(f"  {len(open_issues)} open issues found")

    plan = build_reconcile_plan(results_path, sweep_results, open_issues)
    print_reconcile_plan(plan)

    plan_out = results_path.parent / "reconcile-plan.json"
    with open(plan_out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved to: {plan_out}")
    print(f"Review, then run: python tools/file_issues.py reconcile --apply {plan_out}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scan corpus for errors and manage GitHub issues",
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

    # ── reconcile ──
    sub_recon = subparsers.add_parser(
        "reconcile",
        help="Reconcile open issues against sweep results")
    sub_recon.add_argument(
        "--results", metavar="RESULTS_JSON",
        help="Path to identify_results.json — generates a plan")
    sub_recon.add_argument(
        "--apply", metavar="PLAN_JSON",
        help="Apply a previously-generated reconcile plan")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "file":
        cmd_file(args)
    elif args.command == "reconcile":
        cmd_reconcile(args)


if __name__ == "__main__":
    main()
