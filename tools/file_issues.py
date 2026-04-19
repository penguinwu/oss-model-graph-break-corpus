#!/usr/bin/env python3
"""Post-sweep issue management for the OSS model graph-break corpus.

Two commands:
  sweep-report  Analyze sweep results, generate a reviewable update plan (read-only)
  sweep-update  Apply a reviewed plan to update GitHub issues (writes)

Usage:
  # After a sweep, generate the report
  python tools/file_issues.py sweep-report \
    --explain sweep_results/nightly/2026-04-19/explain_results.json \
    --identify sweep_results/nightly/2026-04-19/identify_results.json

  # Review the plan, then apply it
  python tools/file_issues.py sweep-update \
    --plan sweep_results/nightly/2026-04-19/sweep-report.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_SLUG = "penguinwu/oss-model-graph-break-corpus"
PROXY_URL = "http://localhost:7824/fetch"
ISSUE_MARKER = "<!-- filed-by: otter/file_issues.py -->"


# ── Graph-break classifier ──────────────────────────────────────────────
# Maps break_reasons from explain_results.json to issue-level categories.
# Each rule: (key, issue_number, match_fn(explanation, location) → bool)

GRAPH_BREAK_RULES = [
    ("data_dep_shape_ops", 18, lambda exp, loc:
        "aten.nonzero" in exp or "repeat_interleave.Tensor" in exp
        or "masked_select" in exp or "unique_consecutive" in exp
        or "_unique2" in exp),
    ("non_tensor_output", 28, lambda exp, loc:
        "return a non-Tensor" in exp),
    ("local_scalar_dense", 55, lambda exp, loc:
        "_local_scalar_dense" in exp),
    ("logging_logger", 10, lambda exp, loc:
        "logging.Logger" in exp),
    ("proxy_conversion", 8, lambda exp, loc:
        "as_proxy()" in exp),
    ("exception_handler", 12, lambda exp, loc:
        "no exception handler" in exp),
    ("lock_context_mgr", 11, lambda exp, loc:
        "lock" in exp and "context manager" in exp),
    ("contextvar", 23, lambda exp, loc:
        "ContextVar" in exp),
    ("callable_builtin", 20, lambda exp, loc:
        "callable" in exp and "argument types" in exp),
    ("setattr_class", 24, lambda exp, loc:
        "setattr" in exp and "requires_grad" not in exp),
    ("requires_grad", 19, lambda exp, loc:
        "requires_grad" in exp),
    ("function_get", 25, lambda exp, loc:
        "function.__get__" in exp),
    ("builtin_lt", 21, lambda exp, loc:
        "<built-in function lt>" in exp),
    ("rng_seed", 26, lambda exp, loc:
        "Generator.seed()" in exp or "manual_seed" in exp),
    ("uninit_module", 27, lambda exp, loc:
        "uninitialized nn.Module" in exp),
    ("import_config_skip", 5, lambda exp, loc:
        "import_utils.py" in loc or "configuration_utils.py" in loc),
    ("frame_skip", 2, lambda exp, loc:
        "output_capturing.py" in loc or "generic.py" in loc),
    ("find_spec_skip", 7, lambda exp, loc:
        "find_spec" in exp),
    ("tensor_item", 56, lambda exp, loc:
        "Tensor.item()" in exp),
    ("data_dep_branch", 54, lambda exp, loc:
        "data-dependent branching" in exp),
]

MODEL_SPECIFIC_ISSUES = {
    14: lambda name: "OpenVoice" in name,
    15: lambda name: "GPTSoVITS" in name,
    16: lambda name: "MiniCPM" in name,
    17: lambda name: "Gemma4" in name or "Gemma4Text" in name,
}

CORPUS_INFRA_ISSUES = {
    45: "create_error",
    46: "timeout",
    52: "eager_error",
    53: "cuda_error",
}


# ── GitHub API ───────────────────────────────────────────────────────────

def _get_github_token():
    gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
    if not gh_config.exists():
        return None
    with open(gh_config) as f:
        for line in f:
            if "oauth_token" in line:
                return line.split(":")[-1].strip()
    return None


def _proxy_api(endpoint, method="GET", body=None):
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
        payload["body"] = body

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
        print(f"ERROR: GitHub API {method} {endpoint} → "
              f"{resp.get('status', '?')}: {resp.get('error', 'unknown')}")
        return None

    content = resp.get("content", "")
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def fetch_open_issues():
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


# ── Data loading ─────────────────────────────────────────────────────────

def load_explain_data(explain_path):
    with open(explain_path) as f:
        data = json.load(f)
    return data.get("results", [])


def load_identify_data(identify_path):
    with open(identify_path) as f:
        data = json.load(f)
    results = data.get("results", data) if isinstance(data, dict) else data
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    versions = meta.get("versions", {})
    metadata = {
        "pytorch_version": versions.get("torch", meta.get("pytorch_version", "unknown")),
        "transformers_version": versions.get("transformers", meta.get("transformers_version", "unknown")),
        "timestamp": meta.get("timestamp", "unknown"),
    }
    return results, metadata


# ── Classifier ───────────────────────────────────────────────────────────

def classify_breaks(explain_entries):
    """Classify break_reasons into issue categories.

    Returns:
        {rule_key: {"issue_number": N, "models": {name: {"modes": set, "breaks": int}},
                    "total_breaks": int, "sample_reasons": [str]}}
    """
    classified = defaultdict(lambda: {
        "issue_number": None,
        "models": defaultdict(lambda: {"modes": set(), "breaks": 0}),
        "total_breaks": 0,
        "sample_reasons": [],
    })
    unclassified = defaultdict(lambda: {"models": set(), "count": 0})

    for entry in explain_entries:
        if not entry.get("break_reasons"):
            continue
        model = entry["name"]
        mode = entry.get("mode", "eval")

        for br in entry["break_reasons"]:
            reason_text = br.get("reason", "")
            exp_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', reason_text)
            loc_match = re.search(r'at\s+(\S+\.py):\d+', reason_text)
            explanation = exp_match.group(1).strip() if exp_match else reason_text
            location = loc_match.group(1) if loc_match else ""

            matched = False
            for key, issue_num, match_fn in GRAPH_BREAK_RULES:
                if match_fn(explanation, location):
                    bucket = classified[key]
                    bucket["issue_number"] = issue_num
                    bucket["models"][model]["modes"].add(mode)
                    bucket["models"][model]["breaks"] += 1
                    bucket["total_breaks"] += 1
                    if len(bucket["sample_reasons"]) < 3:
                        bucket["sample_reasons"].append(reason_text[:300])
                    matched = True
                    break

            if not matched:
                short = explanation[:80]
                unclassified[short]["models"].add(model)
                unclassified[short]["count"] += 1

        for issue_num, name_fn in MODEL_SPECIFIC_ISSUES.items():
            if name_fn(model):
                key = f"model_specific_{issue_num}"
                classified[key]["issue_number"] = issue_num
                classified[key]["models"][model]["modes"].add(mode)

    return dict(classified), dict(unclassified)


def classify_infra(identify_entries):
    """Classify identify results into corpus-infra categories.

    Returns:
        {status: {"models": {name: {"modes": set, "error": str}}, ...}}
    """
    infra = defaultdict(lambda: {"models": defaultdict(lambda: {"modes": set(), "error": ""})})

    for entry in identify_entries:
        status = entry.get("status", "")
        name = entry["name"]
        mode = entry.get("mode", "eval")
        error = str(entry.get("fullgraph_error", entry.get("error", "")))[:200]

        if status == "create_error":
            infra["create_error"]["models"][name]["modes"].add(mode)
            infra["create_error"]["models"][name]["error"] = error
        elif status == "timeout":
            infra["timeout"]["models"][name]["modes"].add(mode)
        elif status == "eager_error":
            is_cuda = "CUDA error" in error or "cudaError" in error
            cat = "cuda_error" if is_cuda else "eager_error"
            infra[cat]["models"][name]["modes"].add(mode)
            infra[cat]["models"][name]["error"] = error

    return dict(infra)


# ── Leverage analysis ────────────────────────────────────────────────────

def compute_leverage(explain_entries):
    """For each issue, count models that would become fullgraph if fixed.

    A model becomes fullgraph from fixing issue X if X is its ONLY issue.
    """
    model_issues = defaultdict(lambda: defaultdict(set))

    for entry in explain_entries:
        if not entry.get("break_reasons"):
            continue
        model = entry["name"]
        mode = entry.get("mode", "eval")

        for br in entry["break_reasons"]:
            reason_text = br.get("reason", "")
            exp_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', reason_text)
            loc_match = re.search(r'at\s+(\S+\.py):\d+', reason_text)
            explanation = exp_match.group(1).strip() if exp_match else reason_text
            location = loc_match.group(1) if loc_match else ""

            for key, issue_num, match_fn in GRAPH_BREAK_RULES:
                if issue_num is not None and match_fn(explanation, location):
                    model_issues[model][mode].add(issue_num)
                    break

    leverage = defaultdict(set)
    for model, modes in model_issues.items():
        for mode, issues in modes.items():
            if len(issues) == 1:
                leverage[list(issues)[0]].add(model)

    return {num: sorted(models) for num, models in
            sorted(leverage.items(), key=lambda x: -len(x[1]))}


# ── Cross-references ─────────────────────────────────────────────────────

def compute_cross_refs(classified):
    """Find which issues share models."""
    issue_models = {}
    for key, data in classified.items():
        num = data["issue_number"]
        if num is not None:
            issue_models[num] = set(data["models"].keys())

    cross_refs = defaultdict(set)
    issues = list(issue_models.keys())
    for i, num_a in enumerate(issues):
        for num_b in issues[i + 1:]:
            overlap = issue_models[num_a] & issue_models[num_b]
            if overlap:
                cross_refs[num_a].add(num_b)
                cross_refs[num_b].add(num_a)

    return {num: sorted(refs) for num, refs in cross_refs.items()}


# ── Body generation ──────────────────────────────────────────────────────

def generate_dynamo_body(rule_key, data, metadata, leverage, cross_refs):
    """Generate issue body for a dynamo graph-break issue."""
    issue_num = data["issue_number"]
    models = data["models"]
    total_breaks = data["total_breaks"]
    pt_ver = metadata["pytorch_version"]
    tf_ver = metadata["transformers_version"]
    timestamp = metadata["timestamp"]

    sample = data["sample_reasons"][0] if data["sample_reasons"] else ""
    reason_block = ""
    if sample:
        exp_match = re.search(r'Graph Break Reason:.*?(?=\n\n|\Z)', sample, re.DOTALL)
        reason_block = exp_match.group(0) if exp_match else sample[:200]

    lines = ["## Summary\n"]
    lines.append(f"{len(models)} models produce {total_breaks} graph breaks "
                 f"from this pattern.\n")

    if reason_block:
        lines.append("## Break Reason\n")
        lines.append("```")
        lines.append(reason_block)
        lines.append("```\n")

    lines.append("## Affected Models\n")
    lines.append("| Model | Modes | Breaks |")
    lines.append("|-------|-------|--------|")
    for name in sorted(models.keys()):
        m = models[name]
        modes = ", ".join(sorted(m["modes"]))
        lines.append(f"| {name} | {modes} | {m['breaks']} |")
    lines.append("")

    lev_models = leverage.get(issue_num, [])
    if lev_models:
        lines.append(f"## Impact\n")
        lines.append(f"Fixing this pattern moves **{len(lev_models)} models** "
                     f"to fullgraph (models where this is the only break reason).\n")

    refs = cross_refs.get(issue_num, [])
    if refs:
        lines.append("## Related Issues\n")
        for ref in refs:
            lines.append(f"- #{ref}")
        lines.append("")

    lines.append(f"## Tested On\n")
    lines.append(f"- PyTorch {pt_ver}")
    lines.append(f"- Transformers {tf_ver}")
    lines.append(f"- Sweep date: {timestamp}\n")
    lines.append(ISSUE_MARKER)

    return "\n".join(lines)


def generate_model_specific_body(issue_num, data, explain_entries, metadata,
                                 leverage, cross_refs):
    """Generate body for a model-specific issue."""
    models = data["models"]
    pt_ver = metadata["pytorch_version"]
    tf_ver = metadata["transformers_version"]
    timestamp = metadata["timestamp"]

    lines = ["## Summary\n"]
    model_names = sorted(models.keys())
    lines.append(f"Model-specific tracking for: {', '.join(model_names)}\n")

    lines.append("## Affected Models\n")
    lines.append("| Model | Modes | Status |")
    lines.append("|-------|-------|--------|")
    for name in model_names:
        m = models[name]
        modes = ", ".join(sorted(m["modes"]))
        lines.append(f"| {name} | {modes} | graph_break |")
    lines.append("")

    lines.append("## Break Reasons\n")
    for entry in explain_entries:
        if entry["name"] in models and entry.get("break_reasons"):
            lines.append(f"### {entry['name']} ({entry.get('mode', 'eval')})\n")
            seen = set()
            for br in entry["break_reasons"]:
                reason = br.get("reason", "")
                exp_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', reason)
                exp = exp_match.group(1).strip() if exp_match else reason[:100]
                if exp not in seen:
                    seen.add(exp)
                    lines.append(f"- {exp}")
            lines.append("")

    refs = cross_refs.get(issue_num, [])
    if refs:
        lines.append("## Related Issues\n")
        for ref in refs:
            lines.append(f"- #{ref}")
        lines.append("")

    lines.append(f"## Tested On\n")
    lines.append(f"- PyTorch {pt_ver}")
    lines.append(f"- Transformers {tf_ver}")
    lines.append(f"- Sweep date: {timestamp}\n")
    lines.append(ISSUE_MARKER)

    return "\n".join(lines)


def generate_infra_body(status_key, data, metadata):
    """Generate body for a corpus-infra issue."""
    models = data["models"]
    pt_ver = metadata["pytorch_version"]
    tf_ver = metadata["transformers_version"]
    timestamp = metadata["timestamp"]

    status_labels = {
        "create_error": "fail during model instantiation (before compilation)",
        "timeout": "exceed the 180-second compilation timeout",
        "eager_error": "fail during eager execution (before compilation)",
        "cuda_error": "trigger a CUDA device-side assert during eager execution",
    }

    lines = ["## Summary\n"]
    lines.append(f"{len(models)} models {status_labels.get(status_key, status_key)}.\n")

    lines.append("## Affected Models\n")
    lines.append("| Model | Modes | Error |")
    lines.append("|-------|-------|-------|")
    for name in sorted(models.keys()):
        m = models[name]
        modes = ", ".join(sorted(m["modes"]))
        error = m.get("error", "")[:80]
        lines.append(f"| {name} | {modes} | {error} |")
    lines.append("")

    lines.append(f"## Environment\n")
    lines.append(f"- PyTorch {pt_ver}")
    lines.append(f"- Transformers {tf_ver}")
    lines.append(f"- Sweep date: {timestamp}\n")
    lines.append(ISSUE_MARKER)

    return "\n".join(lines)


# ── Title generation ─────────────────────────────────────────────────────

RULE_TITLES = {
    "data_dep_shape_ops": "Data-dependent output shape ops (nonzero, repeat_interleave, masked_select)",
    "non_tensor_output": "Non-Tensor output from torch ops",
    "local_scalar_dense": "_local_scalar_dense data-dependent op",
    "logging_logger": "logging.Logger calls during forward()",
    "proxy_conversion": "Proxy conversion failure in DETR/detection models",
    "exception_handler": "Observed exception (try/except) in forward()",
    "lock_context_mgr": "Unsupported context manager (lock)",
    "contextvar": "ContextVar.get() not traceable",
    "callable_builtin": "Unsupported builtin callable()",
    "setattr_class": "Unsupported setattr on class type",
    "requires_grad": "Tensor.requires_grad mutation",
    "function_get": "function.__get__ descriptor not traceable",
    "builtin_lt": "Unsupported builtin lt operator with tensor arguments",
    "rng_seed": "RNG seeding (Generator.seed + manual_seed)",
    "uninit_module": "nn.Parameter constructor and uninitialized nn.Module",
    "import_config_skip": "Frame skip breaks from import_utils.py / configuration_utils.py",
    "frame_skip": "Frame skip breaks from output_capturing.py",
    "find_spec_skip": "Skipped function call (find_spec)",
    "tensor_item": "Tensor.item() not traceable",
    "data_dep_branch": "Data-dependent branching",
}

INFRA_TITLES = {
    "create_error": ("model fails to instantiate (create_error)", "models fail to instantiate (create_error)"),
    "timeout": ("model exceeds 180s compile timeout", "models exceed 180s compile timeout"),
    "eager_error": ("model fails during eager execution (eager_error)", "models fail during eager execution (eager_error)"),
    "cuda_error": ("model hits CUDA error during eager execution", "models hit CUDA error during eager execution"),
}


def generate_dynamo_title(rule_key, data):
    base = RULE_TITLES.get(rule_key, rule_key)
    n_models = len(data["models"])
    n_breaks = data["total_breaks"]
    return f"[dynamo] {base} ({n_models} models, {n_breaks} breaks)"


def generate_infra_title(status_key, data):
    titles = INFRA_TITLES.get(status_key, (status_key, status_key))
    n_models = len(data["models"])
    base = titles[0] if n_models == 1 else titles[1]
    return f"[corpus-infra] {n_models} {base}"


# ── Issue parsing ────────────────────────────────────────────────────────

def parse_affected_models(body):
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
                    models[model] = sorted(set(modes)) if modes else ["eval", "train"]
            else:
                break
    return models


# ── Plan building ────────────────────────────────────────────────────────

def build_sweep_report(explain_path, identify_path):
    """Build the full sweep report plan."""
    explain_entries = load_explain_data(explain_path)
    identify_entries, metadata = load_identify_data(identify_path)

    print(f"Loaded {len(explain_entries)} explain entries, "
          f"{len(identify_entries)} identify entries")
    print(f"PyTorch {metadata['pytorch_version']}, {metadata['timestamp']}")

    print("Classifying breaks...")
    classified, unclassified = classify_breaks(explain_entries)

    print("Classifying infra...")
    infra = classify_infra(identify_entries)

    print("Computing leverage...")
    leverage = compute_leverage(explain_entries)

    print("Computing cross-references...")
    cross_refs = compute_cross_refs(classified)

    print("Fetching open issues...")
    open_issues = fetch_open_issues()
    issue_by_number = {i["number"]: i for i in open_issues}
    print(f"  {len(open_issues)} open issues")

    plan_issues = []

    for rule_key, data in sorted(classified.items()):
        issue_num = data["issue_number"]
        if issue_num is None:
            continue

        models_dict = {name: {"modes": sorted(m["modes"]), "breaks": m["breaks"]}
                       for name, m in data["models"].items()}
        data_for_body = {
            "issue_number": issue_num,
            "models": {n: {"modes": m["modes"], "breaks": m["breaks"]}
                       for n, m in models_dict.items()},
            "total_breaks": data["total_breaks"],
            "sample_reasons": data["sample_reasons"],
        }

        if rule_key.startswith("model_specific_"):
            title = issue_by_number[issue_num]["title"] if issue_num in issue_by_number else f"Model-specific #{issue_num}"
            body = generate_model_specific_body(
                issue_num, data_for_body, explain_entries, metadata,
                leverage, cross_refs)
        else:
            title = generate_dynamo_title(rule_key, data_for_body)
            body = generate_dynamo_body(
                rule_key, data_for_body, metadata, leverage, cross_refs)

        current = issue_by_number.get(issue_num)
        current_title = current["title"] if current else None

        lev = leverage.get(issue_num, [])

        plan_issues.append({
            "number": issue_num,
            "rule_key": rule_key,
            "type": "model_specific" if rule_key.startswith("model_specific_") else "dynamo",
            "current_title": current_title,
            "proposed_title": title,
            "proposed_body": body,
            "model_count": len(models_dict),
            "break_count": data["total_breaks"],
            "leverage_models": len(lev),
            "title_changed": current_title != title if current_title else True,
        })

    for status_key, data in sorted(infra.items()):
        issue_num = None
        for num, skey in CORPUS_INFRA_ISSUES.items():
            if skey == status_key:
                issue_num = num
                break
        if issue_num is None:
            continue

        models_dict = {name: {"modes": sorted(m["modes"]), "error": m.get("error", "")}
                       for name, m in data["models"].items()}
        data_for_body = {"models": {n: {"modes": m["modes"], "error": m.get("error", "")}
                                    for n, m in models_dict.items()}}

        title = generate_infra_title(status_key, data_for_body)
        body = generate_infra_body(status_key, data_for_body, metadata)

        current = issue_by_number.get(issue_num)
        current_title = current["title"] if current else None

        plan_issues.append({
            "number": issue_num,
            "rule_key": status_key,
            "type": "corpus-infra",
            "current_title": current_title,
            "proposed_title": title,
            "proposed_body": body,
            "model_count": len(models_dict),
            "break_count": 0,
            "leverage_models": 0,
            "title_changed": current_title != title if current_title else True,
        })

    # Build reverse map: model → [(issue_number, rule_key), ...]
    model_to_issues = defaultdict(list)
    for rule_key, data in classified.items():
        issue_num = data["issue_number"]
        if issue_num is None:
            continue
        for model_name in data["models"]:
            model_to_issues[model_name].append((issue_num, rule_key))
    for status_key, data in infra.items():
        issue_num = None
        for num, skey in CORPUS_INFRA_ISSUES.items():
            if skey == status_key:
                issue_num = num
                break
        if issue_num:
            for model_name in data["models"]:
                model_to_issues[model_name].append((issue_num, status_key))

    # Build set of all models in current corpus and fullgraph models
    all_corpus_models = {e["name"] for e in identify_entries}
    fullgraph_models = set()
    for e in explain_entries:
        if e.get("graph_break_count", 1) == 0:
            fullgraph_models.add(e["name"])

    # Lifecycle: issues with 0 models in current data
    tracked_issue_nums = {p["number"] for p in plan_issues}
    close_candidates = []
    for issue in open_issues:
        num = issue["number"]
        if num not in tracked_issue_nums and ISSUE_MARKER in (issue.get("body") or ""):
            prev_models = parse_affected_models(issue.get("body", ""))
            disposition = {}
            for model_name in sorted(prev_models.keys()):
                if model_name not in all_corpus_models:
                    disposition[model_name] = "removed from corpus"
                elif model_name in fullgraph_models:
                    disposition[model_name] = "fullgraph on current sweep"
                elif model_name in model_to_issues:
                    targets = model_to_issues[model_name]
                    refs = ", ".join(f"#{inum} ({rk})" for inum, rk in targets)
                    disposition[model_name] = f"reclassified → {refs}"
                else:
                    disposition[model_name] = "still breaking, pattern unclassified"

            close_candidates.append({
                "number": num,
                "title": issue["title"],
                "reason": "No models matched in current sweep",
                "previous_model_count": len(prev_models),
                "model_disposition": disposition,
            })

    leverage_ranking = [
        {"issue": num, "models_to_fullgraph": len(models),
         "model_names": models[:5]}
        for num, models in sorted(leverage.items(), key=lambda x: -len(x[1]))
        if len(models) > 0
    ]

    unclassified_list = [
        {"pattern": pat, "model_count": len(d["models"]), "break_count": d["count"]}
        for pat, d in sorted(unclassified.items(), key=lambda x: -x[1]["count"])
    ]

    plan = {
        "metadata": metadata,
        "explain_path": str(explain_path),
        "identify_path": str(identify_path),
        "issues": plan_issues,
        "leverage_ranking": leverage_ranking,
        "close_candidates": close_candidates,
        "unclassified_patterns": unclassified_list,
    }

    return plan


def print_sweep_report(plan):
    """Print human-readable summary of the plan."""
    meta = plan["metadata"]
    issues = plan["issues"]
    leverage = plan["leverage_ranking"]

    dynamo = [i for i in issues if i["type"] in ("dynamo", "model_specific")]
    infra = [i for i in issues if i["type"] == "corpus-infra"]
    changed = [i for i in issues if i["title_changed"]]

    print(f"\n{'=' * 70}")
    print(f"SWEEP REPORT")
    print(f"  PyTorch {meta['pytorch_version']} ({meta['timestamp']})")
    print(f"  {len(dynamo)} dynamo issues, {len(infra)} corpus-infra issues")
    print(f"  {len(changed)} titles would change")
    print(f"{'=' * 70}")

    if leverage:
        print(f"\n--- HIGH-LEVERAGE FIXES (models → fullgraph if fixed) ---\n")
        for entry in leverage[:10]:
            print(f"  #{entry['issue']:3d}: {entry['models_to_fullgraph']} models")

    if changed:
        print(f"\n--- TITLE CHANGES ---\n")
        for i in sorted(changed, key=lambda x: x["number"]):
            print(f"  #{i['number']}: {i['current_title']}")
            print(f"     → {i['proposed_title']}")
            print()

    close = plan.get("close_candidates", [])
    if close:
        print(f"\n--- CLOSE CANDIDATES ({len(close)}) ---\n")
        for c in close:
            print(f"  #{c['number']}: {c['title']}")
            disp = c.get("model_disposition", {})
            if disp:
                print(f"    Previously {c.get('previous_model_count', '?')} models, now:")
                for model, fate in disp.items():
                    print(f"      {model}: {fate}")
            else:
                print(f"    Reason: {c['reason']}")
            print()

    unc = plan.get("unclassified_patterns", [])
    if unc:
        print(f"\n--- UNCLASSIFIED PATTERNS ---\n")
        for u in unc:
            print(f"  [{u['model_count']} models, {u['break_count']} breaks] "
                  f"{u['pattern']}")

    print(f"\n{'=' * 70}")
    total_models = sum(i["model_count"] for i in dynamo)
    total_breaks = sum(i["break_count"] for i in dynamo)
    print(f"DYNAMO: {len(dynamo)} issues covering {total_models} model/issue "
          f"pairs, {total_breaks} total breaks")
    if leverage:
        top = leverage[0]
        print(f"TOP LEVERAGE: #{top['issue']} → {top['models_to_fullgraph']} "
              f"models to fullgraph")
    print(f"{'=' * 70}\n")


# ── Plan execution ───────────────────────────────────────────────────────

def apply_sweep_update(plan):
    """PATCH all issues in the plan."""
    issues = plan["issues"]
    success = 0
    failed = 0

    for entry in sorted(issues, key=lambda x: x["number"]):
        num = entry["number"]
        title = entry["proposed_title"]
        body = entry["proposed_body"]

        print(f"  Updating #{num}: {title[:60]}...")
        result = _proxy_api(
            f"/repos/{REPO_SLUG}/issues/{num}",
            method="PATCH",
            body={"title": title, "body": body},
        )
        if result and "number" in result:
            success += 1
        else:
            print(f"    FAILED")
            failed += 1
        time.sleep(0.5)

    print(f"\nDone: {success} updated, {failed} failed.")


# ── CLI ──────────────────────────────────────────────────────────────────

def cmd_sweep_report(args):
    explain_path = Path(args.explain)
    identify_path = Path(args.identify)

    if not explain_path.exists():
        print(f"ERROR: {explain_path} not found")
        sys.exit(1)
    if not identify_path.exists():
        print(f"ERROR: {identify_path} not found")
        sys.exit(1)

    plan = build_sweep_report(explain_path, identify_path)
    print_sweep_report(plan)

    plan_out = explain_path.parent / "sweep-report.json"
    with open(plan_out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved to: {plan_out}")
    print(f"Review, then run: python tools/file_issues.py sweep-update "
          f"--plan {plan_out}")


def cmd_sweep_update(args):
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found")
        sys.exit(1)

    with open(plan_path) as f:
        plan = json.load(f)

    meta = plan["metadata"]
    issues = plan["issues"]
    print(f"Applying sweep update plan")
    print(f"  PyTorch {meta['pytorch_version']} ({meta['timestamp']})")
    print(f"  {len(issues)} issues to update\n")

    apply_sweep_update(plan)


def main():
    parser = argparse.ArgumentParser(
        description="Post-sweep issue management for the OSS model graph-break corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sub_report = subparsers.add_parser(
        "sweep-report",
        help="Analyze sweep results and generate update plan (read-only)")
    sub_report.add_argument(
        "--explain", required=True, metavar="EXPLAIN_JSON",
        help="Path to explain_results.json")
    sub_report.add_argument(
        "--identify", required=True, metavar="IDENTIFY_JSON",
        help="Path to identify_results.json")

    sub_update = subparsers.add_parser(
        "sweep-update",
        help="Apply a reviewed update plan to GitHub issues")
    sub_update.add_argument(
        "--plan", required=True, metavar="PLAN_JSON",
        help="Path to sweep-report.json plan file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "sweep-report":
        cmd_sweep_report(args)
    elif args.command == "sweep-update":
        cmd_sweep_update(args)


if __name__ == "__main__":
    main()
