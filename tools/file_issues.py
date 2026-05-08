#!/usr/bin/env python3
"""Post-sweep issue management for the OSS model graph-break corpus.

Subcommands:
  sweep-report        Analyze sweep results, generate update plan (read-only)
  sweep-update        Apply a reviewed plan to corpus-repo issues (writes)
  correctness-report  Analyze a correctness sweep, generate issue plan (read-only)
  correctness-apply   Apply a correctness plan (writes to corpus repo)
  pytorch-upstream    Build a self-contained pytorch/pytorch issue body or
                      comment by running a repro script in 1+ venvs and
                      capturing output + env (read-only by default; --post
                      creates the issue / posts the comment).

The first four target our internal corpus repo (penguinwu/oss-model-graph-
break-corpus). The pytorch-upstream subcommand targets pytorch/pytorch and
exists to make first-time upstream filings reproducible end-to-end (see
tools/issue_filing_plan.md §2 for the workflow + checklist).

Usage examples:
  # Sweep workflow (corpus repo)
  python tools/file_issues.py sweep-report \\
    --explain sweep_results/nightly/2026-04-19/explain_results.json \\
    --identify sweep_results/nightly/2026-04-19/identify_results.json

  python tools/file_issues.py sweep-update \\
    --plan sweep_results/nightly/2026-04-19/sweep-report.json

  # Upstream pytorch issue (dry-run by default)
  python tools/file_issues.py pytorch-upstream \\
    --script /tmp/blt_init_repro.py \\
    --venv pt211:~/envs/torch211/bin/python \\
    --venv pt212:~/envs/torch-nightly-cu128/bin/python \\
    --pythonpath /home/pengwu/envs/modellibs/transformers-5.5.3 \\
    --title "Perf regression: ..." \\
    --summary /tmp/issue_summary.md \\
    --output /tmp/issue_body.md

  # Same, posting as a comment to existing issue #182116
  python tools/file_issues.py pytorch-upstream \\
    --script /tmp/blt_init_repro.py --venv pt211:... --venv pt212:... \\
    --comment 182116 --summary /tmp/comment_intro.md --output /tmp/body.md --post
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
REPO_SLUG = "penguinwu/oss-model-graph-break-corpus"
PROXY_URL = "http://localhost:7824/fetch"
ISSUE_MARKER = "<!-- filed-by: otter/file_issues.py -->"

# ── HARD GUARD against accidental pytorch/pytorch posting ───────────────────
# Per Peng directive 2026-05-08 19:56 ET: pytorch-upstream posting must require
# a deliberate code edit (NOT just an environment variable or CLI flag) before
# it can fire. This prevents an agent from EVER posting to pytorch/pytorch
# without explicit human review at the source-code level.
#
# To enable a pytorch/pytorch post: edit this file, set the constant to True,
# get the change reviewed, then run with --post. After posting, set it back
# to False. The mechanical defense-in-depth is by design.
PYTORCH_UPSTREAM_POSTING_ENABLED = False


# ── Graph-break classifier ──────────────────────────────────────────────
# Maps break_reasons from explain_results.json to issue-level categories.
# Each rule: (key, issue_number, match_fn(explanation, location) → bool)

GRAPH_BREAK_RULES = [
    # ── EXPLANATION-based rules (more specific; must come first) ──
    # These match on the actual graph-break reason text. They are MORE specific
    # than the location-based rules below — a break at import_utils.py:1525
    # with "find_spec" in the explanation should be classified as find_spec
    # (Issue #7), NOT as the generic import_utils skip cluster (Issue #5).
    # Bug fixed 2026-05-03: previously find_spec_skip was after
    # import_config_skip in the list, causing 96 find_spec breaks/sweep to be
    # misfiled under Issue #5. Same shadowing pattern would have hit any
    # explanation-based rule placed below the loc-based catch-alls.
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
    ("find_spec_skip", 7, lambda exp, loc:
        "find_spec" in exp),
    ("tensor_item", 56, lambda exp, loc:
        "Tensor.item()" in exp),
    ("data_dep_branch", 54, lambda exp, loc:
        "data-dependent branching" in exp),
    # ── LOCATION-based catch-alls (must come LAST) ──
    # These fire when a break happens at a known transformers wrapper file
    # but didn't match any explanation-based rule above. They are the
    # residual bucket for that file — anything specific about the break
    # text was already handled by an upstream rule.
    ("import_config_skip", 5, lambda exp, loc:
        "import_utils.py" in loc or "configuration_utils.py" in loc),
    ("frame_skip", 2, lambda exp, loc:
        "output_capturing.py" in loc),
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


# ── Correctness classifier ──────────────────────────────────────────────
# Groups divergent models from a correctness sweep by model family.
# Each rule: (key, label, match_fn(name) → bool)

CORRECTNESS_FAMILIES = [
    ("phi4_multimodal", "Phi4 Multimodal vision",
        lambda name: "Phi4Multimodal" in name),
    ("aimv2", "AIMv2 family",
        lambda name: "Aimv2" in name),
    ("doge", "Doge family",
        lambda name: name.startswith("Doge")),
    ("glm_moe_dsa", "GLM MoE DSA family",
        lambda name: "GlmMoeDsa" in name),
    ("gemma_family", "Gemma family (incl. T5Gemma, VaultGemma)",
        lambda name: "Gemma" in name),
    ("idefics2", "Idefics2 family",
        lambda name: "Idefics2" in name),
    ("dia", "Dia family",
        lambda name: name.startswith("Dia")),
]


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


def load_correctness_data(correctness_path, pytorch_version="unknown",
                          transformers_version="unknown"):
    """Load a correctness sweep result file.

    The correctness_results.json metadata does not include framework versions
    (only timestamp + tolerance), so callers must pass them explicitly.
    """
    with open(correctness_path) as f:
        data = json.load(f)
    results = data.get("results", [])
    meta = data.get("metadata", {}) if isinstance(data, dict) else {}
    metadata = {
        "pytorch_version": pytorch_version,
        "transformers_version": transformers_version,
        "timestamp": meta.get("timestamp", "unknown"),
        "tolerance": meta.get("tolerance", {}),
    }
    return results, metadata


def load_identify_data(identify_path):
    # Route through canonical loader so amendments are merged
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from sweep.results_loader import load_raw, load_results_list
    results = load_results_list(identify_path)
    raw = load_raw(identify_path)
    meta = raw.get("metadata", {})
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


def classify_correctness(results):
    """Group divergent models from a correctness sweep by family.

    Returns:
        ({family_key: {"label": str, "models": {name: {...}}}}, [unmatched_names])
    """
    families = defaultdict(lambda: {"label": "", "models": {}})
    unmatched = []

    for r in results:
        if r.get("status") != "divergence":
            continue
        name = r["name"]
        matched = False
        for key, label, match_fn in CORRECTNESS_FAMILIES:
            if match_fn(name):
                families[key]["label"] = label
                families[key]["models"][name] = {
                    "max_diff": r.get("max_diff"),
                    "severity_ratio": r.get("severity_ratio"),
                    "first_divergence": r.get("first_divergence", ""),
                    "compared_fields": r.get("compared_fields", []),
                    "mode": r.get("mode", "eval"),
                }
                matched = True
                break
        if not matched:
            unmatched.append(name)

    return dict(families), unmatched


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


def generate_correctness_title(family_key, data):
    label = data["label"]
    n = len(data["models"])
    return f"[correctness] {label} — divergence vs eager ({n} models)"


def generate_correctness_body(family_key, data, metadata):
    """Generate issue body for a correctness divergence family."""
    models = data["models"]
    pt_ver = metadata["pytorch_version"]
    tf_ver = metadata["transformers_version"]
    timestamp = metadata["timestamp"]
    tol = metadata.get("tolerance", {})

    sev_values = [m["severity_ratio"] for m in models.values()
                  if m["severity_ratio"] is not None]
    diff_values = [m["max_diff"] for m in models.values()
                   if m["max_diff"] is not None]
    overall_max_diff = max(diff_values, default=0)
    overall_max_sev = max(sev_values, default=0)

    lines = ["## Summary\n"]
    lines.append(f"{len(models)} models in this family produce numerical "
                 f"divergence between compiled and eager outputs.\n")
    lines.append(f"- Worst max_diff: **{overall_max_diff:.6g}**")
    lines.append(f"- Worst severity_ratio: **{overall_max_sev:.1f}** "
                 f"(divergence / tolerance)\n")

    lines.append("## Tolerance Used\n")
    lines.append(f"- atol: {tol.get('atol', '?')}")
    lines.append(f"- rtol: {tol.get('rtol', '?')}")
    lines.append(f"- dtype: {tol.get('dtype', '?')}\n")

    lines.append("## Affected Models\n")
    lines.append("| Model | max_diff | severity_ratio | first_divergence |")
    lines.append("|-------|---------:|---------------:|------------------|")
    for name in sorted(models.keys(),
                       key=lambda n: -(models[n]["severity_ratio"] or 0)):
        m = models[name]
        md = m["max_diff"] if m["max_diff"] is not None else 0
        sev = m["severity_ratio"] if m["severity_ratio"] is not None else 0
        fd = m["first_divergence"]
        lines.append(f"| {name} | {md:.6g} | {sev:.1f} | {fd} |")
    lines.append("")

    lines.append("## Reproduction\n")
    lines.append("Run the corpus correctness pass for an affected model:")
    lines.append("```bash")
    lines.append("python sweep/run_correctness.py --models <model_name>")
    lines.append("```\n")

    lines.append("## Tested On\n")
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


# ── Correctness plan building ───────────────────────────────────────────

def build_correctness_report(correctness_path, pytorch_version,
                             transformers_version):
    """Build a correctness-issue creation plan from a sweep result file."""
    results, metadata = load_correctness_data(
        correctness_path, pytorch_version, transformers_version)

    print(f"Loaded {len(results)} correctness entries")
    print(f"PyTorch {metadata['pytorch_version']}, {metadata['timestamp']}")

    print("Classifying divergences...")
    families, unmatched = classify_correctness(results)

    print("Fetching open issues...")
    open_issues = fetch_open_issues()
    title_to_issue = {i["title"]: i for i in open_issues}
    print(f"  {len(open_issues)} open issues")

    plan_issues = []
    for family_key in sorted(families.keys()):
        data = families[family_key]
        title = generate_correctness_title(family_key, data)
        body = generate_correctness_body(family_key, data, metadata)

        existing = title_to_issue.get(title)
        action = "update" if existing else "create"
        sev_values = [m["severity_ratio"] for m in data["models"].values()
                      if m["severity_ratio"] is not None]
        max_sev = max(sev_values, default=0)

        plan_issues.append({
            "family_key": family_key,
            "label": data["label"],
            "action": action,
            "existing_number": existing["number"] if existing else None,
            "proposed_title": title,
            "proposed_body": body,
            "model_count": len(data["models"]),
            "max_severity": max_sev,
        })

    plan = {
        "metadata": metadata,
        "correctness_path": str(correctness_path),
        "issues": plan_issues,
        "unmatched_models": unmatched,
    }
    return plan


def print_correctness_report(plan):
    """Print human-readable summary of the correctness plan."""
    meta = plan["metadata"]
    issues = plan["issues"]
    creates = [i for i in issues if i["action"] == "create"]
    updates = [i for i in issues if i["action"] == "update"]

    print(f"\n{'=' * 70}")
    print(f"CORRECTNESS REPORT")
    print(f"  PyTorch {meta['pytorch_version']} ({meta['timestamp']})")
    print(f"  {len(creates)} new issues to create, {len(updates)} updates")
    print(f"  Tolerance: atol={meta['tolerance'].get('atol')}, "
          f"rtol={meta['tolerance'].get('rtol')}, "
          f"dtype={meta['tolerance'].get('dtype')}")
    print(f"{'=' * 70}\n")

    for i in sorted(issues, key=lambda x: -x["max_severity"]):
        action_tag = "CREATE" if i["action"] == "create" else f"UPDATE #{i['existing_number']}"
        print(f"  [{action_tag}] {i['proposed_title']}")
        print(f"     models={i['model_count']}  max_severity={i['max_severity']:.1f}")

    if plan.get("unmatched_models"):
        print(f"\n--- UNMATCHED DIVERGENT MODELS (need new family rule) ---\n")
        for name in plan["unmatched_models"]:
            print(f"  {name}")

    print(f"\n{'=' * 70}\n")


def apply_correctness_plan(plan):
    """POST new issues / PATCH existing ones based on the correctness plan."""
    issues = plan["issues"]
    success = 0
    failed = 0

    for entry in issues:
        title = entry["proposed_title"]
        body = entry["proposed_body"]
        if entry["action"] == "create":
            print(f"  POST: {title[:60]}...")
            result = _proxy_api(
                f"/repos/{REPO_SLUG}/issues",
                method="POST",
                body={"title": title, "body": body},
            )
        else:
            num = entry["existing_number"]
            print(f"  PATCH #{num}: {title[:60]}...")
            result = _proxy_api(
                f"/repos/{REPO_SLUG}/issues/{num}",
                method="PATCH",
                body={"title": title, "body": body},
            )
        if result and "number" in result:
            success += 1
            print(f"    → #{result['number']}")
        else:
            failed += 1
            print(f"    FAILED")
        time.sleep(0.5)

    print(f"\nDone: {success} succeeded, {failed} failed.")


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

    # Refuse experimental sweeps unless explicitly allowed
    with open(explain_path) as f:
        explain_meta = json.load(f).get("metadata", {})
    run_name = explain_meta.get("run_name")
    if run_name and not getattr(args, "allow_experimental", False):
        print(f"ERROR: this sweep is tagged experimental (run_name={run_name!r}).")
        print("Issue tracker is only fed by official baseline (cron) sweeps.")
        print("Pass --allow-experimental to override (you'll need to clean up later).")
        sys.exit(2)

    plan = build_sweep_report(explain_path, identify_path)
    if run_name:
        plan.setdefault("metadata", {})["run_name"] = run_name
    print_sweep_report(plan)

    plan_out = explain_path.parent / "sweep-report.json"
    with open(plan_out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved to: {plan_out}")
    print(f"Review, then run: python tools/file_issues.py sweep-update "
          f"--plan {plan_out}")


def cmd_correctness_report(args):
    correctness_path = Path(args.correctness)
    if not correctness_path.exists():
        print(f"ERROR: {correctness_path} not found")
        sys.exit(1)

    plan = build_correctness_report(
        correctness_path, args.pytorch_version, args.transformers_version)
    print_correctness_report(plan)

    plan_out = correctness_path.parent / "correctness-report.json"
    with open(plan_out, "w") as f:
        json.dump(plan, f, indent=2)
    print(f"Plan saved to: {plan_out}")
    print(f"Review, then run: python tools/file_issues.py correctness-apply "
          f"--plan {plan_out}")


def cmd_correctness_apply(args):
    plan_path = Path(args.plan)
    if not plan_path.exists():
        print(f"ERROR: {plan_path} not found")
        sys.exit(1)

    with open(plan_path) as f:
        plan = json.load(f)

    meta = plan["metadata"]
    issues = plan["issues"]
    print(f"Applying correctness plan")
    print(f"  PyTorch {meta['pytorch_version']} ({meta['timestamp']})")
    print(f"  {len(issues)} issues to process\n")

    apply_correctness_plan(plan)


def cmd_sweep_update(args):
    plan_path = Path(args.plan)
    if plan_path.exists():
        with open(plan_path) as f:
            plan_meta = json.load(f).get("metadata", {})
        run_name = plan_meta.get("run_name")
        if run_name and not getattr(args, "allow_experimental", False):
            print(f"ERROR: this plan came from an experimental sweep (run_name={run_name!r}).")
            print("Pass --allow-experimental to apply anyway.")
            sys.exit(2)
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


# ── pytorch-upstream subcommand ─────────────────────────────────────────
# Build a self-contained pytorch/pytorch issue body or comment by running a
# repro script in 1+ venvs and capturing output + collect_env. Designed so
# the receiver (an upstream maintainer) has everything needed to reproduce
# without bouncing back to ask. See tools/issue_filing_plan.md §2.
#
# Anti-fragmentation: this lives in file_issues.py to share _get_github_token
# / _proxy_api with the corpus-repo subcommands, even though it targets a
# different repo (pytorch/pytorch). Adding a separate tool would split the
# GitHub API plumbing across two files.

UPSTREAM_REPO_SLUG = "pytorch/pytorch"
UPSTREAM_BODY_MARKER = "<!-- assembled-by: file_issues.py pytorch-upstream -->"


def _scrub_paths(text):
    """Strip private/internal absolute paths from captured output before publishing.

    Replaces:
      - /home/<user>/.../site-packages/    →  .../site-packages/
      - /home/<user>/envs/<venv>/lib/...   →  .../
      - /home/<user>/envs/modellibs/<pkg>-<ver>/  →  .../
      - /usr/local/fbcode/.../             →  .../    (Meta-internal Python prefix)
      - any other /home/<user>/<path>      →  .../

    Idempotent and safe to call on already-scrubbed text. Designed to be applied
    to user-readable output (cProfile / collect_env / script stdout), not to the
    rendered Setup/Run sections (those are templated portably from the start).
    """
    if not text:
        return text
    # Meta-internal Python prefix (fbcode platform builds)
    text = re.sub(r'/usr/local/fbcode/[^/\s]+/lib/python3\.\d+/', '.../python3/', text)
    # Modellibs trees: /home/<user>/envs/modellibs/<pkg>-<ver>/  →  .../
    text = re.sub(r'/home/[^/\s]+/envs/modellibs/[\w\-\.]+/', '.../', text)
    # Standard venv site-packages: /home/<user>/envs/<venv>/lib/python3.X/site-packages/  →  .../site-packages/
    text = re.sub(
        r'/home/[^/\s]+/envs/[^/\s]+/lib/python3\.\d+/site-packages/',
        '.../site-packages/', text,
    )
    # Catch-all for any remaining /home/<user>/... prefix (must come last)
    text = re.sub(r'/home/[^/\s]+/', '.../', text)
    return text


def _parse_venv_spec(spec):
    """Parse a --venv argument: NAME:PATH_TO_PYTHON. Returns (name, python_bin)."""
    if ":" not in spec:
        print(f"ERROR: --venv must be NAME:PATH (got {spec!r})")
        sys.exit(2)
    name, python_bin = spec.split(":", 1)
    python_bin = str(Path(python_bin).expanduser())
    if not Path(python_bin).is_file() or not os.access(python_bin, os.X_OK):
        print(f"ERROR: --venv {name!r} python_bin not executable: {python_bin}")
        sys.exit(2)
    return name, python_bin


def _run_capture(python_bin, args, env_extra=None, timeout_s=1800):
    """Run [python_bin] + args with optional env additions, capture combined output."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        cp = subprocess.run(
            [python_bin] + list(args),
            env=env, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"(timed out after {timeout_s}s)"
    out = cp.stdout or ""
    err = cp.stderr or ""
    if cp.returncode != 0:
        out += f"\n[exit code {cp.returncode}]\n{err}"
    return cp.returncode, out


def cmd_pytorch_upstream(args):
    if not args.venv or len(args.venv) < 1:
        print("ERROR: --venv required (at least one)")
        sys.exit(2)
    if args.comment and args.title:
        print("ERROR: --comment and --title are mutually exclusive (comment posts to existing issue, has no title)")
        sys.exit(2)
    if args.post and not args.comment and not args.title:
        print("ERROR: --post needs either --comment ISSUE# (post to existing) or --title (create new)")
        sys.exit(2)

    script_path = Path(args.script).expanduser().resolve()
    if not script_path.is_file():
        print(f"ERROR: script not found: {script_path}")
        sys.exit(2)
    script_src = script_path.read_text()

    summary_text = ""
    if args.summary:
        summary_path = Path(args.summary).expanduser().resolve()
        if not summary_path.is_file():
            print(f"ERROR: --summary file not found: {summary_path}")
            sys.exit(2)
        summary_text = summary_path.read_text().rstrip()

    env_extra = {}
    if args.pythonpath:
        env_extra["PYTHONPATH"] = ":".join(
            str(Path(p).expanduser()) for p in args.pythonpath.split(":")
        )

    # Parse venvs and capture metadata + outputs for each.
    venvs = [_parse_venv_spec(s) for s in args.venv]
    captured = []  # list of dicts: {name, python_bin, torch_ver, cuda_variant,
                   #                 transformers_ver, diffusers_ver, collect_env_text, run_text}

    # Probe script — captures torch + cuda variant + transformers/diffusers versions in
    # one shot so the rendered Setup section can template the exact `pip install` lines
    # an upstream maintainer would run on a fresh machine.
    PROBE_SRC = (
        "import torch\n"
        "print(f'TORCH_VERSION={torch.__version__}')\n"
        "print(f'TORCH_GIT={torch.version.git_version}')\n"
        "for pkg in ('transformers','diffusers','timm'):\n"
        "    try:\n"
        "        m=__import__(pkg); print(f'{pkg.upper()}_VERSION={m.__version__}')\n"
        "    except Exception: print(f'{pkg.upper()}_VERSION=')\n"
    )

    for name, python_bin in venvs:
        print(f"[upstream] capturing metadata + collect_env + script run for {name} ({python_bin})", file=sys.stderr)
        # 1. version probe (torch + cuda variant + transformers/diffusers/timm)
        _, probe_out = _run_capture(python_bin, ["-c", PROBE_SRC],
                                    env_extra=env_extra, timeout_s=60)
        meta = {}
        for line in (probe_out or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                meta[k.strip()] = v.strip()
        torch_ver = meta.get("TORCH_VERSION", "?")
        # Derive cuda variant from torch version: '2.11.0+cu128' → cu128; '+cpu' → cpu;
        # bare '2.11.0' → 'cu128' (educated default for the current era's release wheels)
        m = re.search(r'\+(cu\d+|cpu|rocm[\d.]+)$', torch_ver)
        cuda_variant = m.group(1) if m else "cu128"

        # 2. collect_env (skipped if --no-collect-env)
        if args.no_collect_env:
            collect_env_text = "(skipped via --no-collect-env)"
        else:
            _, ce_out = _run_capture(python_bin, [
                "-m", "torch.utils.collect_env"
            ], env_extra=env_extra, timeout_s=120)
            collect_env_text = _scrub_paths(ce_out.strip())

        # 3. the script itself (skipped if --no-run)
        if args.no_run:
            run_text = "(--no-run: script not executed by tool)"
        else:
            _, run_out = _run_capture(python_bin, [str(script_path)],
                                      env_extra=env_extra, timeout_s=args.timeout)
            run_text = _scrub_paths(run_out.strip())

        captured.append({
            "name": name,
            "python_bin": python_bin,
            "torch_ver": torch_ver,
            "cuda_variant": cuda_variant,
            "transformers_ver": meta.get("TRANSFORMERS_VERSION", ""),
            "diffusers_ver": meta.get("DIFFUSERS_VERSION", ""),
            "timm_ver": meta.get("TIMM_VERSION", ""),
            "collect_env_text": collect_env_text,
            "run_text": run_text,
        })

    # ── Render the body ──
    # Inject the case_id marker FIRST so the upstream body ties back to the
    # subagents/file-issue invocation. Without this, audit_issue_footers.py
    # cannot link an upstream issue to the case file that generated it.
    # (Adversary review case adv-2026-05-08-161753-file-issue-impl gap #2.)
    case_id_marker = f"<!-- via subagents/file-issue case_id={getattr(args, 'via_skill', 'unknown')} -->"
    body_parts = [case_id_marker, UPSTREAM_BODY_MARKER, ""]
    if summary_text:
        body_parts += [summary_text, ""]

    body_parts += ["## Reproducer", "", "Save as `repro.py`:", "",
                   "```python", script_src.rstrip(), "```", ""]

    # Setup: portable pip install lines per venv (NEVER /home/<user> paths).
    body_parts += ["## Setup", ""]
    for c in captured:
        venv_name = f"{c['name']}_venv"
        is_dev = ".dev" in c["torch_ver"]
        if c["cuda_variant"] == "cpu":
            index_url = "https://download.pytorch.org/whl/nightly/cpu" if is_dev else "https://download.pytorch.org/whl/cpu"
        else:
            index_url = (f"https://download.pytorch.org/whl/nightly/{c['cuda_variant']}"
                         if is_dev else f"https://download.pytorch.org/whl/{c['cuda_variant']}")
        body_parts.append(f"```bash")
        body_parts.append(f"# {c['name']}: torch=={c['torch_ver']}")
        body_parts.append(f"python3 -m venv ~/{venv_name}")
        pre_flag = "--pre " if is_dev else ""
        body_parts.append(
            f"~/{venv_name}/bin/pip install {pre_flag}torch=={c['torch_ver']} \\"
        )
        body_parts.append(f"  --index-url {index_url}")
        if c["transformers_ver"]:
            body_parts.append(f"~/{venv_name}/bin/pip install transformers=={c['transformers_ver']}")
        if c["diffusers_ver"]:
            body_parts.append(f"~/{venv_name}/bin/pip install diffusers=={c['diffusers_ver']}")
        if c["timm_ver"]:
            body_parts.append(f"~/{venv_name}/bin/pip install timm=={c['timm_ver']}")
        body_parts.append(f"```")
        body_parts.append("")

    # Run: generic ~/<NAME>_venv path, NEVER absolute python_bin.
    body_parts += ["## Run", ""]
    for c in captured:
        venv_name = f"{c['name']}_venv"
        body_parts.append(f"```bash")
        body_parts.append(f"~/{venv_name}/bin/python repro.py    # {c['name']}")
        body_parts.append(f"```")
    body_parts.append("")

    body_parts += ["## Captured output", ""]
    for c in captured:
        body_parts += [
            f"<details><summary>{c['name']} (`{c['torch_ver']}`)</summary>", "",
            "```", c["run_text"] or "(no output)", "```", "", "</details>", "",
        ]

    if not args.no_collect_env:
        body_parts += ["## Environment (`python -m torch.utils.collect_env`)", ""]
        for c in captured:
            body_parts += [
                f"<details><summary>{c['name']}</summary>", "",
                "```", c["collect_env_text"] or "(no output)", "```", "", "</details>", "",
            ]

    body = "\n".join(body_parts).rstrip() + "\n"

    # Write body to --output (default stdout).
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(body)
        print(f"[upstream] body written: {out_path} ({len(body)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(body)

    # Optional: actually post.
    if args.post:
        # HARD GUARD: refuse to post unless the code-level constant is True.
        # Per Peng directive 2026-05-08 19:56 ET — the only way to enable a
        # pytorch/pytorch post is a deliberate source-code edit. CLI flag
        # alone is not enough.
        if not PYTORCH_UPSTREAM_POSTING_ENABLED:
            print(
                "ERROR: pytorch-upstream posting is disabled by code-level guard.\n"
                "  To enable: edit tools/file_issues.py and set\n"
                "    PYTORCH_UPSTREAM_POSTING_ENABLED = True\n"
                "  Get the change reviewed by Peng before flipping. After posting,\n"
                "  set the constant back to False. This guard prevents accidental\n"
                "  posts to pytorch/pytorch even when --post is passed.",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.comment:
            endpoint = f"/repos/{UPSTREAM_REPO_SLUG}/issues/{args.comment}/comments"
            payload = {"body": body}
            print(f"[upstream] POSTing comment to {UPSTREAM_REPO_SLUG}#{args.comment}", file=sys.stderr)
        else:
            endpoint = f"/repos/{UPSTREAM_REPO_SLUG}/issues"
            payload = {"title": args.title, "body": body}
            if args.label:
                payload["labels"] = list(args.label)
            print(f"[upstream] POSTing new issue to {UPSTREAM_REPO_SLUG}: title={args.title!r}", file=sys.stderr)
        result = _proxy_api(endpoint, method="POST", body=payload)
        if result and isinstance(result, dict):
            url = result.get("html_url")
            print(f"[upstream] POSTed: {url}", file=sys.stderr)
        else:
            print(f"[upstream] POST failed (see error above)", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[upstream] DRY RUN — pass --post to actually create issue / post comment.", file=sys.stderr)


def _validate_via_skill(case_id: str, body_path: str | None) -> tuple[bool, str, bytes | None]:
    """Validate a --via-skill <case_id> claim before posting.

    Reads subagents/file-issue/invocations/<case_id>.md and checks:
    1. The case file exists
    2. mode_a_verdict in {proceed, proceed-with-fixes}
    3. mode_b_sha256 is non-empty (Mode B reached)
    4. If body_path provided, sha256(body) matches body_sha256 in case file
    5. If body_path provided AND target is corpus issue, body must contain the
       footer marker `<!-- via subagents/file-issue case_id=<case_id> -->`
       (gap #3 — Mode B is trusted to include the marker but defense-in-depth
       requires the tool to check too)

    Returns (ok, error_message, body_bytes). On ok=True, error_message is empty
    and body_bytes is the verified body content (closes TOCTOU race — gap #11).
    """
    import hashlib
    case_file = REPO_ROOT / f"subagents/file-issue/invocations/{case_id}.md"
    if not case_file.is_file():
        return False, (f"--via-skill: case file not found: {case_file}.\n"
                       f"  Did you forget to invoke subagents/file-issue/SKILL.md? "
                       f"Or rebuild the aggregator with "
                       f"`python3 tools/build_invocations_log.py subagents/file-issue/`."), None
    text = case_file.read_text()

    # Parse frontmatter (same logic as build_invocations_log.py)
    if not text.startswith("---\n"):
        return False, f"--via-skill: case file missing YAML frontmatter: {case_file}", None
    end = text.find("\n---\n", 4)
    if end < 0:
        return False, f"--via-skill: case file frontmatter not closed: {case_file}", None
    fm_block = text[4:end]
    fields: dict[str, str] = {}
    for line in fm_block.splitlines():
        m = re.match(r"^([a-z0-9_]+):\s*(.+)$", line.strip())
        if not m:
            continue
        k, v = m.group(1), m.group(2).strip()
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        fields[k] = v

    verdict = fields.get("mode_a_verdict", "")
    if verdict not in ("proceed", "proceed-with-fixes"):
        return False, (f"--via-skill: case {case_id} has mode_a_verdict='{verdict}'. "
                       f"Required: proceed or proceed-with-fixes. Run Mode A again "
                       f"after revisions, or check the case file directly."), None

    mode_b_sha = fields.get("mode_b_sha256", "")
    if not mode_b_sha:
        return False, (f"--via-skill: case {case_id} has no mode_b_sha256 — Mode B did "
                       f"not produce a body, or the case file is incomplete."), None

    body_bytes: bytes | None = None
    if body_path:
        # Read once, hash once (closes TOCTOU race — gap #11). Caller uses these
        # exact bytes for the POST payload.
        body_bytes = Path(body_path).read_bytes()
        actual_sha = hashlib.sha256(body_bytes).hexdigest()
        expected_sha = fields.get("body_sha256", "")
        if not expected_sha:
            return False, (f"--via-skill: case {case_id} missing body_sha256 in "
                           f"frontmatter. Cannot verify integrity of body being posted."), None
        if actual_sha != expected_sha:
            return False, (f"--via-skill: body sha256 mismatch.\n"
                           f"  expected (from case file): {expected_sha}\n"
                           f"  actual (from {body_path}): {actual_sha}\n"
                           f"  The body file has been modified since Mode B wrote it. "
                           f"Re-run Mode B if the modification was intentional."), None

        # Gap #3: enforce footer marker presence at the tool level
        # (Mode B is trusted to include it; this is defense-in-depth.)
        expected_marker = f"<!-- via subagents/file-issue case_id={case_id} -->"
        if expected_marker.encode() not in body_bytes:
            return False, (f"--via-skill: body does not contain required footer "
                           f"marker.\n"
                           f"  expected: {expected_marker}\n"
                           f"  in body file: {body_path}\n"
                           f"  Mode B's persona is supposed to include this marker. "
                           f"Re-run Mode B; it may have drifted."), None

    return True, "", body_bytes


def _update_case_posted_url(case_id: str, posted_url: str) -> None:
    """Atomic-ish update of the `posted_url:` line in a case file's frontmatter
    after a successful POST. Closes gap #4 (manual update step prone to silent
    omission). Caller must have already successfully posted.
    """
    case_file = REPO_ROOT / f"subagents/file-issue/invocations/{case_id}.md"
    text = case_file.read_text()
    # Only operate on the frontmatter block (between leading --- and next ---)
    if not text.startswith("---\n"):
        print(f"WARNING: case file lacks frontmatter; cannot update posted_url: {case_file}",
              file=sys.stderr)
        return
    end = text.find("\n---\n", 4)
    if end < 0:
        print(f"WARNING: case file frontmatter not closed; cannot update posted_url: {case_file}",
              file=sys.stderr)
        return
    fm = text[4:end]
    body_after = text[end:]
    # Replace existing `posted_url:` line (handles quoted/unquoted)
    new_fm, n = re.subn(
        r"^posted_url:\s*.+$",
        f"posted_url: {posted_url}",
        fm, count=1, flags=re.MULTILINE,
    )
    if n == 0:
        # No existing line — append before the closing ---
        new_fm = fm.rstrip() + f"\nposted_url: {posted_url}\n"
    new_text = "---\n" + new_fm + body_after
    case_file.write_text(new_text)


def cmd_corpus_issue(args):
    """Post a single corpus-repo issue, gated by --via-skill validation."""
    ok, err, body_bytes = _validate_via_skill(args.via_skill, args.body)
    if not ok:
        print(err, file=sys.stderr)
        sys.exit(1)
    # Use the bytes _validate_via_skill already hashed (gap #11 — no TOCTOU).
    body_text = body_bytes.decode("utf-8")
    if not args.post:
        print("[DRY-RUN] would post to penguinwu/oss-model-graph-break-corpus")
        print(f"  case_id: {args.via_skill}")
        print(f"  title:   {args.title}")
        print(f"  labels:  {args.label}")
        print(f"  body:    {len(body_text)} chars (sha256 + footer marker verified)")
        print(f"  Add --post to actually create the issue.")
        return
    # Post via GitHub API
    payload = {"title": args.title, "body": body_text}
    if args.label:
        payload["labels"] = list(args.label)
    response = _proxy_api(f"/repos/{REPO_SLUG}/issues", method="POST", body=payload)
    issue_url = response.get("html_url", "<unknown>")
    # Auto-update posted_url in the case file (gap #4)
    _update_case_posted_url(args.via_skill, issue_url)
    print(f"Posted: {issue_url}")
    print(f"  case_id: {args.via_skill}")
    print(f"  case file's posted_url field updated automatically.")
    print(f"  Refresh the aggregate index: "
          f"`python3 tools/build_invocations_log.py subagents/file-issue/`")


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
    sub_report.add_argument(
        "--allow-experimental", action="store_true",
        help="Permit running on a sweep tagged with run_name (experimental). "
             "By default, only baseline (cron) sweeps may update issue tracker.")

    sub_update = subparsers.add_parser(
        "sweep-update",
        help="Apply a reviewed update plan to GitHub issues")
    sub_update.add_argument(
        "--plan", required=True, metavar="PLAN_JSON",
        help="Path to sweep-report.json plan file")
    sub_update.add_argument(
        "--allow-experimental", action="store_true",
        help="Permit applying a plan generated from an experimental sweep.")

    sub_corr_report = subparsers.add_parser(
        "correctness-report",
        help="Analyze a correctness sweep and generate an issue plan (read-only)")
    sub_corr_report.add_argument(
        "--correctness", required=True, metavar="CORRECTNESS_JSON",
        help="Path to correctness_results.json")
    sub_corr_report.add_argument(
        "--pytorch-version", required=True, metavar="VERSION",
        help="PyTorch version used for the sweep (e.g. 2.11.0)")
    sub_corr_report.add_argument(
        "--transformers-version", default="unknown", metavar="VERSION",
        help="Transformers version used for the sweep")

    sub_corr_apply = subparsers.add_parser(
        "correctness-apply",
        help="Apply a reviewed correctness plan to GitHub issues")
    sub_corr_apply.add_argument(
        "--plan", required=True, metavar="PLAN_JSON",
        help="Path to correctness-report.json plan file")

    sub_upstream = subparsers.add_parser(
        "pytorch-upstream",
        help="Build a self-contained pytorch/pytorch issue body or comment "
             "(read-only by default; --post creates issue / posts comment)")
    sub_upstream.add_argument(
        "--script", required=True, metavar="PATH",
        help="Path to the repro script (will be embedded verbatim in the body)")
    sub_upstream.add_argument(
        "--venv", required=True, action="append", metavar="NAME:PYTHON_BIN",
        help="Repeatable. Each spec is NAME:PATH_TO_PYTHON. "
             "Examples: pt211:~/envs/torch211/bin/python, "
             "pt212:~/envs/torch-nightly-cu128/bin/python")
    sub_upstream.add_argument(
        "--pythonpath", default=None, metavar="PATH[:PATH...]",
        help="Optional PYTHONPATH applied to every venv (for environments where "
             "transformers/diffusers/timm live as standalone modellibs trees rather "
             "than pip-installed). Colon-separated.")
    sub_upstream.add_argument(
        "--summary", default=None, metavar="MD_FILE",
        help="Markdown file with the bug description / context paragraph(s). "
             "Goes at the top of the issue body, above the Reproducer section.")
    sub_upstream.add_argument(
        "--title", default=None, metavar="STR",
        help="Issue title — required when posting a NEW issue (not used with --comment).")
    sub_upstream.add_argument(
        "--comment", type=int, default=None, metavar="ISSUE_NUMBER",
        help="If set, the assembled body is posted as a comment to this existing "
             "pytorch/pytorch issue instead of creating a new one. Mutually exclusive with --title.")
    sub_upstream.add_argument(
        "--label", action="append", default=None, metavar="LABEL",
        help="Repeatable. Labels applied when creating a new issue. Ignored with --comment.")
    sub_upstream.add_argument(
        "--output", default=None, metavar="PATH",
        help="Where to write the assembled body. Default: stdout.")
    sub_upstream.add_argument(
        "--no-collect-env", action="store_true",
        help="Skip running `python -m torch.utils.collect_env` in each venv. "
             "Only do this if the env disclosure is captured elsewhere.")
    sub_upstream.add_argument(
        "--no-run", action="store_true",
        help="Skip running the script in each venv (still runs torch version probe + collect_env). "
             "Use when scripts are slow and outputs are already known/captured.")
    sub_upstream.add_argument(
        "--timeout", type=int, default=1800, metavar="SECONDS",
        help="Per-venv script timeout (default: 1800s = 30 min).")
    sub_upstream.add_argument(
        "--post", action="store_true",
        help="Actually post to pytorch/pytorch. Default is dry-run (just write body to --output).")
    sub_upstream.add_argument(
        "--via-skill", required=True, metavar="CASE_ID",
        help="REQUIRED. Case ID from subagents/file-issue invocation (e.g. file-2026-05-08-191500-wav2vec2-ngb). "
             "Tool refuses to run without it. Validates: case file exists, mode_a_verdict in {proceed, proceed-with-fixes}, "
             "mode_b_sha256 non-empty. See subagents/file-issue/SKILL.md.")

    sub_corpus = subparsers.add_parser(
        "corpus-issue",
        help="Post a single issue to penguinwu/oss-model-graph-break-corpus. "
             "REQUIRES --via-skill (case_id from subagents/file-issue/).")
    sub_corpus.add_argument(
        "--via-skill", required=True, metavar="CASE_ID",
        help="REQUIRED. Case ID from subagents/file-issue invocation. "
             "Tool refuses to run without it. Validates: case file exists, "
             "mode_a_verdict in {proceed, proceed-with-fixes}, mode_b_sha256 non-empty, "
             "and body_sha256 matches the --body file content. See subagents/file-issue/SKILL.md.")
    sub_corpus.add_argument(
        "--body", required=True, metavar="BODY_MD",
        help="Path to the markdown body file (typically /tmp/file-issue-<case_id>-body.md). "
             "Content sha256 must match body_sha256 in the case file.")
    sub_corpus.add_argument(
        "--title", required=True, metavar="STR",
        help="Issue title (from Mode B's TITLE: line).")
    sub_corpus.add_argument(
        "--label", action="append", default=None, metavar="LABEL",
        help="Repeatable. Labels to apply to the issue (typically 'for:dynamo-team' etc.).")
    sub_corpus.add_argument(
        "--post", action="store_true",
        help="Actually post to GitHub. Default is dry-run (validates but does not POST).")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "sweep-report":
        cmd_sweep_report(args)
    elif args.command == "sweep-update":
        cmd_sweep_update(args)
    elif args.command == "correctness-report":
        cmd_correctness_report(args)
    elif args.command == "correctness-apply":
        # Deprecated 2026-05-08 per Peng directive: "do not lump many issues
        # that require different fixes into one umbrella." This path was the
        # umbrella-issue filer (filed N family-level issues from one plan).
        # Replaced by per-specific-fix flow via subagents/file-issue/.
        print("ERROR: `correctness-apply` is DEPRECATED as of 2026-05-08.\n"
              "  Reason: this command filed family-level umbrella issues, which "
              "violates Peng's directive that each issue maps to a SINGLE fix.\n"
              "  Replacement: file each correctness regression as its own issue "
              "via subagents/file-issue/ — see subagents/file-issue/SKILL.md.\n"
              "  For triage analysis (read-only), `correctness-report` still works.",
              file=sys.stderr)
        sys.exit(2)
    elif args.command == "pytorch-upstream":
        ok, err, _body = _validate_via_skill(args.via_skill, None)
        if not ok:
            print(err, file=sys.stderr)
            sys.exit(1)
        cmd_pytorch_upstream(args)
    elif args.command == "corpus-issue":
        cmd_corpus_issue(args)


if __name__ == "__main__":
    main()
