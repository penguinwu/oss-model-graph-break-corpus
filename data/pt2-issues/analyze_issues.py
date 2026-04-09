#!/usr/bin/env python3
"""Analyze oncall:pt2 issues for user journey patterns."""

import json
import re
from collections import Counter, defaultdict

with open("/home/pengwu/.myclaw/spaces/AAQANraxXE4/tmp/pt2_all_issues.json") as f:
    issues = json.load(f)

# Filter out bot/CI issues (DISABLED tests, flaky tests, dashboard)
def is_user_issue(issue):
    """Return True if this appears to be a real user-filed issue, not a bot/CI issue."""
    title = issue["title"]
    labels = issue["labels"]

    # Skip DISABLED test tracking issues
    if title.startswith("DISABLED "):
        return False
    # Skip dashboard/tracking issues
    if "Dashboard" in title and issue["comments"] > 100:
        return False
    # Skip if only flaky-tests label (usually bot-filed)
    if "module: flaky-tests" in labels and len([l for l in labels if l.startswith("module:")]) == 1:
        return False

    return True

user_issues = [i for i in issues if is_user_issue(i)]
bot_issues = [i for i in issues if not is_user_issue(i)]

print(f"Total issues: {len(issues)}")
print(f"User issues: {len(user_issues)}")
print(f"Bot/CI issues: {len(bot_issues)}")
print()

# --- CATEGORIZE BY USER JOURNEY TYPE ---

def categorize_journey(issue):
    """Categorize what the user was trying to do."""
    title = issue["title"].lower()
    body = (issue.get("body") or "").lower()
    labels = [l.lower() for l in issue["labels"]]
    text = title + " " + body

    journeys = []

    # torch.compile basic usage
    if "torch.compile" in text or "torchdynamo" in text or "dynamo" in title:
        journeys.append("torch.compile usage")

    # Inductor / code generation
    if "inductor" in text or "module: inductor" in labels or "triton" in text:
        journeys.append("inductor/codegen")

    # Dynamic shapes
    if "dynamic shape" in text or "dynamic_shape" in text or "symbool" in text or "symint" in text or "module: dynamic shapes" in labels or "unbacked" in text:
        journeys.append("dynamic shapes")

    # Export / torch.export
    if "export" in title or "torch.export" in text or "oncall: export" in labels:
        journeys.append("torch.export")

    # Graph breaks
    if "graph break" in text or "graph_break" in text or "unsupported" in text:
        journeys.append("graph breaks")

    # Performance optimization
    if "performance" in text or "slow" in text or "overhead" in text or "latency" in text or "max-autotune" in text or "max_autotune" in text or "module: performance" in labels:
        journeys.append("performance optimization")

    # CUDA graphs
    if "cuda graph" in text or "cuda_graph" in text or "cudagraph" in text or "module: cuda graphs" in labels:
        journeys.append("cuda graphs")

    # Custom ops / extensions
    if "custom op" in text or "custom_op" in text or "torch.library" in text or "custom kernel" in text:
        journeys.append("custom ops")

    # Distributed / FSDP / DDP
    if "distributed" in text or "fsdp" in text or "ddp" in text or "dtensor" in text or "oncall: distributed" in labels:
        journeys.append("distributed training")

    # Correctness / accuracy
    if "correctness" in text or "wrong result" in text or "accuracy" in text or "nan" in text or "module: correctness" in labels or "precision" in text:
        journeys.append("correctness issues")

    # Crashes / segfaults
    if "crash" in text or "segfault" in text or "sigsegv" in text or "module: crash" in labels or "assertion error" in text:
        journeys.append("crashes/segfaults")

    # Error messages (unclear)
    if "error" in title or "runtimeerror" in title or "exception" in title or "traceback" in body[:500]:
        journeys.append("error messages")

    # Regression
    if "regression" in text or "module: regression" in labels or "worked before" in text or "used to work" in text:
        journeys.append("version regression")

    # FlexAttention
    if "flexattention" in text or "flex_attention" in text or "flex attention" in text:
        journeys.append("flex attention")

    # AOTAutograd / AOT dispatch
    if "aot" in text or "aotautograd" in text or "module: aotdispatch" in labels:
        journeys.append("aot autograd")

    # Autograd with compile
    if "autograd" in text or "backward" in text or "gradient" in text:
        journeys.append("autograd + compile")

    # Quantization
    if "quantiz" in text or "quant" in title or "int8" in text or "float8" in text:
        journeys.append("quantization")

    # XPU / device porting
    if "xpu" in text or "module: xpu" in labels or "mps" in text:
        journeys.append("non-cuda devices")

    # Model-specific
    if any(m in text for m in ["huggingface", "transformers", "llama", "gpt", "bert", "resnet", "diffusion", "stable diffusion"]):
        journeys.append("specific model integration")

    if not journeys:
        journeys.append("other")

    return journeys

def categorize_pain_point(issue):
    """Categorize what went wrong."""
    title = issue["title"].lower()
    body = (issue.get("body") or "").lower()
    labels = [l.lower() for l in issue["labels"]]
    text = title + " " + body

    pains = []

    if "runtimeerror" in text or "error:" in title or "raises" in title or "fails" in title or "failed" in title:
        pains.append("runtime error")

    if "unexpected" in text or "wrong" in text or "incorrect" in text:
        pains.append("unexpected behavior")

    if "unclear" in text or "confusing" in text or "cryptic" in text or "obscure" in text:
        pains.append("unclear error message")

    if "documentation" in text or "docs" in text or "document" in text:
        pains.append("missing/unclear documentation")

    if "regression" in text or "module: regression" in labels or "broke" in text or "broken" in text:
        pains.append("version regression")

    if "segfault" in text or "sigsegv" in text or "crash" in text or "abort" in text:
        pains.append("crash/segfault")

    if "hang" in text or "deadlock" in text or "stuck" in text or "infinite" in text:
        pains.append("hang/deadlock")

    if "slow" in text or "overhead" in text or "performance" in text or "latency" in text:
        pains.append("performance degradation")

    if "nan" in text or "inf " in text or "accuracy" in text or "precision" in text:
        pains.append("numerical accuracy")

    if "memory" in text or "oom" in text or "out of memory" in text or "leak" in text:
        pains.append("memory issues")

    if "unsupported" in text or "not supported" in text or "not implemented" in text:
        pains.append("unsupported operation")

    if "internal" in text and "error" in text:
        pains.append("internal compiler error")

    if not pains:
        pains.append("other")

    return pains

# Count journeys and pain points
journey_counts = Counter()
pain_counts = Counter()
journey_pain_map = defaultdict(lambda: Counter())
journey_examples = defaultdict(list)

for issue in user_issues:
    journeys = categorize_journey(issue)
    pains = categorize_pain_point(issue)

    for j in journeys:
        journey_counts[j] += 1
        for p in pains:
            journey_pain_map[j][p] += 1
        if len(journey_examples[j]) < 5:
            journey_examples[j].append(issue)

    for p in pains:
        pain_counts[p] += 1

print("=" * 80)
print("USER JOURNEY PATTERNS (ranked by frequency)")
print("=" * 80)
for journey, count in journey_counts.most_common():
    pct = count / len(user_issues) * 100
    print(f"\n{journey}: {count} issues ({pct:.1f}%)")
    print(f"  Top pain points:")
    for pain, pc in journey_pain_map[journey].most_common(5):
        print(f"    - {pain}: {pc}")
    print(f"  Example issues:")
    for ex in journey_examples[journey][:3]:
        print(f"    #{ex['number']} [{ex['state']}] {ex['title'][:80]}")

print()
print("=" * 80)
print("PAIN POINT PATTERNS (ranked by frequency)")
print("=" * 80)
for pain, count in pain_counts.most_common():
    pct = count / len(user_issues) * 100
    print(f"  {pain}: {count} ({pct:.1f}%)")

# --- Resolution analysis ---
print()
print("=" * 80)
print("RESOLUTION ANALYSIS")
print("=" * 80)
open_issues = [i for i in user_issues if i["state"] == "open"]
closed_issues = [i for i in user_issues if i["state"] == "closed"]
print(f"Open: {len(open_issues)} ({len(open_issues)/len(user_issues)*100:.1f}%)")
print(f"Closed: {len(closed_issues)} ({len(closed_issues)/len(user_issues)*100:.1f}%)")

# High-engagement open issues (likely important user needs)
print()
print("HIGH-ENGAGEMENT OPEN ISSUES (comments >= 5 or reactions >= 2):")
high_engage = sorted(
    [i for i in open_issues if i["comments"] >= 5 or i["reactions"] >= 2],
    key=lambda x: x["comments"] + x["reactions"] * 3,
    reverse=True,
)
for i in high_engage[:20]:
    journeys = categorize_journey(i)
    print(f"  #{i['number']} ({i['comments']}c, {i['reactions']}r) {', '.join(journeys)}")
    print(f"    {i['title'][:90]}")

# --- Documentation gap analysis ---
print()
print("=" * 80)
print("DOCUMENTATION GAP SIGNALS")
print("=" * 80)

# Issues where users explicitly mention docs
doc_mentions = []
for issue in user_issues:
    text = (issue["title"] + " " + (issue.get("body") or "")).lower()
    if any(w in text for w in ["documentation", "docs ", "document ", "how to", "tutorial", "example", "guide"]):
        doc_mentions.append(issue)

print(f"\nIssues mentioning docs/how-to: {len(doc_mentions)}")
for i in doc_mentions[:10]:
    print(f"  #{i['number']} [{i['state']}] {i['title'][:80]}")

# Issues with "needs reproduction" — often user confusion
needs_repro = [i for i in user_issues if "needs reproduction" in i["labels"]]
print(f"\nIssues needing reproduction (possible user confusion): {len(needs_repro)}")

# Feature requests vs bugs
feature_issues = [i for i in user_issues if "feature" in i["labels"]]
print(f"\nFeature requests: {len(feature_issues)}")
for i in feature_issues[:10]:
    print(f"  #{i['number']} [{i['state']}] {i['title'][:80]}")

# Issues about error messages specifically
error_msg_issues = [i for i in user_issues if any(w in (issue["title"] + " " + (issue.get("body") or "")).lower() for w in ["unclear error", "confusing error", "cryptic", "obscure error", "better error"])]
print(f"\nIssues about error message quality: {len(error_msg_issues)}")

# Save detailed analysis for the report
analysis = {
    "total_issues": len(issues),
    "user_issues": len(user_issues),
    "bot_issues": len(bot_issues),
    "journey_counts": dict(journey_counts.most_common()),
    "pain_counts": dict(pain_counts.most_common()),
    "open_count": len(open_issues),
    "closed_count": len(closed_issues),
}

with open("/home/pengwu/.myclaw/spaces/AAQANraxXE4/tmp/pt2_analysis_summary.json", "w") as f:
    json.dump(analysis, f, indent=2)
