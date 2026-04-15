#!/usr/bin/env python3
"""Batch-generate tlparse HTML reports for all trace directories.

Runs tlparse on each trace dir in sweep_results/<version>/traces/ and
generates an index page linking to all reports.

Usage:
    python3 tools/generate_trace_reports.py                          # default: v2.10
    python3 tools/generate_trace_reports.py --version v2.9
    python3 tools/generate_trace_reports.py --output-dir /tmp/reports
    python3 tools/generate_trace_reports.py --skip-existing          # resume interrupted run
"""
import argparse
import html
import json
import os
import subprocess
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def get_corpus_info():
    """Load corpus.json for break counts and categories."""
    corpus_path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
    info = {}
    if not os.path.exists(corpus_path):
        return info
    with open(corpus_path) as f:
        corpus = json.load(f)
    for m in corpus["models"]:
        for mode in ["eval", "train"]:
            data = m.get(mode, {})
            key = f"{m['name']}_{mode}"
            info[key] = {
                "model": m["name"],
                "mode": mode,
                "status": data.get("status", "unknown"),
                "graph_break_count": data.get("graph_break_count", 0),
                "fullgraph_error": data.get("fullgraph_error", ""),
            }
    return info


def get_explain_info(version):
    """Load explain results for break reason categories."""
    explain_path = os.path.join(REPO_ROOT, "sweep_results", version, "explain_results.json")
    info = {}
    if not os.path.exists(explain_path):
        return info
    with open(explain_path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    # Import classify_reason from analyze_explain
    sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
    from analyze_explain import classify_reason
    for r in results:
        if r.get("status") != "ok":
            continue
        key = f"{r['name']}_{r.get('mode', 'eval')}"
        reasons = r.get("break_reasons", [])
        categories = [classify_reason(br.get("reason", "")) for br in reasons]
        # Deduplicate and take top category
        primary = categories[0] if categories else "unknown"
        info[key] = {
            "break_count": r.get("graph_break_count", 0),
            "graph_count": r.get("graph_count", 0),
            "primary_category": primary,
            "categories": list(dict.fromkeys(categories)),  # unique, order preserved
        }
    return info


def run_tlparse(trace_dir, output_dir):
    """Run tlparse on a single trace directory. Returns True on success."""
    try:
        result = subprocess.run(
            ["tlparse", "parse", trace_dir, "-o", output_dir, "--overwrite"],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  ERROR: {e}")
        return False


def generate_index(reports_dir, report_results, corpus_info, explain_info):
    """Generate an HTML index page linking to all reports."""
    # Group by model
    models = defaultdict(dict)
    for trace_name, success in report_results.items():
        parts = trace_name.rsplit("_", 1)
        if len(parts) == 2:
            model_name, mode = parts
        else:
            model_name, mode = trace_name, "eval"
        models[model_name][mode] = {
            "success": success,
            "trace_name": trace_name,
            "corpus": corpus_info.get(trace_name, {}),
            "explain": explain_info.get(trace_name, {}),
        }

    # Sort by total break count (descending)
    def sort_key(item):
        total = 0
        for mode_data in item[1].values():
            total += mode_data.get("explain", {}).get("break_count", 0)
        return -total

    sorted_models = sorted(models.items(), key=sort_key)

    # Generate HTML
    rows = []
    for model_name, modes in sorted_models:
        for mode in ["eval", "train"]:
            if mode not in modes:
                continue
            data = modes[mode]
            trace_name = data["trace_name"]
            corpus = data.get("corpus", {})
            explain = data.get("explain", {})
            status = corpus.get("status", "—")
            breaks = explain.get("break_count", corpus.get("graph_break_count", 0))
            category = explain.get("primary_category", "—")

            if data["success"]:
                link = f'<a href="{html.escape(trace_name)}/index.html">{html.escape(model_name)}</a>'
            else:
                link = f'<span style="color:#999">{html.escape(model_name)} (failed)</span>'

            rows.append(f"""        <tr>
            <td>{link}</td>
            <td>{html.escape(mode)}</td>
            <td>{breaks}</td>
            <td>{html.escape(category)}</td>
            <td>{html.escape(status)}</td>
        </tr>""")

    total = len(report_results)
    ok = sum(1 for v in report_results.values() if v)
    failed = total - ok

    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Graph Break Trace Reports</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2em; }}
        h1 {{ color: #333; }}
        .summary {{ color: #666; margin-bottom: 1em; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
        th {{ background: #f5f5f5; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #f9f9f9; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        input {{ padding: 6px 12px; width: 300px; margin-bottom: 1em; border: 1px solid #ddd; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>Graph Break Trace Reports</h1>
    <p class="summary">{ok} reports generated, {failed} failed, {total} total trace directories.</p>
    <input type="text" id="filter" placeholder="Filter by model name..." onkeyup="filterTable()">
    <table id="reports">
        <thead>
            <tr><th>Model</th><th>Mode</th><th>Breaks</th><th>Primary Category</th><th>Status</th></tr>
        </thead>
        <tbody>
{chr(10).join(rows)}
        </tbody>
    </table>
    <script>
        function filterTable() {{
            const q = document.getElementById('filter').value.toLowerCase();
            const rows = document.querySelectorAll('#reports tbody tr');
            rows.forEach(row => {{
                row.style.display = row.textContent.toLowerCase().includes(q) ? '' : 'none';
            }});
        }}
    </script>
</body>
</html>"""

    index_path = os.path.join(reports_dir, "index.html")
    with open(index_path, "w") as f:
        f.write(index_html)
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate tlparse HTML reports for trace directories",
    )
    parser.add_argument("--version", default="pt2.10",
                        help="PyTorch version directory (default: pt2.10)")
    parser.add_argument("--output-dir",
                        help="Output directory for reports (default: sweep_results/<version>/trace_reports)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip trace dirs that already have reports")
    args = parser.parse_args()

    traces_dir = os.path.join(REPO_ROOT, "sweep_results", args.version, "traces")
    if not os.path.isdir(traces_dir):
        print(f"ERROR: Traces directory not found: {traces_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(REPO_ROOT, "sweep_results", args.version, "trace_reports")
    os.makedirs(output_dir, exist_ok=True)

    trace_dirs = sorted(d for d in os.listdir(traces_dir)
                        if os.path.isdir(os.path.join(traces_dir, d)))

    print(f"Traces directory: {traces_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Trace directories: {len(trace_dirs)}")
    print()

    # Load metadata for the index page
    corpus_info = get_corpus_info()
    explain_info = get_explain_info(args.version)

    report_results = {}
    for i, trace_name in enumerate(trace_dirs, 1):
        trace_path = os.path.join(traces_dir, trace_name)
        report_path = os.path.join(output_dir, trace_name)

        if args.skip_existing and os.path.exists(os.path.join(report_path, "index.html")):
            print(f"[{i}/{len(trace_dirs)}] {trace_name} — skipped (exists)")
            report_results[trace_name] = True
            continue

        print(f"[{i}/{len(trace_dirs)}] {trace_name} ...", end=" ", flush=True)
        success = run_tlparse(trace_path, report_path)
        report_results[trace_name] = success
        print("OK" if success else "FAILED")

    # Generate index page
    print()
    index_path = generate_index(output_dir, report_results, corpus_info, explain_info)

    ok = sum(1 for v in report_results.values() if v)
    failed = sum(1 for v in report_results.values() if not v)
    print(f"Done: {ok} reports generated, {failed} failed")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
