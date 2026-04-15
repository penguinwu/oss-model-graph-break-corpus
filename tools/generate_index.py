#!/usr/bin/env python3
"""Generate a browsable corpus dashboard as docs/index.html.

Shows all 468 models in a filterable, sortable table with status, break
counts, root cause categories, actionability, and links to tlparse reports.

Usage:
    python3 tools/generate_index.py                      # default output: docs/index.html
    python3 tools/generate_index.py --output dashboard.html
"""
import argparse
import html
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from analyze_explain import classify_reason, ACTIONABILITY, ACTIONABILITY_LABELS


# Simplified fixability labels for the dashboard
FIXABILITY = {
    "fixable_in_user_code": "Easy",
    "needs_library_pr": "Medium",
    "needs_compiler_change": "Hard",
    "needs_investigation": "?",
}

FIXABILITY_COLOR = {
    "Easy": "#28a745",
    "Medium": "#f0ad4e",
    "Hard": "#dc3545",
    "?": "#999",
}

STATUS_COLOR = {
    "full_graph": "#28a745",
    "graph_break": "#dc3545",
    "eager_error": "#f0ad4e",
    "create_error": "#999",
    "worker_error": "#999",
    "timeout": "#999",
}


def load_corpus():
    path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
    with open(path) as f:
        return json.load(f)


def load_explain(version="pt2.10"):
    path = os.path.join(REPO_ROOT, "sweep_results", version, "explain_results.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    info = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        key = f"{r['name']}_{r.get('mode', 'eval')}"
        reasons = r.get("break_reasons", [])
        categories = [classify_reason(br.get("reason", "")) for br in reasons]
        primary = categories[0] if categories else ""
        # Get actionability
        action_level = ACTIONABILITY.get(primary, ("needs_investigation", ""))[0] if primary else ""
        info[key] = {
            "break_count": r.get("graph_break_count", 0),
            "graph_count": r.get("graph_count", 0),
            "primary_category": primary,
            "actionability": action_level,
            "fixability": FIXABILITY.get(action_level, ""),
        }
    return info


def check_trace_exists(model_name, mode, docs_dir):
    """Check if a pre-generated tlparse report exists."""
    trace_dir = os.path.join(docs_dir, "traces", f"{model_name}_{mode}")
    return os.path.exists(os.path.join(trace_dir, "index.html"))


def generate_dashboard(corpus, explain_info, output_path):
    docs_dir = os.path.dirname(output_path)
    rows = []

    for m in corpus["models"]:
        name = m["name"]
        source = m["source"]

        for mode in ["eval", "train"]:
            data = m.get(mode, {})
            status = data.get("status", "unknown")
            break_count = data.get("graph_break_count", 0)

            key = f"{name}_{mode}"
            explain = explain_info.get(key, {})
            category = explain.get("primary_category", "")
            fixability = explain.get("fixability", "")

            # Check for tlparse report
            has_trace = check_trace_exists(name, mode, docs_dir)
            trace_link = f'<a href="traces/{html.escape(name)}_{html.escape(mode)}/index.html">view</a>' if has_trace else ""

            status_color = STATUS_COLOR.get(status, "#333")
            fix_color = FIXABILITY_COLOR.get(fixability, "#333")

            rows.append({
                "name": name,
                "source": source,
                "mode": mode,
                "status": status,
                "status_color": status_color,
                "break_count": break_count,
                "category": category,
                "fixability": fixability,
                "fix_color": fix_color,
                "trace_link": trace_link,
            })

    # Stats
    models = corpus["models"]
    total = len(models)
    graph_break_models = sum(1 for m in models if m.get("has_graph_break"))
    full_graph_eval = sum(1 for m in models if m.get("eval", {}).get("status") == "full_graph")
    full_graph_train = sum(1 for m in models if m.get("train", {}).get("status") == "full_graph")

    table_rows = []
    for r in rows:
        table_rows.append(f"""        <tr>
            <td>{html.escape(r['name'])}</td>
            <td>{html.escape(r['source'])}</td>
            <td>{html.escape(r['mode'])}</td>
            <td><span style="color:{r['status_color']}">{html.escape(r['status'])}</span></td>
            <td>{r['break_count'] if r['break_count'] else ''}</td>
            <td>{html.escape(r['category'])}</td>
            <td><span style="color:{r['fix_color']}">{html.escape(r['fixability'])}</span></td>
            <td>{r['trace_link']}</td>
        </tr>""")

    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>OSS Model Compiler Quality Corpus</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 2em; color: #333; }}
        h1 {{ margin-bottom: 0.2em; }}
        .subtitle {{ color: #666; margin-bottom: 1.5em; }}
        .stats {{ display: flex; gap: 2em; margin-bottom: 1.5em; flex-wrap: wrap; }}
        .stat {{ background: #f8f9fa; border-radius: 8px; padding: 1em 1.5em; }}
        .stat-value {{ font-size: 1.8em; font-weight: 700; }}
        .stat-label {{ color: #666; font-size: 0.85em; }}
        .controls {{ display: flex; gap: 1em; margin-bottom: 1em; flex-wrap: wrap; align-items: center; }}
        input, select {{ padding: 6px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 0.9em; }}
        input {{ width: 250px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
        th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; white-space: nowrap; }}
        th {{ background: #f5f5f5; font-weight: 600; position: sticky; top: 0; cursor: pointer; user-select: none; }}
        th:hover {{ background: #e8e8e8; }}
        tr:hover {{ background: #f9f9f9; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .count {{ color: #666; font-size: 0.85em; margin-left: 0.5em; }}
    </style>
</head>
<body>
    <h1>OSS Model Compiler Quality Corpus</h1>
    <p class="subtitle">{total} models &middot; PyTorch 2.10.0 &middot; torch.compile(fullgraph=True)</p>

    <div class="stats">
        <div class="stat"><div class="stat-value">{total}</div><div class="stat-label">Total Models</div></div>
        <div class="stat"><div class="stat-value" style="color:#28a745">{full_graph_eval}</div><div class="stat-label">Full Graph (eval)</div></div>
        <div class="stat"><div class="stat-value" style="color:#28a745">{full_graph_train}</div><div class="stat-label">Full Graph (train)</div></div>
        <div class="stat"><div class="stat-value" style="color:#dc3545">{graph_break_models}</div><div class="stat-label">Models w/ Breaks</div></div>
    </div>

    <div class="controls">
        <input type="text" id="filter" placeholder="Filter by model name..." onkeyup="filterTable()">
        <select id="statusFilter" onchange="filterTable()">
            <option value="">All statuses</option>
            <option value="full_graph">full_graph</option>
            <option value="graph_break">graph_break</option>
            <option value="eager_error">eager_error</option>
            <option value="create_error">create_error</option>
        </select>
        <select id="modeFilter" onchange="filterTable()">
            <option value="">All modes</option>
            <option value="eval">eval</option>
            <option value="train">train</option>
        </select>
        <select id="fixFilter" onchange="filterTable()">
            <option value="">All fixability</option>
            <option value="Easy">Easy</option>
            <option value="Medium">Medium</option>
            <option value="Hard">Hard</option>
        </select>
        <span class="count" id="rowCount"></span>
    </div>

    <table id="corpus">
        <thead>
            <tr>
                <th onclick="sortTable(0)">Model</th>
                <th onclick="sortTable(1)">Source</th>
                <th onclick="sortTable(2)">Mode</th>
                <th onclick="sortTable(3)">Status</th>
                <th onclick="sortTable(4)">Breaks</th>
                <th onclick="sortTable(5)">Root Cause</th>
                <th onclick="sortTable(6)">Fixable?</th>
                <th>Trace</th>
            </tr>
        </thead>
        <tbody>
{chr(10).join(table_rows)}
        </tbody>
    </table>

    <script>
        function filterTable() {{
            const q = document.getElementById('filter').value.toLowerCase();
            const status = document.getElementById('statusFilter').value;
            const mode = document.getElementById('modeFilter').value;
            const fix = document.getElementById('fixFilter').value;
            const rows = document.querySelectorAll('#corpus tbody tr');
            let visible = 0;
            rows.forEach(row => {{
                const cells = row.querySelectorAll('td');
                const name = cells[0].textContent.toLowerCase();
                const rowStatus = cells[3].textContent;
                const rowMode = cells[2].textContent;
                const rowFix = cells[6].textContent;
                const show = name.includes(q)
                    && (!status || rowStatus === status)
                    && (!mode || rowMode === mode)
                    && (!fix || rowFix === fix);
                row.style.display = show ? '' : 'none';
                if (show) visible++;
            }});
            document.getElementById('rowCount').textContent = visible + ' rows';
        }}

        let sortDir = {{}};
        function sortTable(col) {{
            const table = document.getElementById('corpus');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const dir = sortDir[col] = !(sortDir[col] || false);
            rows.sort((a, b) => {{
                let va = a.querySelectorAll('td')[col].textContent;
                let vb = b.querySelectorAll('td')[col].textContent;
                const na = parseFloat(va), nb = parseFloat(vb);
                if (!isNaN(na) && !isNaN(nb)) return dir ? na - nb : nb - na;
                return dir ? va.localeCompare(vb) : vb.localeCompare(va);
            }});
            rows.forEach(row => tbody.appendChild(row));
        }}

        // Initial count
        filterTable();
    </script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(index_html)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate corpus dashboard HTML page")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "docs", "index.html"),
                        help="Output path (default: docs/index.html)")
    parser.add_argument("--version", default="pt2.10",
                        help="PyTorch version for explain data (default: pt2.10)")
    args = parser.parse_args()

    print("Loading corpus data...")
    corpus = load_corpus()
    print(f"  {len(corpus['models'])} models")

    print("Loading explain results...")
    explain_info = load_explain(args.version)
    print(f"  {len(explain_info)} model-mode entries")

    print("Generating dashboard...")
    path = generate_dashboard(corpus, explain_info, args.output)
    print(f"Dashboard written to {path}")


if __name__ == "__main__":
    main()
