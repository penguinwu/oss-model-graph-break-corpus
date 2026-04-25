#!/usr/bin/env python3
"""Generate daily brief from plans + OPEN-LOOPS.md + experiments + Backlog and post to GChat.

Sources:
  - tools/check_plan.py output (plan files in this repo)
  - tools/check_experiments.py output (discovery experiment convention)
  - ~/.myclaw/spaces/AAQANraxXE4/OPEN-LOOPS.md (Otter's open loops)
  - GitHub project board #1 Backlog column (queued tasks aged > N days)

Output: a GChat-friendly message (no markdown tables, blank-line separators,
single-asterisk italics per Otter's GChat formatting rules).

Usage:
    python3 tools/daily_brief.py                    # print to stdout
    python3 tools/daily_brief.py --post             # post to Otter's space
    python3 tools/daily_brief.py --space SPACE_ID   # post to a specific space
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OPEN_LOOPS_PATH = Path.home() / ".myclaw/spaces/AAQANraxXE4/OPEN-LOOPS.md"
OTTER_SPACE = "AAQANraxXE4"
NEEDS_INPUT_AGE_FLAG_DAYS = 2  # call out needs-input items older than this
BACKLOG_AGE_FLAG_DAYS = 7  # call out Backlog cards older than this


def run_check_plan() -> dict:
    import json as _json
    out = subprocess.check_output(
        ["python3", str(REPO / "tools/check_plan.py"), "--json"], text=True
    )
    return _json.loads(out)


def run_check_experiments() -> dict:
    import json as _json
    res = subprocess.run(
        ["python3", str(REPO / "tools/check_experiments.py"), "--json"],
        capture_output=True, text=True, check=False,
    )
    # Returns nonzero on drift; parse stdout regardless.
    return _json.loads(res.stdout) if res.stdout else {"experiment_count": 0, "ok": [], "drift": []}


def fetch_backlog_aged() -> list[dict]:
    """Pull Backlog board items, parse `**Queued at:**` from body, return list of aged items."""
    sys.path.insert(0, str(REPO))
    try:
        from tools._gh_proxy import gh_graphql, PROJECT_NUMBER, PROJECT_OWNER
    except Exception as e:
        return [{"_error": f"_gh_proxy import failed: {e}"}]
    query = """
    query($login: String!, $number: Int!) {
      user(login: $login) {
        projectV2(number: $number) {
          items(first: 50) {
            nodes {
              id
              fieldValues(first: 20) {
                nodes {
                  ... on ProjectV2ItemFieldSingleSelectValue {
                    name
                    field { ... on ProjectV2SingleSelectField { name } }
                  }
                }
              }
              content {
                ... on Issue {
                  number title body url state
                }
              }
            }
          }
        }
      }
    }
    """
    try:
        data = gh_graphql(query, {"login": PROJECT_OWNER, "number": PROJECT_NUMBER})
    except Exception as e:
        return [{"_error": f"GraphQL failed: {e}"}]
    items = data["user"]["projectV2"]["items"]["nodes"]
    today = dt.date.today()
    aged: list[dict] = []
    for it in items:
        c = it.get("content") or {}
        if c.get("state") != "OPEN":
            continue
        # Find Status field value, if present.
        status = None
        for fv in (it.get("fieldValues", {}).get("nodes") or []):
            if fv and fv.get("field", {}).get("name", "").lower() == "status":
                status = fv.get("name")
        # Default-column items often have no status set; treat unset as Backlog.
        # We rely on body marker `**Queued at:** YYYY-MM-DD HH:MM UTC` to age.
        body = c.get("body") or ""
        m = re.search(r"\*\*Queued at:\*\*\s*(\d{4}-\d{2}-\d{2})", body)
        if not m:
            continue
        queued = dt.date.fromisoformat(m.group(1))
        age = (today - queued).days
        if age >= BACKLOG_AGE_FLAG_DAYS:
            aged.append({
                "number": c.get("number"),
                "title": c.get("title"),
                "url": c.get("url"),
                "age_days": age,
                "status": status or "Backlog (unset)",
            })
    aged.sort(key=lambda r: -r["age_days"])
    return aged


def parse_open_loops(path: Path) -> dict:
    """Extract section task counts + needs-input items from OPEN-LOOPS.md."""
    if not path.exists():
        return {"sections": {}, "needs_input": []}
    text = path.read_text()
    sections: dict[str, int] = {}
    needs_input: list[dict] = []
    current_section = None
    in_recently_closed = False

    for line in text.splitlines():
        h = re.match(r"^##\s+(.+?)\s*$", line)
        if h:
            heading = h.group(1)
            if heading.startswith("Recently Closed") or heading.startswith("Stale Loop Audit"):
                in_recently_closed = True
                current_section = None
            else:
                in_recently_closed = False
                current_section = heading
                sections.setdefault(current_section, 0)
            continue
        if in_recently_closed or current_section is None:
            continue
        # Match table rows: | task | type | started | notes |
        m = re.match(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if not m:
            continue
        task, type_, started, _notes = (g.strip() for g in m.groups())
        # Skip header + separator rows
        if task.lower() == "task" or set(task) <= {"-", " "}:
            continue
        # Skip CLOSED rows
        if type_.upper() == "CLOSED":
            continue
        sections[current_section] += 1
        if type_ == "needs-input":
            needs_input.append({
                "section": current_section,
                "task": task.strip("*").strip(),
                "started": started,
            })
    return {"sections": sections, "needs_input": needs_input}


def days_since(date_str: str, today: dt.date) -> int | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", date_str or "")
    if not m:
        return None
    try:
        d = dt.date.fromisoformat(m.group(0))
    except ValueError:
        return None
    return (today - d).days


def short_section(name: str) -> str:
    """Trim 'WS1: Skill evaluation via corpus' → 'WS1'."""
    m = re.match(r"^(WS\d+|Standalone[^:]*)", name)
    return m.group(1) if m else name


def render(
    plan_report: dict,
    loops: dict,
    experiments: dict,
    backlog_aged: list[dict],
    today: dt.date,
) -> str:
    lines = [f"[🦦 Otter] Daily brief — {today.isoformat()}", ""]

    # Plans
    lines.append("*Plans*")
    lines.append("")
    if not plan_report["plans"]:
        lines.append("(no plan files found — discovery glob mismatch?)")
    else:
        for p in plan_report["plans"]:
            tag = "STALE" if p["stale"] else "ok"
            age = "?" if p["age_days"] is None else f"{p['age_days']}d"
            lines.append(f"{tag}  {p['name']}  ({age})")
    lines.append("")

    # Experiment convention drift
    drift = experiments.get("drift", [])
    lines.append(f"*Experiment convention* ({experiments.get('experiment_count', 0)} experiments, {len(drift)} drift)")
    lines.append("")
    if not drift:
        lines.append("(all experiments conform)")
    else:
        for d in drift:
            lines.append(f"DRIFT  {d.get('slug', d.get('item'))}")
            for p in (d.get("problems") or [])[:3]:
                lines.append(f"   {p}")
    lines.append("")

    # Aged Backlog cards
    lines.append(f"*Backlog aged > {BACKLOG_AGE_FLAG_DAYS}d* ({len(backlog_aged)})")
    lines.append("")
    if not backlog_aged:
        lines.append("(none — queue moves)")
    else:
        for b in backlog_aged[:10]:
            if "_error" in b:
                lines.append(f"(board fetch failed: {b['_error']})")
                break
            lines.append(f"#{b['number']} ({b['age_days']}d) — {b['title']}")
    lines.append("")

    # Awaiting your input
    needs = loops["needs_input"]
    lines.append(f"*Awaiting your input* ({len(needs)})")
    lines.append("")
    if not needs:
        lines.append("(none — queue is clear)")
    else:
        for n in needs:
            age = days_since(n["started"], today)
            age_str = "?" if age is None else f"{age}d"
            flag = " ←" if (age is not None and age >= NEEDS_INPUT_AGE_FLAG_DAYS) else ""
            lines.append(f"{short_section(n['section'])} — {n['task']} ({age_str}){flag}")
    lines.append("")

    # Section summary (count only)
    lines.append("*Open loops by section*")
    lines.append("")
    if not loops["sections"]:
        lines.append("(no sections found)")
    else:
        for sec, count in loops["sections"].items():
            if count == 0:
                continue
            lines.append(f"{short_section(sec)}: {count}")
    lines.append("")

    # Footer
    lines.append("—")
    lines.append("Source: check_plan.py + check_experiments.py + OPEN-LOOPS.md + project board #1")
    return "\n".join(lines).rstrip() + "\n"


def post_to_gchat(space_id: str, body: str) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        result = subprocess.run(
            ["gchat", "send", space_id, "--as-bot", "--quiet", "--text-file", path],
            check=False,
        )
        return result.returncode
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--post", action="store_true", help="post to Otter's GChat space")
    parser.add_argument("--space", default=OTTER_SPACE, help="GChat space ID")
    args = parser.parse_args()

    plan_report = run_check_plan()
    loops = parse_open_loops(OPEN_LOOPS_PATH)
    experiments = run_check_experiments()
    backlog_aged = fetch_backlog_aged()
    body = render(plan_report, loops, experiments, backlog_aged, dt.date.today())

    if args.post:
        rc = post_to_gchat(args.space, body)
        if rc != 0:
            print(f"gchat send failed (rc={rc})", file=sys.stderr)
            return rc
        print(f"posted {len(body)} chars to {args.space}", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
