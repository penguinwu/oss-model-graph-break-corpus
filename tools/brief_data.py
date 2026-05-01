#!/usr/bin/env python3
"""Emit daily-brief data as JSON.

Pure data gatherer — does NOT compose prose. Composition is owned by the
`daily-briefing` skill (see `discovery/skills/daily-briefing/SKILL.md`).

Usage:
    python3 tools/brief_data.py                # JSON to stdout

Schema (top-level keys):
  today          str  ISO date
  yesterday      str  ISO date (today - 1)
  commits        list of {sha, msg}        — git log on corpus repo since yesterday 00:00 local
  closed_issues  list of {number, title, url, closed_at}  — repo issues closed in last 24h
  closed_loops   list of str               — bullets from OPEN-LOOPS Recently Closed (yesterday or today)
  plans          dict                      — passthrough of check_plan.py --json
  experiments    dict                      — passthrough of check_experiments.py --json
  backlog_aged   list of {number, title, url, age_days}  — Backlog cards aged >= BACKLOG_AGE_FLAG_DAYS
  open_loops     dict                      — {sections: {name: count}, needs_input: [{section, task, started}]}
  handoff        dict                      — {exists, mtime, first_lines} from HANDOFF.md
  latest_sweep_numeric dict                — most recent sweep's numeric_status stats; the
                                              skill surfaces only when age_hours <= 24 (sweeps
                                              are event-driven, not daily; brief stays quiet
                                              when no new sweep landed)

The skill should never invent fields beyond these. Empty lists / dicts mean
"nothing happened in that bucket" — emit accordingly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE_DIR = Path.home() / ".myclaw/spaces/AAQANraxXE4"
OPEN_LOOPS_PATH = SPACE_DIR / "OPEN-LOOPS.md"
HANDOFF_PATH = SPACE_DIR / "HANDOFF.md"
BACKLOG_AGE_FLAG_DAYS = 7


def run_check_plan() -> dict:
    out = subprocess.check_output(
        ["python3", str(REPO / "tools/check_plan.py"), "--json"], text=True
    )
    return json.loads(out)


def run_check_experiments() -> dict:
    res = subprocess.run(
        ["python3", str(REPO / "tools/check_experiments.py"), "--json"],
        capture_output=True, text=True, check=False,
    )
    return json.loads(res.stdout) if res.stdout else {"experiment_count": 0, "ok": [], "drift": []}


def gather_commits(since_iso: str) -> list[dict]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "log", f"--since={since_iso}", "--pretty=format:%h %s"],
            text=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    commits = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        sha, _, msg = line.partition(" ")
        commits.append({"sha": sha, "msg": msg})
    return commits


def gather_closed_issues(since_iso: str) -> list[dict]:
    sys.path.insert(0, str(REPO))
    try:
        from tools._gh_proxy import gh_get, REPO as REPO_NAME
    except Exception as e:
        return []
    try:
        items = gh_get(
            f"/repos/{REPO_NAME}/issues?state=closed&sort=updated&direction=desc&per_page=20"
        )
    except Exception:
        return []
    if not isinstance(items, list):
        return []
    cutoff = dt.datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=dt.timezone.utc)
    out = []
    for it in items:
        if "pull_request" in it:
            continue
        closed_at = it.get("closed_at")
        if not closed_at:
            continue
        when = dt.datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        if when < cutoff:
            break
        out.append({
            "number": it["number"],
            "title": it["title"],
            "url": it["html_url"],
            "closed_at": closed_at,
        })
    return out


def gather_closed_loops(yesterday: dt.date, today: dt.date) -> list[str]:
    if not OPEN_LOOPS_PATH.exists():
        return []
    text = OPEN_LOOPS_PATH.read_text()
    targets = {yesterday.isoformat(), today.isoformat()}
    bullets: list[str] = []
    capturing = False
    for line in text.splitlines():
        h = re.match(r"^##\s+Recently Closed\s+\((\d{4}-\d{2}-\d{2})\)", line)
        if h:
            capturing = h.group(1) in targets
            continue
        if line.startswith("## "):
            capturing = False
            continue
        if capturing and line.strip().startswith("- "):
            bullets.append(line.strip())
    return bullets


def gather_backlog_aged() -> list[dict]:
    sys.path.insert(0, str(REPO))
    try:
        from tools._gh_proxy import gh_graphql, PROJECT_NUMBER, PROJECT_OWNER
    except Exception:
        return []
    query = """
    query($login: String!, $number: Int!) {
      user(login: $login) {
        projectV2(number: $number) {
          items(first: 50) {
            nodes {
              content { ... on Issue { number title body url state } }
            }
          }
        }
      }
    }
    """
    try:
        data = gh_graphql(query, {"login": PROJECT_OWNER, "number": PROJECT_NUMBER})
    except Exception:
        return []
    items = data["user"]["projectV2"]["items"]["nodes"]
    today = dt.date.today()
    aged: list[dict] = []
    for it in items:
        c = it.get("content") or {}
        if c.get("state") != "OPEN":
            continue
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
            })
    aged.sort(key=lambda r: -r["age_days"])
    return aged


def parse_open_loops(path: Path) -> dict:
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
        m = re.match(r"^\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*$", line)
        if not m:
            continue
        task, type_, started, _notes = (g.strip() for g in m.groups())
        if task.lower() == "task" or set(task) <= {"-", " "}:
            continue
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


def gather_filed_issue_activity() -> dict:
    """Run check_filed_issues.py to detect new activity on Otter-tracked issues.

    Uses --no-update so the brief composition step doesn't race with the cron's
    state file (which check_filed_issues.py owns separately). Returns a dict with
    `new_activity_count`, `total_count`, and `issues` (list, only with new_activity=True).

    Failures are non-fatal: returns a stub if the script crashes (e.g., proxy down).
    """
    script = Path(__file__).parent / "check_filed_issues.py"
    if not script.exists():
        return {"new_activity_count": 0, "total_count": 0, "issues": [], "available": False}
    try:
        res = subprocess.run(
            ["python3", str(script), "--changes-only", "--no-update"],
            capture_output=True, text=True, timeout=60,
        )
        if res.returncode != 0:
            return {"new_activity_count": 0, "total_count": 0, "issues": [],
                    "available": False, "error": res.stderr.strip()[:200]}
        data = json.loads(res.stdout)
        return {
            "new_activity_count": data.get("new_activity_count", 0),
            "total_count": data.get("total_count", 0),
            "issues": data.get("issues", []),
            "available": True,
        }
    except Exception as e:
        return {"new_activity_count": 0, "total_count": 0, "issues": [],
                "available": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def read_handoff() -> dict:
    if not HANDOFF_PATH.exists():
        return {"exists": False}
    stat = HANDOFF_PATH.stat()
    mtime = dt.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    text = HANDOFF_PATH.read_text()
    head = "\n".join(text.splitlines()[:40])
    return {"exists": True, "mtime": mtime, "first_lines": head}


def gather_latest_sweep_numeric() -> dict:
    """Find the most recent sweep with identify_results.json and emit numeric_status stats.

    Sweeps are event-driven (not daily) — this gatherer always returns the latest
    sweep's numeric data along with mtime/age. The skill decides whether to surface
    based on age (new since last brief = age <= 24h is a reasonable default).

    Returns {available: False, ...reason} if no sweeps with numeric data exist.
    Returns {available: True, path, mtime, age_hours, sweep_label, status_counts,
             total_with_numeric, divergent_count, top_divergent[]} otherwise.
    """
    sweep_root = REPO / "sweep_results"
    if not sweep_root.exists():
        return {"available": False, "reason": "no_sweep_results_dir"}

    # Find all identify_results.json files; sort by mtime desc.
    candidates = list(sweep_root.rglob("identify_results.json"))
    if not candidates:
        return {"available": False, "reason": "no_identify_results_found"}
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Walk from newest until we find one with numeric_status fields.
    for path in candidates[:10]:  # cap at 10 most recent to avoid scanning ancient archives
        try:
            data = json.load(open(path))
            results = data.get("results", data) if isinstance(data, dict) else data
        except Exception:
            continue
        numeric = [r for r in results if isinstance(r, dict) and "numeric_status" in r]
        if not numeric:
            continue

        from collections import Counter
        statuses = Counter(r["numeric_status"] for r in numeric)
        divergent = [r for r in numeric
                     if r["numeric_status"] in ("divergence", "nan_inf_introduced",
                                                 "shape_mismatch", "dtype_mismatch")]
        divergent.sort(key=lambda r: r.get("numeric_severity_ratio", 0), reverse=True)

        mtime = path.stat().st_mtime
        age_hours = (dt.datetime.now().timestamp() - mtime) / 3600.0
        # sweep_label: parent dir relative to sweep_results
        try:
            label = str(path.parent.relative_to(sweep_root))
        except ValueError:
            label = path.parent.name

        return {
            "available": True,
            "path": str(path),
            "sweep_label": label,
            "mtime": dt.datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
            "age_hours": round(age_hours, 1),
            "total_with_numeric": len(numeric),
            "status_counts": dict(statuses),
            "divergent_count": len(divergent),
            "top_divergent": [
                {"name": r["name"], "variant": r.get("variant"),
                 "status": r["numeric_status"],
                 "max_diff": r.get("numeric_max_diff"),
                 "severity_ratio": r.get("numeric_severity_ratio"),
                 "first_divergence": r.get("numeric_first_divergence")}
                for r in divergent[:10]
            ],
        }

    return {"available": False, "reason": "no_recent_sweeps_with_numeric_status"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--since-iso", help="lookback timestamp (default: yesterday 00:00 local)")
    args = p.parse_args()

    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)
    if args.since_iso:
        since_iso = args.since_iso
    else:
        # yesterday 00:00 local
        since = dt.datetime.combine(yesterday, dt.time.min).astimezone()
        since_iso = since.isoformat()

    payload = {
        "today": today.isoformat(),
        "yesterday": yesterday.isoformat(),
        "since_iso": since_iso,
        "commits": gather_commits(since_iso),
        "closed_issues": gather_closed_issues(since_iso),
        "closed_loops": gather_closed_loops(yesterday, today),
        "plans": run_check_plan(),
        "experiments": run_check_experiments(),
        "backlog_aged": gather_backlog_aged(),
        "open_loops": parse_open_loops(OPEN_LOOPS_PATH),
        "filed_issue_activity": gather_filed_issue_activity(),
        "handoff": read_handoff(),
        "latest_sweep_numeric": gather_latest_sweep_numeric(),
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
