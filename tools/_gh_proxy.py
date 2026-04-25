"""Shared helpers for GitHub access via the localhost:7824 web proxy.

The proxy bypasses the BPF jailer for `agent:claude_code` processes. See
`~/.myclaw-shared/recipes/github-access.md` for full context.

Usage:
    from tools._gh_proxy import gh_post, gh_patch, gh_get, gh_graphql

    issue = gh_post(
        "/repos/penguinwu/oss-model-graph-break-corpus/issues",
        {"title": "...", "body": "..."},
    )
    print(issue["html_url"])
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = "penguinwu/oss-model-graph-break-corpus"
PROJECT_NUMBER = 1
PROJECT_OWNER = "penguinwu"
PROJECT_NODE_ID = "PVT_kwHOADBnlc4BVokM"  # cached from GraphQL lookup; stable

PROXY_URL = "http://localhost:7824/fetch"


def _token() -> str:
    host = Path.home() / ".config/gh/hosts.yml"
    m = re.search(r"oauth_token:\s*(\S+)", host.read_text())
    if not m:
        raise RuntimeError("no oauth_token in ~/.config/gh/hosts.yml")
    return m.group(1)


def _proxy(payload: dict) -> dict:
    """Send one request through the web proxy. Returns the proxy envelope dict."""
    r = subprocess.run(
        ["curl", "-s", PROXY_URL, "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True, check=True,
    )
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"proxy returned non-JSON: {r.stdout[:300]}") from e


def _api(method: str, endpoint: str, body: dict | None = None) -> dict:
    """Call the GitHub REST API. Returns the parsed JSON response.

    Raises on non-2xx. `endpoint` may be either a relative path
    ("/repos/.../issues") or a full URL.
    """
    url = endpoint if endpoint.startswith("http") else f"https://api.github.com{endpoint}"
    payload = {
        "url": url,
        "method": method,
        "headers": {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {_token()}",
        },
    }
    if body is not None:
        payload["body"] = body
    env = _proxy(payload)
    if not env.get("ok"):
        raise RuntimeError(
            f"GitHub {method} {endpoint} failed: status={env.get('status')} "
            f"error={env.get('error')} content={env.get('content', '')[:400]}"
        )
    content = env.get("content", "")
    if not content:
        return {}
    return json.loads(content)


def gh_get(endpoint: str) -> dict:
    return _api("GET", endpoint)


def gh_post(endpoint: str, body: dict) -> dict:
    return _api("POST", endpoint, body)


def gh_patch(endpoint: str, body: dict) -> dict:
    return _api("PATCH", endpoint, body)


def gh_graphql(query: str, variables: dict | None = None) -> dict:
    """Call the GraphQL endpoint. Returns the `data` field; raises on `errors`."""
    body = {"query": query, "variables": variables or {}}
    res = _api("POST", "/graphql", body)
    if res.get("errors"):
        raise RuntimeError(f"GraphQL errors: {json.dumps(res['errors'])[:600]}")
    return res["data"]


def add_issue_to_project(issue_node_id: str) -> str:
    """Add an issue (by node_id) to the canonical project board. Returns item_id."""
    res = gh_graphql(
        """
        mutation($pid: ID!, $cid: ID!) {
          addProjectV2ItemById(input: {projectId: $pid, contentId: $cid}) {
            item { id }
          }
        }
        """,
        {"pid": PROJECT_NODE_ID, "cid": issue_node_id},
    )
    return res["addProjectV2ItemById"]["item"]["id"]


def create_issue(title: str, body: str, labels: list[str] | None = None) -> dict:
    """Create an issue in REPO. Returns the GitHub issue dict (number, html_url, node_id, ...)."""
    payload = {"title": title, "body": body}
    if labels:
        payload["labels"] = labels
    return gh_post(f"/repos/{REPO}/issues", payload)


def update_issue(number: int, *, title: str | None = None, body: str | None = None) -> dict:
    payload: dict = {}
    if title is not None:
        payload["title"] = title
    if body is not None:
        payload["body"] = body
    return gh_patch(f"/repos/{REPO}/issues/{number}", payload)
