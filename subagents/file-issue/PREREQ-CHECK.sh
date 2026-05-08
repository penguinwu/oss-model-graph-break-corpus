#!/usr/bin/env bash
# PREREQ-CHECK.sh — verify all dependencies for the file-issue subagent skill.
# Exit 0 = ready. Exit 1 = at least one prerequisite missing (stderr names which).
#
# Run before first invocation, and any time the skill complains about missing infra.
# Per subagents/file-issue/SKILL.md Pre-requisites section.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLAUDE_MD="${HOME}/.myclaw/spaces/AAQANraxXE4/CLAUDE.md"

failed=0

check() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  [OK]   $name"
    else
        echo "  [MISS] $name" >&2
        failed=$((failed + 1))
    fi
}

echo "file-issue prereq check (REPO_ROOT=$REPO_ROOT):"

# 1. Local CLAUDE.md trigger present
check "local CLAUDE.md trigger for file-issue" \
    "grep -q 'subagents/file-issue/SKILL.md' '$CLAUDE_MD'"

# 2. tools/file_issues.py has --via-skill required on corpus-issue
check "tools/file_issues.py corpus-issue --via-skill required" \
    "python3 '$REPO_ROOT/tools/file_issues.py' corpus-issue --help 2>&1 | grep -q 'via-skill.*required\\|required.*via-skill'"

# 3. tools/file_issues.py has --via-skill required on pytorch-upstream
check "tools/file_issues.py pytorch-upstream --via-skill required" \
    "python3 '$REPO_ROOT/tools/file_issues.py' pytorch-upstream --help 2>&1 | grep -q 'via-skill'"

# 4. tools/build_invocations_log.py exists + runs
check "tools/build_invocations_log.py available" \
    "python3 '$REPO_ROOT/tools/build_invocations_log.py' --help"

# 5. GitHub auth (only matters if actually posting; warn rather than block)
if command -v gh >/dev/null 2>&1; then
    if gh auth status >/dev/null 2>&1; then
        echo "  [OK]   gh auth configured"
    else
        echo "  [WARN] gh auth not configured (only matters at post time)" >&2
    fi
else
    echo "  [WARN] gh CLI not on PATH (only matters at post time)" >&2
fi

# 6. Web proxy (only matters for pytorch upstream; warn)
if curl -sf -m 3 -x http://localhost:7824 https://api.github.com/zen >/dev/null 2>&1; then
    echo "  [OK]   web proxy at localhost:7824 reachable"
else
    echo "  [WARN] web proxy at localhost:7824 unreachable (only matters for pytorch upstream)" >&2
fi

# 7. invocations/ dir exists
check "subagents/file-issue/invocations/ dir" \
    "test -d '$REPO_ROOT/subagents/file-issue/invocations'"

# 8. persona.md exists
check "subagents/file-issue/persona.md present" \
    "test -f '$REPO_ROOT/subagents/file-issue/persona.md'"

if [ "$failed" -gt 0 ]; then
    echo
    echo "FAIL: $failed prerequisite(s) missing. Address above before invoking the skill." >&2
    exit 1
fi

echo
echo "PASS: all prerequisites met."
exit 0
