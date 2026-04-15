#!/usr/bin/env python3
"""Feedback Space Monitor — triage new messages from the GChat feedback space.

Reads new messages from the Graph Break Corpus Feedback space, classifies
them, and logs to audit. The cron prompt handles responses directly.

No Task creation, no Rocky delegation — Otter handles everything end-to-end.
GitHub issues are the single source of truth for tracking.

Usage:
    # Check for new messages since last run
    python3 tools/feedback_monitor.py

    # Check messages from the last N hours (override last-check state)
    python3 tools/feedback_monitor.py --since 4h

    # Dry run — classify but don't update state
    python3 tools/feedback_monitor.py --dry-run

    # Show audit log
    python3 tools/feedback_monitor.py --log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_SPACE = "spaces/AAQABmB_3Is"
STATE_FILE = REPO_ROOT / ".feedback_monitor_state.json"
AUDIT_LOG = REPO_ROOT / ".feedback_monitor_audit.jsonl"

# Message IDs to skip (pinned template, bot messages)
SKIP_MESSAGE_IDS = set()

# Prefixes used in agent-generated responses (defense-in-depth).
# Primary defense: agent messages sent with --as-bot are invisible to gchat read.
# This list is a safety net in case a message slips through as --as-user.
AGENT_PREFIXES = (
    "[🦦 Otter]:",           # Otter's standard GChat prefix (with emoji)
    "[Otter]:",              # Otter prefix without emoji (GChat may strip it)
    "*Technical Report:",     # Technical report posts
    "Got it — logged as",
    "Thanks for the report!",
    "Full report:",
    "Full Technical Report",
    "Update: ",
    "Update:",
    "Follow-up",
    "Quick follow-up",
    "On it,",
    "Great catch,",
    "Hi, I'm Otter",
    "Testing thread reply",
    "(Correction to my",      # Correction messages
)

# Regex pattern for agent-style prefixes: [AgentName]: at start of message.
# Catches any agent posting as --as-user regardless of specific prefix list.
import re
AGENT_PREFIX_PATTERN = re.compile(r"^\[[\w🦦🐯🪶 ]+\]:")

def _run(cmd, check=True):
    """Run a shell command, return stdout."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=60
    )
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None
    return result.stdout.strip()


def load_state():
    """Load monitor state (last check timestamp, processed message IDs)."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_check_epoch": 0, "processed_messages": []}


def save_state(state):
    """Save monitor state."""
    # Keep only last 500 processed message IDs to prevent unbounded growth
    state["processed_messages"] = state["processed_messages"][-500:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def audit_log(entry):
    """Append a triage decision to the audit log."""
    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_new_messages(since=None):
    """Read messages from the feedback space since last check.

    Returns list of dicts with keys: id, sender, text, timestamp, thread_id.
    """
    cmd = f"gchat read {FEEDBACK_SPACE} -c 50 --no-resolve --json"
    if since:
        cmd += f" --since {since}"

    output = _run(cmd)
    if not output:
        return []

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        print("WARNING: Failed to parse gchat JSON output", file=sys.stderr)
        return []

    # Navigate the JSON structure: {data: {data: [messages]}}
    raw_messages = data.get("data", {}).get("data", [])
    if not raw_messages:
        return []

    messages = []
    for m in raw_messages:
        msg_id = m.get("google_message_name", "")
        thread_name = m.get("google_thread_name", "")

        # Extract thread ID from thread name (spaces/X/threads/Y -> Y)
        thread_id = None
        if "/threads/" in thread_name:
            thread_id = thread_name.split("/threads/")[-1]

        messages.append({
            "id": msg_id,
            "sender": m.get("sender_name", ""),
            "sender_fbid": m.get("sender_work_user_fbid", ""),
            "timestamp": str(m.get("creation_timestamp", "")),
            "thread_id": thread_id,
            "is_thread_reply": m.get("is_thread_reply", False),
            "text": m.get("message_body", ""),
        })

    return messages


def respond_in_thread(msg_id, response_text):
    """Reply in the same GChat thread as the original message."""
    # Message ID format: spaces/X/messages/Y.Z
    # Thread format: spaces/X/threads/Y (use the part before the dot)
    parts = msg_id.split("/messages/")
    if len(parts) == 2:
        space_name = parts[0]
        thread_key = parts[1].split(".")[0]
        thread_name = f"{space_name}/threads/{thread_key}"
    else:
        thread_name = None

    # Write response to a temp file to avoid shell quoting issues
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(response_text)
        tmp_path = f.name

    try:
        if thread_name:
            cmd = f'gchat send {FEEDBACK_SPACE} --thread "{thread_name}" --text-file {tmp_path} --as-bot'
        else:
            cmd = f'gchat send {FEEDBACK_SPACE} --text-file {tmp_path} --as-bot'

        result = subprocess.run(
            cmd, shell=True,
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    finally:
        os.unlink(tmp_path)


def classify_message(text):
    """Basic keyword-based classification.

    Returns one of: bug, feature, data_correction, question, noise.
    The cron prompt should override this with LLM-based classification.
    """
    text_lower = text.lower()

    # Skip obvious noise
    if len(text_lower.strip()) < 10:
        return "noise"

    # Check for bug indicators
    bug_words = ["error", "crash", "broken", "fails", "bug", "traceback",
                 "exception", "wrong output", "incorrect"]
    if any(w in text_lower for w in bug_words):
        return "bug"

    # Check for data correction
    data_words = ["wrong status", "should be", "incorrect data",
                  "data correction", "model.*wrong", "misclassified"]
    if any(w in text_lower for w in data_words):
        return "data_correction"

    # Check for feature request
    feature_words = ["feature", "would be nice", "can you add",
                     "request", "support for", "new flag"]
    if any(w in text_lower for w in feature_words):
        return "feature"

    # Check for question
    question_words = ["how do", "how to", "what is", "where is",
                      "can i", "is there", "?"]
    if any(w in text_lower for w in question_words):
        return "question"

    return "question"  # Default to question


def show_audit_log(n=20):
    """Display the last N audit log entries."""
    if not AUDIT_LOG.exists():
        print("No audit log found.")
        return

    entries = []
    with open(AUDIT_LOG) as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    for entry in entries[-n:]:
        ts = entry.get("timestamp", "?")[:19]
        cat = entry.get("classification", "?")
        action = entry.get("action", "?")
        sender = entry.get("sender", "?")
        print(f"[{ts}] {cat:16s} | {action:20s} | {sender}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor the Graph Break Corpus Feedback space"
    )
    parser.add_argument("--since", help="Override: check messages since (e.g., 4h, 1d)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify but don't create Tasks or respond")
    parser.add_argument("--log", action="store_true",
                        help="Show recent audit log entries")
    parser.add_argument("--log-count", type=int, default=20,
                        help="Number of audit log entries to show")
    args = parser.parse_args()

    if args.log:
        show_audit_log(args.log_count)
        return

    state = load_state()

    # Determine time window
    if args.since:
        since = args.since
    elif state["last_check_epoch"] > 0:
        minutes_ago = (time.time() - state["last_check_epoch"]) / 60
        # Add 10% buffer to avoid missing messages at boundaries
        minutes_ago = max(minutes_ago * 1.1, 30)
        since = f"{int(minutes_ago)}m"
    else:
        since = "24h"  # First run: look back 24 hours

    print(f"Checking feedback space since {since}...")
    messages = read_new_messages(since)

    # Filter out already-processed messages
    processed = set(state["processed_messages"])
    new_messages = [m for m in messages if m["id"] not in processed]

    # Filter out bot messages and template
    # Primary defense: agent messages sent with --as-bot don't appear in gchat read.
    # Defense-in-depth layers for --as-user messages that slip through:
    #   1. Exact prefix match (AGENT_PREFIXES tuple)
    #   2. Regex pattern match for any [AgentName]: prefix (AGENT_PREFIX_PATTERN)
    def _is_agent_message(msg):
        text = msg.get("text", "")
        if any(text.startswith(p) for p in AGENT_PREFIXES):
            return True
        if AGENT_PREFIX_PATTERN.match(text):
            return True
        return False

    user_messages = [
        m for m in new_messages
        if m["id"] not in SKIP_MESSAGE_IDS
        and m.get("sender", "")  # Skip messages with no sender
        and not _is_agent_message(m)
    ]

    if not user_messages:
        print("No new user messages.")
        if not args.dry_run:
            state["last_check_epoch"] = time.time()
            save_state(state)
        return

    print(f"Found {len(user_messages)} new message(s) to process.")

    for msg in user_messages:
        is_reply = msg.get("is_thread_reply", False)
        classification = classify_message(msg["text"])
        # Thread replies are typically responses/follow-ups, not new bug reports.
        # Classify them as questions (needs_answer) so the cron prompt can decide
        # how to handle them, rather than creating duplicate Tasks.
        if is_reply and classification in ("bug", "feature", "data_correction"):
            classification = "question"
        reply_tag = " (thread reply)" if is_reply else ""
        print(f"\n--- Message from {msg['sender']}{reply_tag} ---")
        print(f"Text: {msg['text'][:200]}...")
        print(f"Classification: {classification}")

        if args.dry_run:
            audit_log({
                "message_id": msg["id"],
                "sender": msg["sender"],
                "classification": classification,
                "action": "dry_run",
                "reasoning": "Dry run mode — no action taken",
            })
            # Don't mark as processed — dry-run should be read-only
            continue

        if classification == "noise":
            audit_log({
                "message_id": msg["id"],
                "sender": msg["sender"],
                "classification": "noise",
                "action": "skipped",
                "reasoning": "Message too short or not actionable",
            })
            state["processed_messages"].append(msg["id"])
            continue

        # All actionable messages (question, bug, feature, data_correction)
        # are logged for the cron prompt to handle directly.
        # No Task creation — GitHub issues are the source of truth.
        action = "needs_answer" if classification == "question" else "needs_action"
        audit_log({
            "message_id": msg["id"],
            "sender": msg["sender"],
            "classification": classification,
            "action": action,
            "text_preview": msg["text"][:200],
        })
        state["processed_messages"].append(msg["id"])

    state["last_check_epoch"] = time.time()
    if not args.dry_run:
        save_state(state)
    print(f"\nProcessed {len(user_messages)} message(s). State {'unchanged (dry-run)' if args.dry_run else 'saved'}.")


if __name__ == "__main__":
    main()
