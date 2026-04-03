#!/usr/bin/env python3
"""Feedback Space Monitor — triage new messages from the GChat feedback space.

Reads new messages from the Graph Break Corpus Feedback space, classifies
them, creates Meta Tasks for actionable items, and responds in-thread.

This script is designed to be called by a MyClaw cron job. It handles the
mechanical plumbing (message reading, state tracking, dedup) so the cron
prompt only needs to handle classification and response generation.

Usage:
    # Check for new messages since last run
    python3 tools/feedback_monitor.py

    # Check messages from the last N hours (override last-check state)
    python3 tools/feedback_monitor.py --since 4h

    # Dry run — classify but don't create Tasks or respond
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
ROCKY_SPACE = "spaces/AAQA_65oV7k"  # Rocky's DM space for validation delegation
STATE_FILE = REPO_ROOT / ".feedback_monitor_state.json"
AUDIT_LOG = REPO_ROOT / ".feedback_monitor_audit.jsonl"
TASK_TAG = "graph-break-corpus"

# Message IDs to skip (pinned template, bot messages)
SKIP_MESSAGE_IDS = set()

# Categories that need Rocky's independent validation before corpus changes
NEEDS_VALIDATION = {"bug", "data_correction"}


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
    cmd = f"gchat read {FEEDBACK_SPACE} -c 50 --no-resolve --quiet"
    if since:
        cmd += f" --since {since}"

    output = _run(cmd)
    if not output:
        return []

    messages = []
    current_msg = None

    for line in output.split("\n"):
        # gchat read output format:
        # [timestamp] sender (spaces/X/messages/Y):
        # message text
        if line.startswith("[") and "(" in line and "):" in line:
            if current_msg and current_msg.get("text"):
                messages.append(current_msg)

            # Parse header line
            try:
                ts_end = line.index("]")
                timestamp = line[1:ts_end]
                rest = line[ts_end + 2:]
                sender_end = rest.index(" (")
                sender = rest[:sender_end]
                msg_id_start = rest.index("(") + 1
                msg_id_end = rest.index(")")
                msg_id = rest[msg_id_start:msg_id_end]

                # Extract thread ID if present (spaces/X/messages/Y/threads/Z)
                thread_id = None
                if "/threads/" in msg_id:
                    thread_id = msg_id.split("/threads/")[0].split("/messages/")[1]

                current_msg = {
                    "id": msg_id,
                    "sender": sender,
                    "timestamp": timestamp,
                    "thread_id": thread_id,
                    "text": "",
                }
            except (ValueError, IndexError):
                continue
        elif current_msg is not None:
            if current_msg["text"]:
                current_msg["text"] += "\n" + line
            else:
                current_msg["text"] = line

    if current_msg and current_msg.get("text"):
        messages.append(current_msg)

    return messages


def check_duplicate_tasks(title_keywords):
    """Check if a similar Task already exists with the graph-break-corpus tag.

    Returns the Task number if a duplicate is found, None otherwise.
    """
    cmd = f'meta tasks.task list --tags={TASK_TAG} --is-open --output=json 2>/dev/null'
    output = _run(cmd, check=False)
    if not output:
        return None

    try:
        tasks = json.loads(output)
        for task in tasks:
            task_title = task.get("title", "").lower()
            if any(kw.lower() in task_title for kw in title_keywords):
                return task.get("number")
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def create_task(title, description, category, gchat_msg_id):
    """Create a Meta Task for an actionable feedback item.

    Returns the Task number if successful, None otherwise.
    """
    tag_map = {
        "bug": "bug",
        "feature": "feature",
        "data_correction": "data",
    }
    tag = tag_map.get(category, category)
    full_tag = f"{TASK_TAG},{tag}"

    # Include link to originating GChat message
    desc_with_link = f"{description}\n\nSource: GChat feedback space ({gchat_msg_id})"

    cmd = (
        f'meta tasks.task create '
        f'--title="{title}" '
        f'--description="{desc_with_link}" '
        f'--add-tag={full_tag} '
        f'--output=json 2>/dev/null'
    )
    output = _run(cmd, check=False)
    if not output:
        return None

    try:
        result = json.loads(output)
        return result.get("number") or result.get("id")
    except (json.JSONDecodeError, TypeError):
        # Try to extract task number from non-JSON output
        if output and "T" in output:
            for word in output.split():
                if word.startswith("T") and word[1:].isdigit():
                    return word
    return None


def respond_in_thread(msg_id, response_text):
    """Reply in the same GChat thread as the original message."""
    # Extract thread key from message ID
    # Message ID format: spaces/X/messages/Y
    parts = msg_id.split("/messages/")
    if len(parts) == 2:
        thread_key = parts[1].split(".")[0]  # Thread key is the first part
    else:
        thread_key = None

    if thread_key:
        cmd = f'gchat send {FEEDBACK_SPACE} --thread {thread_key} --quiet'
    else:
        cmd = f'gchat send {FEEDBACK_SPACE} --quiet'

    result = subprocess.run(
        cmd, shell=True, input=response_text,
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0


def delegate_to_rocky(task_num, category, msg_text, msg_id):
    """Send Rocky a validation request via DM for bugs and data corrections.

    Otter is the single face in the feedback space. Rocky validates
    behind the scenes and reports findings back via DM.
    """
    category_label = {
        "bug": "Bug report",
        "data_correction": "Data correction",
    }.get(category, category)

    delegation_msg = (
        f"🔍 *Validation request — {category_label} (T{task_num})*\n\n"
        f"A user reported this in the Graph Break Corpus Feedback space. "
        f"I've created a Task and acknowledged in-thread. "
        f"Can you independently validate?\n\n"
        f"*Report:*\n{msg_text[:500]}\n\n"
        f"*What I need:*\n"
        f"• Reproduce the issue (if applicable)\n"
        f"• Run validate.py to check corpus integrity\n"
        f"• Check if golden set is affected\n"
        f"• Report your findings back here\n\n"
        f"Repo: ~/projects/oss-model-graph-break-corpus/\n"
        f"Source message: {msg_id}"
    )

    cmd = f'gchat send {ROCKY_SPACE} --quiet'
    result = subprocess.run(
        cmd, shell=True, input=delegation_msg,
        capture_output=True, text=True, timeout=30
    )
    return result.returncode == 0


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
    user_messages = [
        m for m in new_messages
        if m["id"] not in SKIP_MESSAGE_IDS
        and "users/" not in m.get("sender", "")  # Skip unresolved bot users
    ]

    if not user_messages:
        print("No new user messages.")
        state["last_check_epoch"] = time.time()
        save_state(state)
        return

    print(f"Found {len(user_messages)} new message(s) to process.")

    for msg in user_messages:
        classification = classify_message(msg["text"])
        print(f"\n--- Message from {msg['sender']} ---")
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
            state["processed_messages"].append(msg["id"])
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

        if classification == "question":
            # Questions are answered directly — no Task created
            # The cron prompt handles generating the answer
            audit_log({
                "message_id": msg["id"],
                "sender": msg["sender"],
                "classification": "question",
                "action": "needs_answer",
                "reasoning": "Question detected — needs LLM-generated answer",
                "text_preview": msg["text"][:200],
            })
            state["processed_messages"].append(msg["id"])
            continue

        # Bug, feature, or data_correction — check for duplicates and create Task
        title_keywords = [w for w in msg["text"].split()[:10] if len(w) > 3]
        dup_task = check_duplicate_tasks(title_keywords)

        if dup_task:
            # Respond with link to existing task
            response = (
                f"Thanks for the report! This looks related to an existing item: "
                f"T{dup_task}. I'll add your context there."
            )
            respond_in_thread(msg["id"], response)
            audit_log({
                "message_id": msg["id"],
                "sender": msg["sender"],
                "classification": classification,
                "action": f"duplicate_of_T{dup_task}",
                "reasoning": "Matched existing open Task by keywords",
            })
        else:
            # Create a new Task
            title_text = msg["text"][:80].replace('"', "'").replace("\n", " ")
            task_title = f"[{classification}] {title_text}"
            task_num = create_task(
                task_title,
                msg["text"][:500],
                classification,
                msg["id"],
            )

            if task_num:
                category_label = {
                    "bug": "Bug report",
                    "feature": "Feature request",
                    "data_correction": "Data correction",
                }.get(classification, classification)

                response = (
                    f"Got it — logged as {category_label} (T{task_num}). "
                    f"I'll investigate and follow up here."
                )
                respond_in_thread(msg["id"], response)

                # Delegate to Rocky for independent validation (bugs & data corrections)
                if classification in NEEDS_VALIDATION:
                    delegated = delegate_to_rocky(
                        task_num, classification, msg["text"], msg["id"]
                    )
                    delegation_status = "delegated_to_rocky" if delegated else "delegation_failed"
                else:
                    delegation_status = "no_delegation_needed"

                audit_log({
                    "message_id": msg["id"],
                    "sender": msg["sender"],
                    "classification": classification,
                    "action": f"created_T{task_num}",
                    "delegation": delegation_status,
                    "reasoning": "New actionable item — Task created",
                })
            else:
                audit_log({
                    "message_id": msg["id"],
                    "sender": msg["sender"],
                    "classification": classification,
                    "action": "task_creation_failed",
                    "reasoning": "meta tasks.task create returned no Task number",
                    "text_preview": msg["text"][:200],
                })

        state["processed_messages"].append(msg["id"])

    state["last_check_epoch"] = time.time()
    save_state(state)
    print(f"\nProcessed {len(user_messages)} message(s). State saved.")


if __name__ == "__main__":
    main()
