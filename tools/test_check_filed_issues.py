#!/usr/bin/env python3
"""Tests for check_filed_issues.py self-noise filter (Peng directive 2026-05-11).

The github-issue-monitor + brief paths must NOT surface activity caused by
Otter (a) self-comments, (b) self body edits, (c) self issue creation,
(d) self labels/state changes. Stage 2 (in _classify_comments) handles (a);
Stage 3 (added 2026-05-11) handles (b)+(c)+(d).

Run: PYTHONPATH=$(pwd) python3 tools/test_check_filed_issues.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_filed_issues as cfi


def _iss(repo: str, number: int, *, author: str = "external_user",
         state: str = "open", updated_at: str = "2026-05-11T00:00:00Z",
         comments_count: int = 0, scope: str = "primary") -> dict:
    return {
        "repo": repo, "scope": scope, "number": number,
        "title": f"test issue {number}", "state": state,
        "html_url": f"https://github.com/{repo}/issues/{number}",
        "updated_at": updated_at, "created_at": updated_at,
        "comments_count": comments_count, "labels": [],
        "author": author,
    }


class SelfNoiseFilterTests(unittest.TestCase):
    """Pin the 4 scopes Peng required: (a) self-comments, (b) self body edits,
    (c) self issue creation, (d) self labels/state changes."""

    def setUp(self):
        self._orig_events = cfi._fetch_events_since
        self._orig_comments = cfi._fetch_new_comments
        self._events: list[dict] = []
        self._comments: list[dict] = []
        cfi._fetch_events_since = lambda repo, number, prev, token: [
            e for e in self._events if e.get("created_at", "") > (prev or "")
        ]
        cfi._fetch_new_comments = lambda repo, number, n_new, prev, token: [
            c for c in self._comments if c.get("created_at", "") > (prev or "")
        ]

    def tearDown(self):
        cfi._fetch_events_since = self._orig_events
        cfi._fetch_new_comments = self._orig_comments

    # --- (c) self issue creation ---

    def test_brand_new_self_authored_no_comments_is_suppressed(self):
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T01:30:00Z")
        flagged = cfi.diff_against_state([iss], state={}, token="fake")
        self.assertFalse(flagged[0]["new_activity"])

    def test_brand_new_external_authored_surfaces(self):
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 200,
                   author="external_user", comments_count=0,
                   updated_at="2026-05-11T01:30:00Z")
        flagged = cfi.diff_against_state([iss], state={}, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    # --- (b) self body edits ---

    def test_self_body_edit_only_is_suppressed(self):
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "renamed"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])
        self.assertTrue(flagged[0]["self_only_issue_update"])

    def test_external_body_edit_surfaces(self):
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": "external_user"}, "event": "renamed"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    # --- (d) self labels / state changes ---

    def test_self_label_only_is_suppressed(self):
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "labeled"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])

    def test_self_close_then_external_reopen_surfaces(self):
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "closed"},
            {"created_at": "2026-05-11T13:30:00Z",
             "actor": {"login": "external_user"}, "event": "reopened"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:30:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    # --- mixed: external comment is the signal we want ---

    def test_external_comment_on_self_authored_issue_surfaces(self):
        self._comments = [
            {"created_at": "2026-05-11T14:00:00Z",
             "user": {"login": "external_user"},
             "body": "Reproduced; here are details."},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=1,
                   updated_at="2026-05-11T14:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])
        self.assertEqual(flagged[0]["non_self_comment_count"], 1)

    # --- regression: combined self-comment + self-body-edit ---

    def test_self_comment_plus_self_body_edit_is_suppressed(self):
        self._comments = [
            {"created_at": "2026-05-11T13:00:00Z",
             "user": {"login": cfi.AUTHOR}, "body": "self closure note"},
        ]
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "renamed"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=1,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])

    # --- defensive: no change is no change ---

    def test_no_change_is_no_change(self):
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T01:30:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#124": {
            "last_updated_at": "2026-05-11T01:30:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])
        self.assertEqual(flagged[0]["signal"], "no change")


class AdversaryGapTests(unittest.TestCase):
    """Pins for gaps surfaced by adversary-review case adv-2026-05-11-094000-self-noise-filter.

    Three high/medium-severity gaps were addressed in the same commit:
      Gap 1 — Stage 3 must fire even when comments_increased (previous gating
              shadowed external issue-level events when self-comment landed).
      Gap 2 — empty /events list ≠ no activity: GitHub does not log pure
              description edits, so the author-heuristic distinguishes self
              body edits (suppress) from external body edits (surface).
      Gap 3 — brand-new self-authored issue should still surface external
              label/state events (no fast-path bypass).
    """

    def setUp(self):
        self._orig_events = cfi._fetch_events_since
        self._orig_comments = cfi._fetch_new_comments
        self._events: list[dict] = []
        self._comments: list[dict] = []
        cfi._fetch_events_since = lambda repo, number, prev, token: [
            e for e in self._events if e.get("created_at", "") > (prev or "")
        ]
        cfi._fetch_new_comments = lambda repo, number, n_new, prev, token: [
            c for c in self._comments if c.get("created_at", "") > (prev or "")
        ]

    def tearDown(self):
        cfi._fetch_events_since = self._orig_events
        cfi._fetch_new_comments = self._orig_comments

    def test_external_body_edit_with_self_comment_surfaces_gap1(self):
        """Self-comment + external rename in same window → must surface (Gap 1)."""
        self._comments = [
            {"created_at": "2026-05-11T12:30:00Z",
             "user": {"login": cfi.AUTHOR}, "body": "self note"},
        ]
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": "external_user"}, "event": "renamed"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author=cfi.AUTHOR, comments_count=1,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    def test_empty_events_with_self_authored_is_suppressed_gap2(self):
        """Pure body edit by self (no /events log) → must suppress (Gap 2)."""
        # _events deliberately empty — body edits don't appear in /events
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])

    def test_empty_events_with_external_authored_surfaces_gap2(self):
        """Pure body edit on external-authored issue → fail-open and surface (Gap 2)."""
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author="external_user", comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    def test_brand_new_self_authored_with_external_label_surfaces_gap3(self):
        """Otter files + external user labels before our first poll → must surface (Gap 3)."""
        self._events = [
            {"created_at": "2026-05-11T12:50:00Z",
             "actor": {"login": "external_user"}, "event": "labeled"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        flagged = cfi.diff_against_state([iss], state={}, token="fake")
        self.assertTrue(flagged[0]["new_activity"])

    def test_self_close_immediately_after_creation_is_suppressed(self):
        """Otter files + self-closes (no external action) → suppress."""
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "closed"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 124,
                   author=cfi.AUTHOR, state="closed", comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        flagged = cfi.diff_against_state([iss], state={}, token="fake")
        self.assertFalse(flagged[0]["new_activity"])

    def test_priority_author_comment_overrides_self_only_update(self):
        """Priority author commenting must surface as priority signal regardless of Stage 3."""
        self._comments = [
            {"created_at": "2026-05-11T13:00:00Z",
             "user": {"login": "anijain2305"}, "body": "looking at this"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author=cfi.AUTHOR, comments_count=1,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertTrue(flagged[0]["new_activity"])
        self.assertTrue(flagged[0]["priority"])
        self.assertTrue(flagged[0]["priority_signals"][0].startswith("anijain2305"))

    def test_self_label_then_self_unlabel_is_suppressed(self):
        """Multi-event self-only sequence → all() check pins suppression."""
        self._events = [
            {"created_at": "2026-05-11T12:30:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "labeled"},
            {"created_at": "2026-05-11T13:00:00Z",
             "actor": {"login": cfi.AUTHOR}, "event": "unlabeled"},
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])
        self.assertTrue(flagged[0]["self_only_issue_update"])

    def test_event_with_missing_actor_key_does_not_crash(self):
        """Defensive: events with no actor field (transfers, system actions)."""
        self._events = [
            {"created_at": "2026-05-11T13:00:00Z",
             "event": "transferred"},  # no actor field
        ]
        iss = _iss("penguinwu/oss-model-graph-break-corpus", 5,
                   author=cfi.AUTHOR, comments_count=0,
                   updated_at="2026-05-11T13:00:00Z")
        state = {"penguinwu/oss-model-graph-break-corpus#5": {
            "last_updated_at": "2026-05-11T01:00:00Z",
            "last_comments_count": 0,
        }}
        # Should NOT raise; current behavior with empty-actor: a == "" not == AUTHOR,
        # but `a and a != AUTHOR` filters out empties → non_self_issue_event=False
        # → suppress as if all-self. Pin this so refactors don't crash on missing actor.
        flagged = cfi.diff_against_state([iss], state=state, token="fake")
        self.assertFalse(flagged[0]["new_activity"])


if __name__ == "__main__":
    unittest.main()
