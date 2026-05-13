#!/usr/bin/env python3
"""Tests for tools/count_breaks_per_pattern.py.

Per methodology R12: distinguish (model_classes, pair_rows, distinct_breaks);
filter duplicate-suppressed entries by default.
"""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from count_breaks_per_pattern import count_pattern, DUPLICATE_SUPPRESSED_MARKER


def _row(name: str, mode: str, reasons: list[str]) -> dict:
    """Build a fixture row with explicit reasons."""
    return {
        "name": name,
        "mode": mode,
        "break_reasons": [{"reason": r} for r in reasons],
    }


class TestUnitHierarchy(unittest.TestCase):
    """Pin the three-unit hierarchy from R12."""

    def test_one_model_two_modes_one_break_each(self):
        """Same model in eval+train hits a pattern once each → 1 class / 2 pair-rows / 2 breaks."""
        sweep = {"results": [
            _row("FooModel", "eval", ["Hit pattern X here"]),
            _row("FooModel", "train", ["Hit pattern X here"]),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"))
        self.assertEqual(r["model_classes"], 1)
        self.assertEqual(r["pair_rows"], 2)
        self.assertEqual(r["distinct_breaks"], 2)

    def test_two_models_one_mode_each(self):
        """Two distinct models → 2 classes / 2 pair-rows / 2 breaks."""
        sweep = {"results": [
            _row("FooModel", "eval", ["Hit pattern X here"]),
            _row("BarModel", "eval", ["Hit pattern X here"]),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"))
        self.assertEqual(r["model_classes"], 2)
        self.assertEqual(r["pair_rows"], 2)
        self.assertEqual(r["distinct_breaks"], 2)


class TestDuplicateSuppressedFilter(unittest.TestCase):
    """Pin the duplicate-suppressed filter (default ON)."""

    def test_suppressed_entries_are_skipped_by_default(self):
        """Original break + 4 duplicate-suppressed copies → counted as 1 break."""
        sweep = {"results": [
            _row("FooModel", "eval", [
                "Hit pattern X here (original)",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
            ]),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"))
        self.assertEqual(r["distinct_breaks"], 1, "suppressed entries should not count as distinct breaks")
        self.assertEqual(r["suppressed_skipped"], 4)

    def test_filter_can_be_disabled(self):
        """With --no-filter-duplicate-suppressed, all entries count (over-counts)."""
        sweep = {"results": [
            _row("FooModel", "eval", [
                "Hit pattern X here (original)",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
            ]),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"), filter_duplicate_suppressed=False)
        self.assertEqual(r["distinct_breaks"], 3)
        self.assertEqual(r["suppressed_skipped"], 0)


class TestSpecificFilter(unittest.TestCase):
    """Pin the loose+specific double-filter behavior."""

    def test_specific_narrows_loose_pattern(self):
        """Loose 'Reconstruction' matches 2 entries; specific 'DictItemsIterator' narrows to 1."""
        sweep = {"results": [
            _row("FooModel", "eval", ["Reconstruction failure: SomeOtherThing"]),
            _row("BarModel", "eval", ["Reconstruction failure: DictItemsIterator()"]),
        ]}
        loose = count_pattern(sweep, re.compile("Reconstruction"))
        self.assertEqual(loose["distinct_breaks"], 2)
        narrowed = count_pattern(sweep, re.compile("Reconstruction"), specific_re=re.compile("DictItemsIterator"))
        self.assertEqual(narrowed["distinct_breaks"], 1)
        self.assertEqual(narrowed["model_classes"], 1)
        self.assertEqual(narrowed["sample_models"], ["BarModel"])

    def test_no_match_returns_zero(self):
        sweep = {"results": [_row("FooModel", "eval", ["totally unrelated reason"])]}
        r = count_pattern(sweep, re.compile("nonexistent pattern"))
        self.assertEqual(r["model_classes"], 0)
        self.assertEqual(r["pair_rows"], 0)
        self.assertEqual(r["distinct_breaks"], 0)


class TestRegressionPinUnitConflation(unittest.TestCase):
    """Regression test for the 2026-05-13 unit-conflation bug.

    A model with 2 modes hitting a pattern N times each must report
    1 class / 2 pair-rows / 2N breaks. The historical bug reported
    "2N models" by conflating pair-rows with model-class count.
    """

    def test_unit_conflation_regression(self):
        sweep = {"results": [
            _row("FooModel", "eval", ["pattern X"] * 3),
            _row("FooModel", "train", ["pattern X"] * 3),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"))
        self.assertEqual(r["model_classes"], 1, "1 distinct model class, NOT pair_rows count")
        self.assertEqual(r["pair_rows"], 2)
        self.assertEqual(r["distinct_breaks"], 6)


class TestRegressionPinDuplicateSuppressedCounting(unittest.TestCase):
    """Regression test for the 2026-05-13 duplicate-suppressed counting bug.

    Mode A on the #27 EDIT surfaced a 16-vs-8 number discrepancy: validation
    script counted 16 nn.Parameter ctor breaks when body's graph_break_count
    semantics (deduped) said 8. Root cause: counting "(suppressed due to
    duplicate graph break)" entries as separate breaks. Filter is on by default.
    """

    def test_duplicate_suppressed_counting_regression(self):
        # 4 pair-rows, each with 1 original + 1 suppressed = 8 entries but 4 distinct breaks
        sweep = {"results": [
            _row("BBP_ForCG", "eval", ["nn.Parameter break", f"nn.Parameter break ({DUPLICATE_SUPPRESSED_MARKER})"]),
            _row("BBP_ForCG", "train", ["nn.Parameter break", f"nn.Parameter break ({DUPLICATE_SUPPRESSED_MARKER})"]),
            _row("BBP_Model", "eval", ["nn.Parameter break", f"nn.Parameter break ({DUPLICATE_SUPPRESSED_MARKER})"]),
            _row("BBP_Model", "train", ["nn.Parameter break", f"nn.Parameter break ({DUPLICATE_SUPPRESSED_MARKER})"]),
        ]}
        r = count_pattern(sweep, re.compile("nn.Parameter"))
        self.assertEqual(r["distinct_breaks"], 4, "with default dedup filter, 8 entries collapse to 4 distinct breaks")
        self.assertEqual(r["suppressed_skipped"], 4)
        self.assertEqual(r["model_classes"], 2)
        self.assertEqual(r["pair_rows"], 4)


class TestShapeValidation(unittest.TestCase):
    """Pin the dict-without-results / non-list shape error path (adversary GAP 1)."""

    def test_dict_without_results_key_raises(self):
        """Silent zero on shape-mismatched input is exactly the false-confidence class R12 prevents."""
        sweep = {"foo": "bar", "explain_results": [_row("FooModel", "eval", ["pattern X"])]}
        with self.assertRaises(ValueError) as cm:
            count_pattern(sweep, re.compile("pattern X"))
        self.assertIn("'results' key", str(cm.exception))

    def test_non_list_results_raises(self):
        sweep = {"results": "this is a string not a list"}
        with self.assertRaises(ValueError) as cm:
            count_pattern(sweep, re.compile("X"))
        self.assertIn("expected list", str(cm.exception))

    def test_bare_list_input_works(self):
        """Bare list input (the else branch of the dict check) must work end-to-end."""
        rows = [_row("Foo", "eval", ["X"]), _row("Bar", "eval", ["X"])]
        r = count_pattern(rows, re.compile("X"))
        self.assertEqual(r["model_classes"], 2)
        self.assertEqual(r["pair_rows"], 2)
        self.assertEqual(r["distinct_breaks"], 2)


class TestSuppressedAfterPattern(unittest.TestCase):
    """Pin order-of-operations: pattern check FIRST, then suppressed filter (adversary GAP 2).

    Lets users investigate the marker itself via --pattern "suppressed" +
    --no-filter-duplicate-suppressed. Also makes suppressed_skipped report
    "of pattern matches, N were dynamo-dedupe markers" — actionable signal.
    """

    def test_suppressed_skipped_counts_only_pattern_matching_entries(self):
        sweep = {"results": [
            _row("FooModel", "eval", [
                "Hit pattern X here",
                f"Hit pattern X here ({DUPLICATE_SUPPRESSED_MARKER})",
                # An unrelated suppressed entry that does NOT match the pattern:
                f"Different reason ({DUPLICATE_SUPPRESSED_MARKER})",
            ]),
        ]}
        r = count_pattern(sweep, re.compile("pattern X"))
        # Pattern matches 2 entries; 1 is suppressed → distinct_breaks=1, suppressed_skipped=1
        self.assertEqual(r["distinct_breaks"], 1)
        self.assertEqual(r["suppressed_skipped"], 1, "suppressed_skipped should reflect pattern-matching skips, not all skips")

    def test_user_can_investigate_marker_with_filter_off(self):
        sweep = {"results": [
            _row("FooModel", "eval", [
                "X (original)",
                f"X ({DUPLICATE_SUPPRESSED_MARKER})",
            ]),
        ]}
        r = count_pattern(sweep, re.compile("suppressed"), filter_duplicate_suppressed=False)
        self.assertEqual(r["distinct_breaks"], 1)


if __name__ == "__main__":
    unittest.main()
