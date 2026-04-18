"""Shared constants for the OSS Model Graph Break Corpus.

Single source of truth for conventions used across tools/ and sweep/.
When a convention changes, update it here — not in each script.
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_RESULTS_DIR = os.path.join(REPO_ROOT, "sweep_results")
CORPUS_PATH = os.path.join(REPO_ROOT, "corpus", "corpus.json")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

VERSION_DIR_PREFIX = "pt"
VERSION_DIR_PATTERN = re.compile(r"^pt2\.\d+$")

VALID_STATUSES = frozenset({
    "full_graph", "graph_break", "eager_error", "create_error",
    "compile_error", "explain_error", "timeout", "worker_error",
    "zombie", "skipped",
})

FIX_TRANSITION = ("graph_break", "full_graph")
REGRESSION_TRANSITION = ("full_graph", "graph_break")


def find_version_dirs():
    """Discover sweep version directories in canonical order.

    Returns list of (label, path) tuples sorted by version number.
    Only matches exact versioned dirs (pt2.8, pt2.9, etc.), not
    suffixed variants (pt2.10_full, pt2.11-fresh).
    """
    if not os.path.isdir(SWEEP_RESULTS_DIR):
        return []

    def version_key(name):
        parts = re.sub(r"^(pt|v)", "", name).split(".")
        return tuple(int(p) for p in parts if p.isdigit())

    dirs = sorted(
        (d for d in os.listdir(SWEEP_RESULTS_DIR)
         if VERSION_DIR_PATTERN.match(d)
         and os.path.isfile(os.path.join(SWEEP_RESULTS_DIR, d, "identify_results.json"))),
        key=version_key,
    )
    return [(d, os.path.join(SWEEP_RESULTS_DIR, d)) for d in dirs]
