"""Cohort file validator — fail-loud on the failure modes that produced 2026-05-06.

Surfaced by adversary-review case_id adv-2026-05-07-124100-cohort-regen-fix
(subagents/adversary-review/invocations/). The original cohort-regen mitigation
left several asymmetric paths where the loader only WARN'd on conditions the
sanity-check skill claimed were STRICT. This module collapses validation into
one place with explicit error codes that callers (run_sweep.py CLI, tests,
the cohort-invariant executor) can consume uniformly.

Required cohort shape (dict form):
    {
      "_metadata": {
        "derived_from": "<path to source>",   # required
        "filter": "<filter expr>",             # required
        "model_count": <int>,                  # required
        "source_versions": {                   # required, must be non-empty
          "torch": "<ver>",
          "transformers": "<ver>",
          "diffusers": "<ver>",
        },
        # optional:
        "derived_at": "<ISO8601>",
        "generated_by": "<tool name>",
      },
      "models": [ {name, source, ...}, ... ],
    }

Bare-list cohorts (a JSON list with no _metadata) are REJECTED unless the
caller explicitly opts in with allow_bare=True.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REQUIRED_VERSION_KEYS = ("torch", "transformers", "diffusers")
REQUIRED_METADATA_KEYS = ("derived_from", "filter", "model_count", "source_versions")


class CohortValidationError(Exception):
    """Raised when a cohort fails validation. .code is a stable identifier."""

    def __init__(self, code: str, message: str, *, details: Optional[dict] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class ValidatedCohort:
    specs: list
    source_versions: dict
    metadata: dict  # full _metadata block (or {} for bare-list with allow_bare=True)
    is_bare: bool


def validate_cohort(
    cohort_path: Path,
    *,
    allow_bare: bool = False,
    allow_empty_versions: bool = False,
    allow_partial_versions: bool = False,
    allow_version_mismatch: bool = False,
    allow_stale: bool = False,
    version_info: Optional[dict] = None,
) -> ValidatedCohort:
    """Validate a cohort file. Returns ValidatedCohort or raises CohortValidationError.

    Failure codes (stable for tooling):
        FILE_NOT_FOUND       — cohort path does not exist
        INVALID_JSON         — cohort file is not parseable JSON
        BARE_LIST_REJECTED   — flat list without _metadata (allow_bare=False)
        MISSING_METADATA_KEY — _metadata present but missing one of the required keys
        EMPTY_SOURCE_VERSIONS — source_versions is {} (allow_empty_versions=False)
        PARTIAL_SOURCE_VERSIONS — source_versions missing torch/transformers/diffusers
        VERSION_MISMATCH     — source_versions differs from launch version_info
        STALE_COHORT         — cohort file mtime is older than the source it derived from
        INVALID_MODELS_LIST  — "models" key is not a list, or absent in dict form
    """
    if not cohort_path.is_file():
        raise CohortValidationError("FILE_NOT_FOUND", f"cohort path not found: {cohort_path}")

    try:
        loaded = json.loads(cohort_path.read_text())
    except json.JSONDecodeError as e:
        raise CohortValidationError("INVALID_JSON", f"cohort not parseable: {e}")

    # Shape: bare list vs dict-with-_metadata
    if isinstance(loaded, list):
        if not allow_bare:
            raise CohortValidationError(
                "BARE_LIST_REJECTED",
                f"cohort is a flat list with no _metadata block; this is the exact 2026-05-06 "
                f"failure shape. Either regenerate via tools/generate_cohort.py, or pass "
                f"--allow-bare-cohort to override (recorded in sweep_state.json).",
            )
        return ValidatedCohort(specs=loaded, source_versions={}, metadata={}, is_bare=True)

    if not isinstance(loaded, dict):
        raise CohortValidationError(
            "INVALID_JSON",
            f"cohort top-level must be a list (legacy) or dict with 'models' (canonical); "
            f"got {type(loaded).__name__}",
        )

    if "models" not in loaded or not isinstance(loaded["models"], list):
        raise CohortValidationError(
            "INVALID_MODELS_LIST",
            f"cohort dict missing required 'models' list",
        )

    metadata = loaded.get("_metadata", {}) or {}
    specs = loaded["models"]

    # Required _metadata keys
    missing_keys = [k for k in REQUIRED_METADATA_KEYS if k not in metadata]
    if missing_keys:
        raise CohortValidationError(
            "MISSING_METADATA_KEY",
            f"_metadata missing required key(s): {', '.join(missing_keys)}",
            details={"missing": missing_keys},
        )

    source_versions = metadata.get("source_versions", {}) or {}

    # Empty source_versions
    if not source_versions:
        if not allow_empty_versions:
            raise CohortValidationError(
                "EMPTY_SOURCE_VERSIONS",
                f"_metadata.source_versions is empty; version-compat cannot be verified. "
                f"Pass --allow-empty-versions to override (recorded in sweep_state.json).",
            )
    else:
        # Partial source_versions (some required keys missing)
        partial_missing = [k for k in REQUIRED_VERSION_KEYS if k not in source_versions]
        if partial_missing and not allow_partial_versions:
            raise CohortValidationError(
                "PARTIAL_SOURCE_VERSIONS",
                f"_metadata.source_versions missing required key(s): {', '.join(partial_missing)}. "
                f"Cross-version mismatches in unchecked libraries (e.g. transformers) silently "
                f"corrupt sweep results. Pass --allow-partial-versions to override.",
                details={"missing": partial_missing, "present": list(source_versions.keys())},
            )

    # Version-compat (only checks libraries present in BOTH source_versions and version_info)
    if version_info and source_versions:
        mismatches = []
        for pkg in REQUIRED_VERSION_KEYS:
            src = source_versions.get(pkg)
            cur = version_info.get(pkg)
            if src and cur and src != cur:
                mismatches.append({"package": pkg, "source": src, "launching": cur})
        if mismatches and not allow_version_mismatch:
            raise CohortValidationError(
                "VERSION_MISMATCH",
                f"version mismatch between cohort and launch env: "
                + "; ".join(f"{m['package']}: source={m['source']!r} launching={m['launching']!r}"
                            for m in mismatches)
                + ". Pass --allow-version-mismatch to override.",
                details={"mismatches": mismatches},
            )

    # Stale cohort: source.mtime > cohort.mtime
    if not allow_stale:
        derived_from = metadata.get("derived_from")
        if derived_from:
            source_path = _resolve_derived_from(derived_from)
            if source_path and source_path.is_file():
                source_mtime = source_path.stat().st_mtime
                cohort_mtime = cohort_path.stat().st_mtime
                if source_mtime > cohort_mtime:
                    raise CohortValidationError(
                        "STALE_COHORT",
                        f"cohort {cohort_path.name} was derived from {derived_from} "
                        f"but the source is now newer (cohort mtime {cohort_mtime:.0f} < source mtime {source_mtime:.0f}). "
                        f"Regenerate via tools/generate_cohort.py, or pass --allow-stale-cohort to override.",
                        details={
                            "cohort_mtime": cohort_mtime,
                            "source_mtime": source_mtime,
                            "derived_from": str(derived_from),
                            "source_path": str(source_path),
                        },
                    )

    return ValidatedCohort(
        specs=specs,
        source_versions=source_versions,
        metadata=metadata,
        is_bare=False,
    )


def _resolve_derived_from(derived_from: str) -> Optional[Path]:
    """Resolve the derived_from path. May be relative to repo root or absolute."""
    p = Path(derived_from)
    if p.is_absolute():
        return p
    # Try relative to repo root (one level above sweep/)
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / p
    if candidate.is_file():
        return candidate
    # Fall back to literal interpretation (cwd-relative)
    return p
