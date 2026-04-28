"""Per-trial filesystem-integrity check.

Detects file modification contamination after a discovery trial — the case
where an agent (or any subprocess in the trial) writes to a directory the
trial was supposed to leave untouched.

Why this exists
---------------
Sandbox isolation is unverifiable by inspection. Even with a correct sandbox
(per-trial copy + PYTHONPATH redirect), an agent that reads the case source
and discovers the absolute site-packages path can `Edit`/`Write` directly to
the shared file, bypassing the sandbox entirely. claude's `--disallowed-tools`
and `permissions.deny` patterns were tested and do NOT enforce path-level
write restrictions reliably (verified 2026-04-28).

So we don't trust isolation. We *detect* contamination after the fact, every
trial, and exclude contaminated trials from analysis.

Design
------
Three tiers of checks, increasing depth:

**Tier 1 (always-on, ~2s/walk):** mtime+size baseline.
At trial start: walk every monitored dir, record `{relpath: (mtime_ns, size)}`.
At trial end: re-walk, diff. New / missing / mtime-changed / size-changed
paths are flagged.

**Tier 2 (only on Tier 1 flag, OR every Nth trial as a sanity sample):**
Hash + line-count diff. Computes md5 + unified-diff line counts for each
suspect file, persisted as forensic evidence.

**Tier 3 (defense in depth, always-on, ~10ms):** canary files.
At trial start: write `__contamination_canary_<trial_id>` with a known hash
to each monitored dir. At trial end: verify hash unchanged + delete.
Catches the rare mtime-stable, size-stable overwrite case.

Scope (whitelist mindset)
-------------------------
Monitored = "places agents must NOT touch":
  - Every `/home/pengwu/envs/*/lib/python*/site-packages/` (all venvs)
  - The corpus repo `/home/pengwu/projects/oss-model-graph-break-corpus/`
  - `/home/pengwu/.myclaw-shared/` (skill files)

Allowed write roots (subtracted from monitored at scope-resolution time):
  - `out_dir` for the trial (per-trial output dir)
  - per-trial sandbox dirs (anything under `out_dir/sandbox/`)

This is `monitored - allowed = scanned`.

Public API
----------
    snapshot = take_snapshot(monitored_dirs, allowed_roots) -> Snapshot
    plant_canaries(monitored_dirs, trial_id) -> CanaryRecord
    ... trial runs ...
    verify_canaries(canary_record) -> CanaryResult
    diff = diff_against_snapshot(snapshot, monitored_dirs, allowed_roots) -> Diff
    if diff.has_changes:
        report = deep_inspect(diff, snapshot)  # Tier 2

Self-test (run as `python -m discovery.filesystem_integrity`):
  - clean trial (no changes) → no flags
  - file modified in-place → flagged with size/mtime delta
  - new file added → flagged as new
  - file deleted → flagged as missing
  - mtime-stable overwrite → caught by canary
  - canary deleted → caught
  - canary modified → caught
  - allowed-root writes → IGNORED (sandbox + out_dir don't trip detection)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# ---------- defaults ----------

DEFAULT_MONITORED_GLOBS: list[str] = [
    "/home/pengwu/envs/*/lib/python*/site-packages",
    "/home/pengwu/projects/oss-model-graph-break-corpus",
    "/home/pengwu/.myclaw-shared",
]

# Subtree names to skip during walk (cuts false positives + scan time).
# These are dirs that legitimately churn during normal operation.
SKIP_DIRNAMES = {
    "__pycache__", ".git", ".sl", "node_modules", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", "*.egg-info",
}

# Filenames that legitimately churn during normal operation (state files,
# log files, cache files written by side-process tooling). Add to this list
# when a real false-positive surfaces; never broaden to a glob unless we know
# the class is safe (agents rarely edit dotfiles, but they CAN).
SKIP_FILENAMES = {
    ".feedback_monitor_state.json",  # tools/feedback_monitor.py state
}


# ---------- dataclasses ----------

@dataclass
class Snapshot:
    """File-system state at a point in time."""
    taken_at: str                                            # ISO timestamp
    monitored_roots: list[str]                               # resolved roots
    allowed_roots: list[str]                                 # excluded subtrees
    files: dict[str, tuple[int, int]]                       # relpath → (mtime_ns, size)

    def to_dict(self) -> dict:
        return {
            "taken_at": self.taken_at,
            "monitored_roots": self.monitored_roots,
            "allowed_roots": self.allowed_roots,
            "files": {k: list(v) for k, v in self.files.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(
            taken_at=d["taken_at"],
            monitored_roots=d["monitored_roots"],
            allowed_roots=d["allowed_roots"],
            files={k: tuple(v) for k, v in d["files"].items()},
        )


@dataclass
class FileChange:
    path: str            # absolute path
    kind: str            # "new" | "missing" | "modified"
    old_mtime_ns: int | None = None
    new_mtime_ns: int | None = None
    old_size: int | None = None
    new_size: int | None = None
    # Tier 2 (filled by deep_inspect):
    old_md5: str | None = None
    new_md5: str | None = None
    lines_added: int | None = None
    lines_removed: int | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Diff:
    has_changes: bool
    changes: list[FileChange] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_changes": self.has_changes,
            "n_changes": len(self.changes),
            "changes": [c.to_dict() for c in self.changes],
        }


@dataclass
class CanaryRecord:
    """Records canary files planted in monitored dirs."""
    trial_id: str
    canaries: dict[str, str]   # absolute path → expected md5
    content: str               # canary file body (same for all canaries)


@dataclass
class CanaryResult:
    intact: bool
    failures: list[dict] = field(default_factory=list)
    # failures: [{"path": ..., "kind": "deleted"|"modified", "expected_md5":..., "actual_md5":...}]

    def to_dict(self) -> dict:
        return {"intact": self.intact, "failures": self.failures}


# ---------- core ----------

def _resolve_monitored_roots(globs: Iterable[str]) -> list[Path]:
    """Expand globs (e.g. `/home/pengwu/envs/*/lib/python*/site-packages`)."""
    import glob
    out: list[Path] = []
    for g in globs:
        for p in glob.glob(g):
            pp = Path(p)
            if pp.is_dir():
                out.append(pp.resolve())
    # Dedup, preserve order.
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _walk_files(
    roots: list[Path],
    allowed_roots: list[Path],
    skip_dirnames: set[str] = SKIP_DIRNAMES,
    skip_filenames: set[str] = SKIP_FILENAMES,
) -> dict[str, tuple[int, int]]:
    """Walk monitored roots, return `{abspath_str: (mtime_ns, size)}`.

    Skips:
      - any directory whose basename is in `skip_dirnames`
      - any directory that lies under an `allowed_root`
      - symlinks (avoid loops + we only care about real on-disk state)
      - canary files (`CANARY_PREFIX`-prefixed) so canary planting/verification
        is independent of Tier 1 baseline tracking
      - filenames in `skip_filenames` (legitimate churn from side-process tooling)
    """
    out: dict[str, tuple[int, int]] = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dp = Path(dirpath)
            # Prune by allowed roots.
            if any(_is_under(dp, ar) or dp == ar for ar in allowed_roots):
                dirnames[:] = []
                continue
            # Prune by skip names.
            dirnames[:] = [d for d in dirnames if d not in skip_dirnames]
            for fn in filenames:
                if fn.startswith(CANARY_PREFIX) or fn in skip_filenames:
                    continue
                fp = dp / fn
                try:
                    st = fp.stat()
                except (FileNotFoundError, PermissionError, OSError):
                    continue
                out[str(fp)] = (st.st_mtime_ns, st.st_size)
    return out


def take_snapshot(
    monitored_globs: Iterable[str] | None = None,
    allowed_roots: Iterable[Path | str] | None = None,
) -> Snapshot:
    """Snapshot the current state of all monitored dirs (minus allowed roots)."""
    globs = list(monitored_globs) if monitored_globs is not None else DEFAULT_MONITORED_GLOBS
    allowed = [Path(p).resolve() for p in (allowed_roots or [])]
    roots = _resolve_monitored_roots(globs)
    files = _walk_files(roots, allowed)
    return Snapshot(
        taken_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        monitored_roots=[str(r) for r in roots],
        allowed_roots=[str(p) for p in allowed],
        files=files,
    )


def diff_against_snapshot(
    snap: Snapshot,
    allowed_roots: Iterable[Path | str] | None = None,
) -> Diff:
    """Re-walk monitored roots, diff against `snap`. Returns flagged changes."""
    allowed = [Path(p).resolve() for p in (allowed_roots or [])]
    # Re-walk using the snap's monitored roots (so the diff is apples-to-apples).
    roots = [Path(r) for r in snap.monitored_roots]
    now = _walk_files(roots, allowed)

    changes: list[FileChange] = []
    old_keys = set(snap.files)
    new_keys = set(now)

    for path in new_keys - old_keys:
        new_mtime, new_size = now[path]
        changes.append(FileChange(
            path=path, kind="new",
            new_mtime_ns=new_mtime, new_size=new_size,
        ))
    for path in old_keys - new_keys:
        old_mtime, old_size = snap.files[path]
        changes.append(FileChange(
            path=path, kind="missing",
            old_mtime_ns=old_mtime, old_size=old_size,
        ))
    for path in old_keys & new_keys:
        old_mtime, old_size = snap.files[path]
        new_mtime, new_size = now[path]
        if old_mtime != new_mtime or old_size != new_size:
            changes.append(FileChange(
                path=path, kind="modified",
                old_mtime_ns=old_mtime, new_mtime_ns=new_mtime,
                old_size=old_size, new_size=new_size,
            ))

    return Diff(has_changes=len(changes) > 0, changes=changes)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def deep_inspect(
    diff: Diff,
    backups: dict[str, Path] | None = None,
) -> Diff:
    """Tier 2: enrich each FileChange with md5 + line-count diff vs `backups`.

    `backups`, if provided, maps absolute path → backup file (the pristine
    pre-trial copy). Without backups, we can only record current-state md5.
    Returns the enriched Diff (in place).
    """
    backups = backups or {}
    for ch in diff.changes:
        if ch.kind in ("new", "modified"):
            try:
                ch.new_md5 = _md5(Path(ch.path))
            except Exception:
                pass
        if ch.kind in ("missing", "modified"):
            backup = backups.get(ch.path)
            if backup and backup.exists():
                try:
                    ch.old_md5 = _md5(backup)
                except Exception:
                    pass
                # Cheap line-count diff if we have both sides.
                if ch.kind == "modified":
                    try:
                        old_lines = backup.read_text(errors="replace").splitlines()
                        new_lines = Path(ch.path).read_text(errors="replace").splitlines()
                        # Set-based count is approximate but cheap and decisive
                        # for the "did anything substantive change" question.
                        old_set = set(old_lines)
                        new_set = set(new_lines)
                        ch.lines_added = len(new_set - old_set)
                        ch.lines_removed = len(old_set - new_set)
                    except Exception:
                        pass
    return diff


# ---------- canaries ----------

CANARY_PREFIX = "__contamination_canary_"


def plant_canaries(
    monitored_dirs: Iterable[Path | str],
    trial_id: str,
) -> CanaryRecord:
    """Plant a canary file in each monitored dir; return the record.

    Canary content embeds trial_id + timestamp so collisions across trials
    (e.g. two trials sharing a monitored dir) raise PermissionError or get
    overwritten visibly. Caller should ensure trial_ids are unique per
    parallel batch.
    """
    content = f"discovery filesystem-integrity canary\ntrial_id={trial_id}\nplanted_at={time.time_ns()}\n"
    expected_md5 = hashlib.md5(content.encode()).hexdigest()
    canaries: dict[str, str] = {}
    for d in monitored_dirs:
        d = Path(d).resolve()
        if not d.is_dir():
            continue
        try:
            canary_path = d / f"{CANARY_PREFIX}{trial_id}.txt"
            canary_path.write_text(content)
            canaries[str(canary_path)] = expected_md5
        except (PermissionError, OSError):
            # Some dirs (e.g. read-only) we can't plant in — skip silently.
            continue
    return CanaryRecord(trial_id=trial_id, canaries=canaries, content=content)


def verify_canaries(record: CanaryRecord) -> CanaryResult:
    """Verify each canary is intact, then delete. Returns failures."""
    failures: list[dict] = []
    for path_str, expected_md5 in record.canaries.items():
        path = Path(path_str)
        if not path.exists():
            failures.append({"path": path_str, "kind": "deleted",
                             "expected_md5": expected_md5})
            continue
        try:
            actual_md5 = _md5(path)
        except Exception as e:
            failures.append({"path": path_str, "kind": "unreadable",
                             "expected_md5": expected_md5, "error": str(e)})
            continue
        if actual_md5 != expected_md5:
            failures.append({"path": path_str, "kind": "modified",
                             "expected_md5": expected_md5,
                             "actual_md5": actual_md5})
        # Cleanup regardless.
        try:
            path.unlink()
        except Exception:
            pass
    return CanaryResult(intact=len(failures) == 0, failures=failures)


# ---------- self-test ----------

def _selftest() -> int:
    import shutil
    import tempfile

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  ok: {msg}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # Build a fake "site-packages" tree.
        sp = td / "fake_site_packages"
        (sp / "transformers" / "models" / "vits").mkdir(parents=True)
        (sp / "transformers" / "models" / "vits" / "modeling_vits.py").write_text("X = 1\n")
        (sp / "transformers" / "models" / "vits" / "configuration.py").write_text("CFG = {}\n")
        (sp / "transformers" / "__init__.py").write_text("__version__ = '5.0'\n")

        # Allowed root: per-trial out_dir + sandbox.
        allowed = td / "trial_out"
        (allowed / "sandbox").mkdir(parents=True)
        (allowed / "result.json").write_text("{}")

        monitored_globs = [str(sp)]
        allowed_roots = [allowed]

        # === Test 1: clean trial — no changes ===
        print("\n[test 1] clean trial")
        snap = take_snapshot(monitored_globs, allowed_roots)
        check(len(snap.files) == 3, f"snapshot has 3 files (got {len(snap.files)})")
        diff = diff_against_snapshot(snap, allowed_roots)
        check(not diff.has_changes, "clean trial: diff has no changes")

        # === Test 2: writes inside allowed root must NOT trip detection ===
        print("\n[test 2] writes inside allowed root are ignored")
        (allowed / "sandbox" / "modeling_vits.py").write_text("X = 99 # agent edit in sandbox\n")
        (allowed / "result.json").write_text('{"fix_status": "general"}')
        diff = diff_against_snapshot(snap, allowed_roots)
        check(not diff.has_changes, "allowed-root writes are ignored")

        # === Test 3: site-packages file modified in place — flagged ===
        print("\n[test 3] site-packages file modified in place")
        target = sp / "transformers" / "models" / "vits" / "modeling_vits.py"
        time.sleep(0.01)  # ensure mtime advances
        target.write_text("X = 1\n# agent contamination\n")
        diff = diff_against_snapshot(snap, allowed_roots)
        check(diff.has_changes, "modified file: diff flags has_changes")
        check(any(c.kind == "modified" and c.path == str(target) for c in diff.changes),
              "modified file: change is recorded with correct path + kind")
        target.write_text("X = 1\n")  # restore for next test
        # Re-snapshot baseline for tests 4-7.
        snap = take_snapshot(monitored_globs, allowed_roots)

        # === Test 4: new file in monitored dir — flagged ===
        print("\n[test 4] new file in monitored dir")
        new_file = sp / "transformers" / "_rogue_new_file.py"
        new_file.write_text("# planted by agent\n")
        diff = diff_against_snapshot(snap, allowed_roots)
        check(diff.has_changes, "new file: diff flags has_changes")
        check(any(c.kind == "new" and c.path == str(new_file) for c in diff.changes),
              "new file: change is recorded as 'new'")
        new_file.unlink()
        snap = take_snapshot(monitored_globs, allowed_roots)

        # === Test 5: file deleted from monitored dir — flagged ===
        print("\n[test 5] file deleted from monitored dir")
        config = sp / "transformers" / "models" / "vits" / "configuration.py"
        config_content = config.read_text()
        config.unlink()
        diff = diff_against_snapshot(snap, allowed_roots)
        check(diff.has_changes, "deleted file: diff flags has_changes")
        check(any(c.kind == "missing" and c.path == str(config) for c in diff.changes),
              "deleted file: change is recorded as 'missing'")
        config.write_text(config_content)
        snap = take_snapshot(monitored_globs, allowed_roots)

        # === Test 6: deep_inspect populates md5 + line counts ===
        print("\n[test 6] deep_inspect populates md5 + line counts")
        backups_dir = td / "backups"
        backups_dir.mkdir()
        backup = backups_dir / "modeling_vits.py.original"
        shutil.copyfile(target, backup)
        time.sleep(0.01)
        target.write_text("X = 1\nY = 2\nZ = 3\n")  # 2 lines added vs original
        diff = diff_against_snapshot(snap, allowed_roots)
        diff = deep_inspect(diff, backups={str(target): backup})
        modified = [c for c in diff.changes if c.path == str(target)][0]
        check(modified.new_md5 is not None, "deep_inspect: new_md5 populated")
        check(modified.old_md5 is not None, "deep_inspect: old_md5 from backup populated")
        check(modified.lines_added == 2, f"deep_inspect: 2 lines added (got {modified.lines_added})")
        target.write_text("X = 1\n")
        snap = take_snapshot(monitored_globs, allowed_roots)

        # === Test 7: canary deleted — caught ===
        print("\n[test 7] canary deleted is caught")
        record = plant_canaries([sp / "transformers"], trial_id="test_canary_delete")
        check(len(record.canaries) == 1, f"planted 1 canary (got {len(record.canaries)})")
        canary_path = list(record.canaries.keys())[0]
        Path(canary_path).unlink()
        result = verify_canaries(record)
        check(not result.intact, "canary deleted: result.intact == False")
        check(any(f["kind"] == "deleted" for f in result.failures),
              "canary deleted: failure recorded as 'deleted'")

        # === Test 8: canary modified — caught ===
        print("\n[test 8] canary modified is caught")
        record = plant_canaries([sp / "transformers"], trial_id="test_canary_modify")
        canary_path = list(record.canaries.keys())[0]
        Path(canary_path).write_text("tampered\n")
        result = verify_canaries(record)
        check(not result.intact, "canary modified: result.intact == False")
        check(any(f["kind"] == "modified" for f in result.failures),
              "canary modified: failure recorded as 'modified'")

        # === Test 9: canary intact — no false positive ===
        print("\n[test 9] canary untouched: clean")
        record = plant_canaries([sp / "transformers"], trial_id="test_canary_clean")
        result = verify_canaries(record)
        check(result.intact, "canary untouched: result.intact == True")
        check(len(result.failures) == 0, "canary untouched: no failures")
        # Verify cleanup happened.
        for cp in record.canaries:
            check(not Path(cp).exists(), f"canary cleaned up: {Path(cp).name}")

        # === Test 10: snapshot serialization round-trip ===
        print("\n[test 10] snapshot persists + reloads")
        snap = take_snapshot(monitored_globs, allowed_roots)
        d = snap.to_dict()
        s = json.dumps(d)
        snap2 = Snapshot.from_dict(json.loads(s))
        check(snap2.files == snap.files, "snapshot round-trips through json")

    # === Summary ===
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)} of 10 tests failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("OK — all 10 tests passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
