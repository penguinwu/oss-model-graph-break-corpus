# Design Debt & Open Issues

Tracked issues from design reflections and post-mortems.
Review this before starting new features or sweeps.

## Status Key
- `open` — identified, not yet addressed
- `in-progress` — actively being worked on
- `done` — fixed, with date and commit/PR
- `wont-fix` — decided not to fix, with reason

---

## Issues

### 1. Orchestrator architecture: coordinator holds data
- **Status:** `open`
- **Identified:** 2026-04-17 (overnight data loss post-mortem)
- **Risk:** Medium — mitigated by streaming callback, but orchestrator still owns loop state
- **Description:** `run_pass()` accumulates results in memory and the orchestrator owns both coordination and data collection. If designed from scratch, workers would write their own results to a shared directory, orchestrator would just coordinate.
- **Mitigated by:** `result_callback` parameter (streaming writes), but this is a bolt-on, not a rethink.

### 2. No signal handling → data loss on kill
- **Status:** `done` (2026-04-17, commit d10d0c9)
- **Identified:** 2026-04-17 (overnight data loss post-mortem)
- **Description:** SIGTERM from nightly restart killed the process with no cleanup. Added signal handler that flushes files, writes INTERRUPTED marker, and exits cleanly.

### 3. Redundant checkpoint.jsonl
- **Status:** `done` (2026-04-17, commit d10d0c9)
- **Identified:** 2026-04-17
- **Description:** `results.jsonl` and `checkpoint.jsonl` received identical data after streaming was added. Removed checkpoint.jsonl; resume now reads from results.jsonl directly.

### 4. Retry appends instead of replacing
- **Status:** `open`
- **Identified:** 2026-04-17
- **Risk:** Low — merge/summary code uses last-wins logic, but the JSONL file contains both the timeout entry and the retry entry for the same model.
- **Description:** When a timed-out model retries successfully, both entries exist in results.jsonl. In-memory state is correct, but file has duplicates. Not a correctness bug (downstream handles it), but a data integrity smell.

### 5. Config mixes portable and machine-specific concerns
- **Status:** `open`
- **Identified:** 2026-04-17
- **Risk:** Medium — blocks other users from running our configs without editing them
- **Description:** `python_bin: "/home/pengwu/envs/torch211/bin/python"` is hardcoded. Device, paths, and worker counts are machine-specific but live alongside experiment-defining fields (models, configs, dynamo flags). Should separate "what to test" from "how to execute" — e.g., env var overrides or a separate execution profile.

### 6. No auto-resume on failure
- **Status:** `done` (2026-04-17, commit d10d0c9)
- **Identified:** 2026-04-17
- **Risk:** Was high — overnight kills required manual --resume
- **Description:** Added `scripts/run_with_retry.sh` wrapper that detects existing results and auto-resumes on non-zero exit. MAX_RETRIES=3.

### 7. Experiment and sweep produce different output formats
- **Status:** `open`
- **Identified:** 2026-04-17
- **Risk:** Medium — comparing results requires a custom script (tools/compare_results.py); undermines "one system" goal
- **Description:** Experiment runner writes flat JSONL with `{model, config, mode, status, ...}`. Sweep writes nested JSON `{metadata, results: [...]}` with different field names (`name` vs `model`, `source` field, extra fields like `gpu_mem_mb`). Converging these would make validation automatic.

### 8. Hardcoded backend and fullgraph in worker
- **Status:** `open`
- **Identified:** 2026-04-16 (generalization discussion)
- **Risk:** Medium — blocks use cases beyond graph break analysis (e.g., AOT eager, inductor)
- **Description:** `worker.py` hardcodes `torch.compile(model, fullgraph=True, backend="eager")`. To support other use cases (success rate testing, inductor performance), backend and fullgraph should be configurable via experiment config. Not urgent — current use case is graph breaks only — but needed for the "run your own experiments" story.

### 9. Model enumeration tied to HF/diffusers/custom/timm
- **Status:** `open`
- **Identified:** 2026-04-16
- **Risk:** Low — current sources cover the corpus well
- **Description:** `resolve_models()` has hardcoded suite names. Adding a new source requires code changes. A plugin/registry pattern would be cleaner but isn't needed until we have more sources.
