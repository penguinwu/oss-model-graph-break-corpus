# Open Loops

Things we noticed that deserve attention — issues, design improvements, open questions, identified risks. Tracked so they don't disappear.

**Default: fix immediately.** Most loops can be closed in minutes. This file is for exceptions — genuinely blocked or needs Peng's input. Every deferred item has a stated reason.

## Status Key
- `open` — identified, not yet addressed (with reason why deferred)
- `in-progress` — actively being worked on
- `closed` — fixed, with date and commit/PR
- `wont-close` — decided not to fix, with reason

---

## Open

### 1. Orchestrator architecture: coordinator holds data
- **Identified:** 2026-04-17 (overnight data loss post-mortem)
- **Why deferred:** Structural refactor, mitigated by streaming callback for now
- **Risk:** Medium — orchestrator still owns loop state; callback is a bolt-on
- **Description:** `run_pass()` accumulates results in memory and owns both coordination and data collection. Ideally workers write their own results, orchestrator just coordinates.

### 7. Experiment and sweep produce different output formats
- **Identified:** 2026-04-17
- **Why deferred:** Comparison script works as interim bridge
- **Risk:** Medium — comparing results requires custom tooling; undermines "one system" goal
- **Description:** Experiment runner uses `{model, config, mode, status}`, sweep uses `{name, source, mode, status}` with different structure.

### 8. Hardcoded backend and fullgraph in worker
- **Identified:** 2026-04-16
- **Why deferred:** Current use case is graph breaks only; no other backend needed yet
- **Risk:** Medium — blocks use cases beyond graph break analysis
- **Description:** `worker.py` hardcodes `torch.compile(model, fullgraph=True, backend="eager")`. Backend and fullgraph should be configurable via experiment config.

### 9. Model enumeration tied to HF/diffusers/custom/timm
- **Identified:** 2026-04-16
- **Why deferred:** Current sources cover the corpus well; no new sources planned
- **Risk:** Low — adding a new source requires code changes
- **Description:** `resolve_models()` has hardcoded suite names. Plugin/registry pattern would be cleaner.

## Closed

### 2. No signal handling → data loss on kill
- **Closed:** 2026-04-17 (commit d10d0c9)
- **Description:** Added SIGTERM/SIGINT handler that flushes files, writes INTERRUPTED marker, exits cleanly.

### 3. Redundant checkpoint.jsonl
- **Closed:** 2026-04-17 (commit d10d0c9)
- **Description:** Removed checkpoint.jsonl; resume reads from results.jsonl directly.

### 4. Retry appends instead of replacing
- **Closed:** 2026-04-17 (commit 8d19f05)
- **Description:** Retries appended duplicate entries. Now deduplicates results.jsonl on clean completion, keeping last entry per model/config/mode.

### 5. Config mixes portable and machine-specific concerns
- **Closed:** 2026-04-17
- **Description:** Already handled: `SWEEP_PYTHON` env var overrides config, documented in README. Portable configs omit `python_bin`; machine-specific configs include it.

### 6. No auto-resume on failure
- **Closed:** 2026-04-17 (commit d10d0c9)
- **Description:** Added `scripts/run_with_retry.sh` — auto-resume wrapper with MAX_RETRIES=3.
