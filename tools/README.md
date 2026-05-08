# tools/ — index for repo developers and agents

Every script in `tools/` should be listed here with a one-line synopsis and a use-when. If you add a new tool, add an entry. If a tool is missing from this index, that's a documentation bug — fix it.

User-facing entry points (`query.py`, `reproduce.py`) are also listed in the top-level [README.md](../README.md). This index covers everything, including the dev/agent-only tooling.

---

## Sweep operations

| Tool | Use when |
|---|---|
| **`run_experiment.py`** | Unified front-end for all sweep workflows. Subcommands: `sweep` (identify+explain), `explain`, `correctness`, `nightly`, `selftest`, `check-env`, `refresh-nightly`, `validate-shapes`, `merge`, `corpus`. **Start here for any sweep task.** Run `tools/run_experiment.py --help` for the full command list. |
| **`smoke_test.py`** | 3-model smoke for the full sweep pipeline. Run when validating sweep harness changes (`skills/test-sweep-changes/SKILL.md` Gate 2). |
| **`sweep_compare.py`** | Compare two sweep result dirs with invariant-checked partition (improvements / regressions / steady-state / new / removed). Used by post-sweep due diligence Step 2. |
| **`sweep_watchdog_check.py`** | Decision script (no side effects) that classifies an in-flight sweep as PROGRESSING / IDLE / HUNG / CRASHED / DONE based on PID + checkpoint progress + DONE marker. `--pass identify\|explain` selects which checkpoint files to read. Cron-installed by `skills/sweep.md` watchdog setup. |
| **`brief_data.py`** | Emit JSON snapshot of repo state (plans / experiments / commits / closed issues / open loops) for the daily-briefing skill. Cron-driven, but you can run on-demand to see what the daily brief sees. |
| **`generate_cohort.py`** | Generate a sweep cohort from a prior sweep's results, with provenance metadata (`_metadata.derived_from`, `source_versions`, `filter`, etc.). Standalone version of the `--save-cohort` mechanism inside `run_experiment.py sweep` — use when you want to BUILD a cohort without ALSO running the sweep. **Always use this (or `--save-cohort`) instead of hand-rolling cohort files** — bare flat lists fail sanity-check INV-A2. Filter syntax: `status in ok,graph_break` / `status == foo` / `status != foo`. Default fail-loud on empty/partial source_versions; `--allow-empty-versions` / `--allow-partial-versions` to override. Requires Python 3.9+. |
| **`derive_sweep_commands.py`** | Mechanically derive launch commands for gate / sample / full stages from one experiment config — the canonical multi-stage launcher (post-2026-05-07). All three stages use the SAME flags as the full launch; only sub-sample size + output dir vary. Modes: `--validate` / `--emit` / `--run`. Requires `settings.python_bin` and `settings.modellib_pins` in source config. Skip-to-full guardrail via `/tmp/derive_sweep_state/` (override: `--allow-skip-gate`). |
| **`check_cohort_invariants.py`** | Mechanical executor of `skills/sweep_sanity_check.md` invariants. Pre-launch on cohort files: A2/A3/A4. Post-sweep (`--post-sweep`) on results files: SP1 (spec provenance), A1, C1/C2, D1 (catastrophic divergences `max_diff > 1e-3`), D2 (noise-floor), G1 (untriaged). Exit 1 = STRICT_FAIL. Run after EVERY sweep before declaring it "good." |

## Analysis

| Tool | Use when |
|---|---|
| **`query.py`** | Query the graph break corpus. User-facing primary tool. Examples: `--status graph_break`, `--error deepcopy`, `--source hf`. |
| **`reproduce.py`** | Reproduce a graph break for a single model. Use when you need to investigate a specific failure. Pair with `--explain` to get break analysis. |
| **`analyze_sweep.py`** | Status breakdown of a sweep result by source / variant / mode. First thing to run on raw sweep output. |
| **`analyze_explain.py`** | Graph break taxonomy + counts from explain pass results. Used to build the comparison reports. |
| **`analyze_trend.py`** | Graph break trends across PyTorch versions. Use when comparing nightly N vs nightly N+1. |
| **`compare.py`** | Compare two sweep results or corpus snapshots — high-level diff. Use when checking "did my change move the numbers?" |
| **`compare_results.py`** | Compare experiment-runner results against sweep results (was the experiment representative?). |
| **`compare_explain_methods.py`** | Compare old explain pass vs new TORCH_LOGS counting backend on the same models. Methodology validation, not for routine analysis. |
| **`per_source_stats.py`** | Per-source breakdown for a sweep result file. Quick numbers without writing your own SQL. |
| **`daily_summary.py`** | Gather mechanical activity data for the daily-summary cron. Run on-demand if you want to see today's activity snapshot. |
| **`generate_nightly_summary.py`** | Generate the markdown report from a nightly sweep result. Used by the nightly cron at `tools/run_experiment.py nightly`. |

## Corpus management

| Tool | Use when |
|---|---|
| **`update_corpus.py`** | Merge sweep results into `corpus/corpus.json` with changelog generation. Run after every successful sweep that should update the canonical corpus. |
| **`validate.py`** | Corpus + tool integrity check. Run before publishing a release or before any cross-version comparison. |
| **`generate_index.py`** | Regenerate `docs/index.html` (the browsable corpus dashboard). Run after `update_corpus.py`. |
| **`generate_traces.py`** | Pre-generate tlparse HTML reports for top-N models by graph break count. Used to populate the traces section of the dashboard. |
| **`generate_trace_reports.py`** | Batch-generate tlparse HTML for all trace dirs. Heavier than `generate_traces.py`; used during full corpus refreshes. |
| **`amend_sweep.py`** | Amend a completed sweep with verified post-fix results (e.g., when a few models had a transient issue and were re-run individually). |

## Issue management

| Tool | Use when |
|---|---|
| **`file_issues.py`** | Post-sweep issue management. Subcommands include `pytorch-upstream` (file/comment with reproduction; **mandatory entry point per `skills/sweep.md` trigger** for any pytorch issue with a repro), `sweep-report`, `sweep-update`, `correctness-report`, `correctness-apply`. |
| **`check_filed_issues.py`** | Check status of issues we've filed across external repos. Detects new comments, status changes, and merges. Cron-friendly. |
| **`github_issue_monitor.py`** | Monitor GitHub issues for new activity and alert via GChat. Background-run; configure per project. |
| **`pr_landing_check.py`** | Authoritative pytorch/pytorch PR landing check — handles ghstack rewriting where `git log` lies. Use this, not `git log`, to confirm a PR landed. |
| **`check_pr_status.py`** | Lightweight PR status check (open / merged / closed). Faster than `pr_landing_check.py` when ghstack-correctness isn't needed. |
| **`queue_task.py`** | Add a card to the project board's Backlog. Use whenever you commit to deferred work — captures the commitment so it survives session end. |

## Monitoring & hygiene

| Tool | Use when |
|---|---|
| **`feedback_monitor.py`** | Triage new messages from the GChat feedback space. Maintains separate `processed_messages` and `replied_messages` state (see source for `list_needs_reply()` and `mark_replied()`). Cron-driven; primary consumer is the daily routine. |
| **`check_experiments.py`** | Drift detector for the discovery-experiment convention. Surfaces experiments whose dir layout or plan.md state is malformed. Used by the daily plan-brief cron. |
| **`check_plan.py`** | Walk all `**/plan.md` files, parse frontmatter, surface plans that are `STALE` (last_check >3 days) or `needs-input >=2 days`. Forcing function for plan freshness. |

## Environment / venv management

| Tool | Use when |
|---|---|
| **`bootstrap_modellibs.py`** | Provision standalone modellibs trees per `sweep/modellibs.json`. Run once per machine to populate `~/envs/modellibs/<lib>-<ver>/`. Required before any sweep that uses `--transformers`/`--diffusers`/`--timm`. |
| **`cleanup_modellibs_from_venvs.py`** | Phase 5 of venv-decouple-modellibs migration: remove modellibs from torch venvs (they live in dedicated dirs now). One-time per venv; generates a restore script for safety. |
| **`restore_modellibs_to_venvs.sh`** | Restore script auto-generated by `cleanup_modellibs_from_venvs.py`. Run only if you need to roll back the migration. |
| **`version_check.py`** | Pre-sweep environment validation: check installed package versions against corpus metadata. Called by `tools/run_experiment.py check-env`; rarely run directly. |

## Maintenance / debugging

| Tool | Use when |
|---|---|
| **`fix_worker.py`** | Investigate and patch dimension-mismatch errors (the "Cluster D" failure pattern). Use when a model's input-gen consistently fails with shape errors. |

---

## Conventions

- New tools must add an entry here in the same commit. CI doesn't enforce yet (open loop), but the discipline is mandatory.
- **New tools must also add `tools/test_<name>.py` in the same commit** (per [`docs/testing.md`](../docs/testing.md) — TDD is non-negotiable in this repo).
- **Bug fixes to existing tools must add a failing-then-passing test in the same commit** (regression guard).
- One-line synopsis = top of module docstring.
- Use-when describes the trigger, not the mechanism.
- Group by the categories above; if you need a new category, add it.
- Tests (`test_*.py`) and proxy modules (`_gh_proxy.py`) are not user-callable and excluded from this index.
