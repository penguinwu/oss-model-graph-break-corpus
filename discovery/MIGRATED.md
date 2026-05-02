# discovery/ has MIGRATED

**Authoritative copy:** [penguinwu/pt2-skill-discovery](https://github.com/penguinwu/pt2-skill-discovery)
**Local clone:** `~/projects/pt2-skill-discovery/`
**Migration completed:** 2026-05-02 (Phase 6a per `MIGRATION_PLAN.md`)

## DO NOT EDIT FILES IN THIS DIRECTORY

Edits made here will not flow to the new repo. The contents of `corpus/discovery/` are kept on disk as a fallback during the dual-run period (Tier 6 of the migration plan), but **all new discovery work must go to pt2-skill-discovery**.

This banner is itself the audit trail: if you find yourself about to edit something in `corpus/discovery/`, stop and switch context to the new repo.

## Where things moved

| Was here | Now lives at |
|---|---|
| `corpus/discovery/{runner,run_config,launch_parallel,merge_results,revalidate,validate_runner,perf,_measure_case,variants,_lifecycle_gate,clean_sandboxes,smoke_test}.py` | `pt2-skill-discovery/scripts/` |
| `corpus/discovery/cases/*.py` | `pt2-skill-discovery/cases/` |
| `corpus/discovery/experiments/*` | `pt2-skill-discovery/experiments/` |
| `corpus/discovery/skills/*` | `pt2-skill-discovery/skills/` |
| `corpus/discovery/{design,EXPERIMENT_LIFECYCLE}.md` | `pt2-skill-discovery/design/` |

The new repo has its own `CLAUDE.md` and operating rules. The `from sweep.explain import run_graph_break_analysis` import was duplicated to `pt2-skill-discovery/scripts/explain_helper.py` (~75 LOC, self-contained — no runtime dep on corpus).

## Cold-start verification

Phase 5 Tier 4 (cold-start isolation) passed on 2026-05-02: with `corpus/discovery/` temporarily moved aside, all of pt2-skill-discovery's smoke tests (Layer 1 synthetic + Layer 2 6/6 cases) and utilities (merge_results, revalidate, clean_sandboxes) ran cleanly. The new repo has zero runtime dependencies on this directory.

## Phase 6b (deletion) is gated on

≥1 real workload completing in pt2-skill-discovery without falling back to this dir, then the contents will be `git rm`'d. Until then, this dir stays on disk read-only-by-convention.
