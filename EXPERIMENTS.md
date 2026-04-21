# Experiments Index

Canonical baselines for each experiment type, used to validate invariants when extending or comparing sweeps. See `CLAUDE.md` § "Validate Invariants Before Reporting New Experiments" for the rule.

**Maintenance:** add a row when a new experiment type ships. Update the canonical-result path when a new authoritative sweep lands. If a row's result file isn't on disk, the row doesn't get added (Closure Discipline).

## Active experiments

| Experiment | Dimension tested | Canonical result | PT version | Status enum | Invariants under extension |
|---|---|---|---|---|---|
| **Graph-break sweep (default)** | `torch.compile(fullgraph=True, backend="eager")` | `corpus/corpus.json` (per-version stratified) + `sweep_results/pt2.11/identify_results.json` (raw) | 2.11.0 | `full_graph` / `graph_break` / `compile_error` / `create_error` / `eager_error` / `timeout` | `create_error`, `eager_error`, `timeout` sets are invariant on the same PT + model set |
| **Phase 3 correctness** | eager output vs compiled output (same input, same seed) | `correctness/correctness_results.json` | 2.11.0 (will re-baseline on nightly) | currently `match` / `divergence` / `nan_inf` / `shape_mismatch` — vocab migrating to `pass` / `verification_failure` / `nan_inf` / `shape_mismatch` (2026-04-21) | `create_error` and `eager_error` sets must match the graph-break baseline on the same PT + HF model set. *Currently failing this invariant — Phase 3 worker has a wrapper-creation bug.* |

## How to use this for invariant validation

1. Identify the new experiment's dimension (the thing it varies).
2. Pick the comparable baseline (same PT version, same model set, same source filter).
3. Compute the upstream-status sets (`create_error`, `eager_error`, etc.) on both.
4. Check: any model that's `create_error` in the new sweep but a healthy status in baseline? That's a methodology violation. Report it first.

Example invariant query (Phase 3 vs graph-break baseline):

```python
# Phase 3 create_errors that should not be create_errors
new_ce = {r['name'] for r in p3['results'] if r['mode']=='eval' and r['status']=='create_error'}
baseline_healthy = {r['name'] for r in p11['results'] if r.get('mode')=='eval' and r.get('config')=='default' and r['status'] in ('full_graph','graph_break')}
violations = new_ce & baseline_healthy
# Phase 3 had 169 violations on 2026-04-20 — methodology bug
```

## Planned / not yet shipped

| Experiment | Dimension | Notes |
|---|---|---|
| Dynamic shapes | trace with `dynamic=True` or `mark_dynamic` | Static-shape statuses must match a default sweep |
| Inductor backend | `backend="inductor"` | Eager-backend statuses must match default sweep |
| Train-mode correctness | eager-train vs compiled-train output + grad | Phase 3 V2 — needs forward-grad comparison rules |

## Out of scope

- TIMM full sweep (deferred — see `OPEN-LOOPS.md`)
- Diffusers correctness (Phase 3 V2)
