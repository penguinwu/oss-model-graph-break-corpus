# [{experiment_title}] {model_name} case

**Experiment:** [`{experiment_slug}`](https://github.com/penguinwu/oss-model-graph-break-corpus/tree/main/discovery/experiments/{experiment_slug})
**Master plan:** [`plan.md`](https://github.com/penguinwu/oss-model-graph-break-corpus/blob/main/discovery/experiments/{experiment_slug}/plan.md) — methodology, matrix, questions, what we record. Read this before launching.
**Umbrella:** #{umbrella_issue}
**Owner:** Otter
**Status:** Backlog (case file not yet authored)

---

This issue holds the **model-specific** setup, results, and discussion for case `{case_id}` (`{model_name}`). The methodology is in the master plan — do not restate it here.

## Case-specific setup (fill in when case is authored)

| Item | Value |
|---|---|
| Case ID | `{case_id}` |
| Model | `{model_name}` |
| Case file | `discovery/cases/{case_id}.py` (TBD) |
| Baseline (perf) | `discovery/cases/{case_id}.baseline.json` (TBD) |
| Pristine source backups | `/tmp/discovery-runs/{case_id}/*.original` (TBD) |
| Baseline eager output | `/tmp/discovery-runs/{case_id}/baseline_eager_output.pt` (TBD) |
| Validator | `/tmp/discovery-runs/{case_id}/validate.py` (TBD) |
| Source files agent may edit | TBD (model + test script) |
| Source files OFF-LIMITS | sdpa_attention.py, decomposition tables, anything outside the allowed set |

## Baseline numbers (no fix, with breaks)

(Fill in after pre-flight, before launch.)

- `graph_break_count`: TBD
- `eager_ms` (tier-1): TBD
- `compiled_ms` (tier-1): TBD
- `speedup`: TBD
- `compile_s`: TBD
- `max_diff_compiled_vs_eager` (baseline drift): TBD

## Launch command (when ready)

```bash
cd ~/projects/oss-model-graph-break-corpus
nohup /home/pengwu/envs/torch211/bin/python -m discovery.run_case \
  --case {case_id} \
  --variants V0,V2,V4,V6 \
  --skills none,/path/to/debug-graph-breaks/SKILL.md \
  --n 3 \
  --timeout 1800 \
  > /tmp/discovery-runs/{case_id}/launches/launch.log 2>&1 &
```

(Drop the second `--skills` arm to `none` only if running without the skill axis.)

## Trials

(Filled live as trials complete. One row per trial.)

| Trial | exit | elapsed | gb_final | max_diff | speedup | strategy summary |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## Final report (deliverable)

PR adds `discovery/experiments/{experiment_slug}/reports/{case_id}.md` per the master plan's report convention. PR description links back to this issue. Merge → this issue moves to Done.

## Status updates (comment thread)

Comments below should mark per-trial milestones, surprises, and pivots specific to this case. Cross-case observations belong on the umbrella (#{umbrella_issue}).
