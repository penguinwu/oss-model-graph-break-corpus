# [{experiment_title}] {model_name} case

**Experiment:** [`{experiment_slug}`](https://github.com/penguinwu/oss-model-graph-break-corpus/tree/main/discovery/experiments/{experiment_slug})
**Master plan:** [`plan.md`](https://github.com/penguinwu/oss-model-graph-break-corpus/blob/main/discovery/experiments/{experiment_slug}/plan.md) — methodology, matrix, questions, what we record. Read this before launching.
**Analysis methodology:** [`per-case-analysis SKILL`](https://github.com/penguinwu/oss-model-graph-break-corpus/blob/main/discovery/skills/per-case-analysis/SKILL.md) — 7-phase blueprint (Phase 0 audit + A-F). Follow this when the run completes.
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

## Pre-launch checklist

### Canonical (snapshotted from master plan on {snapshot_date})

{canonical_checklist}

### Case-specific additions

(Add case-specific items below — quirks like train mode, M-RoPE config, etc. Empty if no quirks.)

- [ ] (none yet)

If any item above is unchecked, fix or document the deviation BEFORE launching.

## Launch command (when ready)

**Standard matrix (always run, ~6 hr wall):** V0+V2 only — see master plan §"Methodology" for why V4/V6 are conditional.

```bash
cd ~/projects/oss-model-graph-break-corpus
nohup /home/pengwu/envs/torch211/bin/python -m discovery.run_case \
  --case {case_id} \
  --variants V0,V2 \
  --skills none,/home/pengwu/projects/oss-model-graph-break-corpus/discovery/skills/debug-graph-breaks/SKILL.md \
  --n 3 \
  --timeout 1800 \
  > /tmp/discovery-runs/{case_id}/launches/launch.log 2>&1 &
```

**Conditional follow-ups (only if Phase B triggers):**

- V4 trigger: any V0/V2 trial used a canonical escape hatch (`custom_op` / `disable` / `cond` / `allow_in_graph` / `nonstrict_trace` / `leaf_function`) or `torch.compiler.is_compiling()`. Relaunch with `--variants V4`.
- V6 trigger: any V0/V2 trial flipped a `torch._dynamo.config` flag. Relaunch with `--variants V6`.

(Drop the second `--skills` arm to `none` only if running without the skill axis.)

## Trials

(Filled live as trials complete. One row per trial.)

| Trial | exit | elapsed | gb_final | max_diff | speedup | strategy summary |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## Post-run checklist

Tick after the harness completes and before declaring the case done.

- [ ] *Phase 0 audit clean* (per `per-case-analysis SKILL`): artifact completeness, .original SHA-verified against pristine source, no serious flags, internal consistency, value bounds sane, stream integrity OK.
- [ ] *validation_v2 added* to all trial result.json files (run `python -m discovery.revalidate --case {case_id} --run-id <run_id>` if not native to the per-case validate.py).
- [ ] *fingerprints.csv produced* per Phase A, classifying every trial.
- [ ] *Phase B aggregations* computed (fix_status × variant × skill table; perf distributions; strategy clusters).
- [ ] *Phase C-D-E walked* in the findings doc — questions answered, surprises documented, open observations flagged honestly (no invented mechanisms).
- [ ] *Findings doc landed* at `discovery/experiments/{experiment_slug}/reports/{case_id}/findings.md` per Phase F.
- [ ] *PR opened* adding the report; PR description links back to this issue.

## Final report (deliverable)

PR adds `discovery/experiments/{experiment_slug}/reports/{case_id}/findings.md` + `fingerprints.csv` per the master plan's report convention. PR description links back to this issue. Merge → this issue moves to Done.

The findings doc MUST open with a **Setup section** (per `per-case-analysis SKILL` Phase F template) that links back to: this issue, the umbrella (#{umbrella_issue}), the master plan, and a one-line glossary of variants + skill arms. The Setup section lets an outside reader land on the findings doc and orient without grepping the repo.

## Lessons learned (harvested at case wrap)

(Filled in when the case wraps. Anything this case taught us about the harness, the variant catalog, the fingerprint axes, the master plan, the methodology — short bullets. Cross-case synthesis pulls from these.)

## Status updates (comment thread)

Comments below should mark per-trial milestones, surprises, and pivots specific to this case. Cross-case observations belong on the umbrella (#{umbrella_issue}).
