# Sweep Experiment Spec — Design Doc

**Revision:** 4
**Owner:** otter (with Peng-in-the-loop)
**Created:** 2026-05-07
**Status:** draft — pending Peng approval before implementation. Rev 4 incorporates rev 3 adversary review + Peng's directive to include `derive_spec_from_prior.py` in v1 (because it directly supports the NGB verify re-sweep — the immediate next launch we want to do).

---

## 1. Problem statement

When launching a sweep experiment, three commands must be issued in sequence:

1. **Gate** — a small (~5 model) mini-sweep that validates the launch path on a microcosm before committing real compute
2. **Sample** — a slightly larger (~20 model) sub-sample sweep that validates cohort + stack at moderate cost (~15 min)
3. **Full** — the actual canonical sweep on the full cohort (~hours)

For these to give meaningful confidence, all three must use **identical configuration** except for the cohort sub-sample size and minor stage-specific overrides. The sample's success implies the full launch's reproducibility; the gate's success implies the sample's reproducibility. **Drift between any two stages destroys the confidence chain.**

Today (2026-05-07) we observed **three separate failures of this confidence chain in one week**:

- **2026-05-06 broken cohort:** the full canonical NGB verify launched without a sample-sweep gate — broken cohort discovered hours later.
- **2026-05-07 morning, generic gates passed but launch path differed:** standard `test-sweep-changes` 5 gates passed on torch211 + canary models, but the M2 launch was on torch-nightly-cu126 + cohort models. Standard gates didn't validate the launch path.
- **2026-05-07 afternoon, customized gate matched my (incomplete) launch plan, both wrong:** even after explicitly customizing the 5-gate to the planned M2, BOTH the gate and the M2 plan were missing `--compile-kwargs '{"fullgraph": false}'` and `--dynamo-config nested_graph_breaks=true` because I derived flags from MEMORY of NGB verify, not from the prior NGB run's recorded `args`.

In all three cases, discipline was the only guardrail and discipline failed. **We need a mechanical solution.**

## 2. Goal

Eliminate drift between gate, sample, and full launches **by construction** — make them all derive from a single declarative spec, so that hand-typing any launch flag becomes the wrong-by-construction path.

The spec is the **single source of truth** for an experiment's configuration. A script reads the spec and emits the exact bash command for any stage. There is no other supported way to launch a sweep.

Additionally, the spec system must support **grounding in prior runs**: a script that reads a prior run's `sweep_state.json` and emits a spec, so reproducing a prior experiment doesn't depend on memory of what flags were used.

## 3. Spec format

JSON (Python stdlib parses it; no extra dependency; easier to mechanically validate than YAML; CI-friendly).

Spec files live at `experiments/specs/<name>.json`. The `name` field is the canonical experiment identifier and must match the filename.

### 3.1 Required fields

```json
{
  "spec_version": 1,
  "name": "ngb-verify-2026-05-07",
  "description": "NGB verify pass on the regenerated 190-model cohort, right stack",
  "venv": "~/envs/torch-nightly-cu126/bin/python",
  "torch": "2.13.0.dev20260502+cu126",
  "modellibs": {
    "transformers": "5.6.2",
    "diffusers": "0.38.0"
  },
  "cohort": {
    "derive_from": "sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json",
    "filter": "status == ok",
    "source_sha256": "<hash>"
  },
  "modes": ["eval", "train"],
  "workers": 2,
  "timeout_s": 180,
  "passes": ["identify"],
  "compile_kwargs": {"fullgraph": false},
  "dynamo_config": {"nested_graph_breaks": true}
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `spec_version` | int | yes | Spec format version. Currently `1`. Validator REFUSES unknown versions. |
| `name` | string | yes | Canonical experiment identifier; MUST match filename (`<name>.json`) |
| `description` | string | yes | One-line human-readable purpose; appears in launch report |
| `venv` | string | yes | Absolute path to Python interpreter (script expands `~`) |
| `torch` | string | yes | Expected `torch.__version__` from the venv. Validator probes the venv at validate-time and REFUSES on mismatch. |
| `modellibs` | object | yes | Map of `package` → `version`. Each MUST have `~/envs/modellibs/<pkg>-<ver>/` present. Empty `{}` only if launch uses ZERO modellibs (rare). |
| `cohort` | object | yes | Block with EXACTLY ONE form: `derive_from + filter` / `path` / `inline`. See §3.1.5. |
| `modes` | array<string> | yes | Subset of `["eval", "train"]`, non-empty |
| `workers` | int | yes | Worker count (1-8) |
| `timeout_s` | int | yes | Per-model timeout in seconds, positive |
| `passes` | array<string> | yes | Subset of `["identify", "explain"]`, non-empty, in canonical order if both |
| `compile_kwargs` | object | yes | Passed verbatim as `--compile-kwargs <json>`. Use `{}` for defaults. **Required** to force explicit declaration — no defaulting. |
| `dynamo_config` | object | yes | Passed verbatim as repeated `--dynamo-config <key=value>`. Use `{}` for none. **Required** to force explicit declaration. |

### 3.1.5 Cohort block — three forms

EXACTLY ONE of the following keys must be present in `cohort`:

**Form 1: derivation from a prior run (PREFERRED — self-describing)**
```json
"cohort": {
  "derive_from": "sweep_results/experiments/<prior-run>/<results-file>.json",
  "filter": "status == ok",
  "source_sha256": "7f3a..."
}
```

| Sub-field | Required | Description |
|---|---|---|
| `derive_from` | yes | Path to a prior run's results file (typically `identify_results.json` or `explain_results.json`) |
| `filter` | yes | Filter expression (see §3.1.6) |
| `source_sha256` | yes | SHA256 of the source file at spec-creation time. Validator REFUSES if source's current sha256 doesn't match. Forces awareness when prior run is regenerated. v1 expects the spec author to compute this manually (`sha256sum <source-file>`); v1.1 may add a `--refresh-content-hashes` helper. **Placeholder handling:** if the spec author writes a literal `<sha256 captured at spec-creation time>` (or any value matching `<.*>` literal), the validator REFUSES with a help message containing the exact `sha256sum <derive_from-path>` command — distinct from a generic "sha256 mismatch" error. (Closes rev-3 adversary new gap #1.) |

(`cohort.cache` was in rev 2 but is dropped from v1 per Peng's "start simple" — it was documentation-only and created a stale-cache failure mode. v1.1 may add it back if there's demand.)

**Form 2: pre-built cohort snapshot**
```json
"cohort": {
  "path": "experiments/configs/nested_gb_cohort_2026-05-06.json",
  "expected_sha256": "a91f..."
}
```

| Sub-field | Required | Description |
|---|---|---|
| `path` | yes | Path to a cohort file (canonical with `_metadata` block) |
| `expected_sha256` | yes | SHA256 of the cohort file. Validator REFUSES on drift. |

**Form 3: ad-hoc inline (rare; deferred to v2)**
```json
"cohort": {
  "inline": [
    {"name": "Bart", "source": "hf"},
    {"name": "ConvBert", "source": "hf"}
  ]
}
```
Requires `"bare-list"` in `spec.allow_flags`. Validator REFUSES otherwise.

**Why derivation form is preferred:** self-describing (reader sees the WHY), reproducible (regenerate cohort from spec), source provenance baked in, composable bumps (change filter or source), aligns with `--models-from` which is the documented canonical CLI path.

### 3.1.6 Filter expression syntax

Same grammar as `tools/generate_cohort.py`:
- `status == <value>`
- `status != <value>`
- `status in <comma-separated values>`

v1 grammar; v2 may add `wall_time_s < N` etc.

### 3.2 Optional fields (with defaults)

| Field | Type | Default | Description |
|---|---|---|---|
| `device` | string | `"cuda"` | `cuda` or `cpu` |
| `auto_retry` | bool | `true` | If `false`, passes `--no-auto-retry` |
| `inductor_config` | object | `{}` | Repeated `--inductor-config <key=val>` |
| `dynamic_dim` | string\|null | `null` | `batch` / `all` / `null` (no flag) |
| `limit` | int\|null | `null` | If set, applies to all stages on top of sub-sample size; gate/sample use `min(stage_size, limit)`; full uses `--limit N`. Validation: `limit < gate_size` REJECTED. |
| `stability` | string\|null | `null` | `stable` / `unstable` / `null` |
| `setup_script` | string\|null | `null` | Path to `--setup-script` |
| `strict_modellibs` | bool | `true` | **Default ON** for spec-emitted commands. Override only via `allow_flags: ["non-strict-modellibs"]`. |
| `cuda_variant` | string\|null | `null` | `cu128` / `cu126` / `null` |
| `allow_flags` | array<string> | `[]` | Subset of `["bare-list", "empty-versions", "partial-versions", "stale-cohort", "version-mismatch", "non-strict-modellibs"]`. **Note:** `bare-list` is auto-added for gate/sample only if Phase 6 sub-sample preservation isn't yet landed (see §3.4). For Form 3 (inline), MUST be set explicitly. |
| `version_mismatch_allow` | array<string> | `[]` | Per-package allow list for version-coherence violations. Subset of `["torch", "transformers", "diffusers", "timm"]`. Used by §4.3 asymmetric check: a cohort declares package X but spec doesn't → REFUSED unless X is in this list. The bare `version-mismatch` allow_flag (above) is BROAD — applies to all packages. This field is NARROW — applies only to listed packages. (Replaces rev 2's parameterized `version-mismatch:<pkg>` syntax — separate field is cleaner per adversary feedback.) |
| `output_dir_template` | string | `"sweep_results/experiments/{name}-{stage}-{date}-{time}"` | Format vars: `{name}`, `{stage}`, `{date}` (YYYY-MM-DD), `{time}` (HHMMSS). HHMMSS prevents same-day collision. |
| `run_name_template` | string | `"{name}-{stage}"` | |
| `gate_size` | int | `5` | Models in gate stage |
| `sample_size` | int | `20` | Models in sample stage |
| `sub_sample_seed` | int\|null | `null` | If `null` (default), uses `tools/sample_cohort.py`'s `_seed_from_cohort` (sha256 of cohort path + mtime — anchors sample to cohort content). If integer, passes as `--seed N` for cross-cohort repro experiments. (Default changed from rev 2's `42`, which was a regression vs. `_seed_from_cohort`.) |
| `watchdog` | object | `{"enabled": true, "interval_min": 10, "post_to": "spaces/AAQANraxXE4"}` | Watchdog cron config |
| `wake_cron_eta_min` | int\|null | `null` | If set, schedule wake at +eta minutes |
| `post_sweep_check` | string | passes-aware: `"tools/check_cohort_invariants.py --post-sweep {output_dir}/{results_basename}.json"` | Canonical post-sweep check command. Wake-cron uses this. Format vars: `{output_dir}`, `{venv}`, `{name}`, `{stage}`, `{results_basename}`. `{results_basename}` is auto-derived from `passes`: `identify` if `passes == ["identify"]`, else `explain` (the last canonical pass). Per adversary feedback — rev 2 hardcoded `identify_results.json` which would be wrong for explain runs. |
| `nohup` | bool\|null | `null` | `null` = auto (foreground for gate/sample, nohup for full); explicit override applies to all stages |

### 3.3 Stage-specific derivations

Three stages: `gate`, `sample`, `full`. Each derives a launch command from the same spec, with these stage-specific transformations:

| Aspect | Gate | Sample | Full |
|---|---|---|---|
| Sub-sample size | `min(gate_size, limit)` | `min(sample_size, limit)` | `limit` if set, else full cohort |
| Sub-sample written to | `/tmp/<name>-gate.json` | `/tmp/<name>-sample.json` | N/A |
| `--models` / `--models-from` | Sub-sample → `--models` (NO `--allow-bare-cohort` because Phase 1.5 preserves _metadata in sub-samples — they are CANONICAL cohorts per §3.4) | Same as gate | If Form 1: `--models-from` + `--filter`; if Form 2: `--models <path>`; Form 3 deferred to v2 |
| `--output-dir` | template with `{stage}=gate` | template with `{stage}=sample` | template with `{stage}=full` |
| `--run-name` | template with `{stage}=gate` | template with `{stage}=sample` | template with `{stage}=full` |
| nohup | Foreground by default | Foreground by default | nohup by default |
| Watchdog | Disabled | Conditional (only if estimated runtime >30 min) | Required (per `sweep.md` §4) |
| Wake-cron | Disabled | Optional | If `spec.wake_cron_eta_min` set |

**Load-bearing invariant:** all other flags (workers, modes, timeout, modellibs, compile_kwargs, dynamo_config, dynamic_dim, stability, setup_script, strict_modellibs, etc.) come from spec verbatim. **No stage may override any spec field except as explicitly listed in this table.** This is the load-bearing invariant — gate/sample/full differ ONLY in cohort sub-sample + structurally-required flags.

### 3.4 Sub-sample _metadata preservation

`tools/sample_cohort.py` MUST be upgraded to preserve the parent cohort's `_metadata` block when writing a sub-sample. The sub-sample will additionally have:

```json
"_metadata": {
  ...all parent fields preserved (derived_from, filter, source_versions, etc.)...,
  "sub_sampled_from": "<parent cohort path>",
  "sub_sample_size": 5,
  "sub_sample_seed_resolved": 13188663043791988478,
  "sub_sample_parent_sha256": "<parent cohort sha256 at sample time>"
}
```

The recorded `sub_sample_seed_resolved` is the integer seed actually used (whether explicit from `spec.sub_sample_seed` or derived via `_seed_from_cohort` when spec is `null`). Reading the sub-sample's `_metadata` always reproduces the sample. (Field renamed from rev 2's `sub_sample_seed` per rev-3 adversary new gap #4 — the recorded value is the resolved int, not the spec setting.)

**Consequence:** sub-samples are CANONICAL cohorts (not bare lists). The validator runs full §4 validation on them. `--allow-bare-cohort` is NOT needed for spec-derived sub-samples. This closes adversary gap #14 cleanly: sub-samples carry provenance + are version-checkable.

The legacy bare-list mode for sub-samples is preserved for backward compatibility (`tools/sample_cohort.py --bare-list`) but the spec system never invokes it.

## 4. Validation rules (script-enforced before any command emitted)

### 4.1 Spec self-consistency

- `spec_version == 1`
- `name` field matches filename
- `venv` path exists; is executable
- `torch` field matches venv's actual `torch.__version__` (probed by script invoking `<venv>/python -c 'import torch; print(torch.__version__)'`)
- For each `pkg` in `modellibs`: `~/envs/modellibs/<pkg>-<ver>/` exists
- `cohort` block has EXACTLY ONE form key (`derive_from` / `path` / `inline`)
- `modes` non-empty subset of `["eval", "train"]`
- `passes` non-empty subset of `["identify", "explain"]` and in canonical order if both
- `workers` is positive integer ≤ 8
- `timeout_s` is positive integer
- `allow_flags` entries all from documented set
- `gate_size < sample_size`
- If `limit` set: `limit >= gate_size`
- All required fields present (NO defaulting of `compile_kwargs` / `dynamo_config` / `passes` / `modellibs` to empty — explicit declaration required)
- `dynamic_dim` (if set) is `"batch"` or `"all"`
- `stability` (if set) is `"stable"` or `"unstable"`
- `cuda_variant` (if set) is `"cu128"` or `"cu126"`

### 4.2 Modellibs ↔ source coherence

For each model source the cohort will use (computed from cohort form: derive_from path / cohort path's source / inline):
- If the source uses `transformers` and `transformers` is NOT in `spec.modellibs` → REFUSED (silent fall-through to venv copy is the 2026-05-06 failure mode)
- Same for `diffusers`, `timm`

### 4.3 Spec ↔ cohort version coherence (asymmetric)

Read cohort's effective `_metadata.source_versions`:
- For Form 1: read source file's `metadata.versions`
- For Form 2: read cohort file's `_metadata.source_versions`
- For Form 3: not applicable (no cohort metadata)

Then: **for every package in cohort's source_versions, the spec MUST have a matching entry** (in `torch:` field for torch, in `modellibs:` for others) with the SAME version. Asymmetric:
- Cohort declares X, spec declares X, match → OK
- Cohort declares X, spec declares ≠X → REFUSED unless `"version-mismatch"` (broad) in allow_flags OR `<pkg>` in `version_mismatch_allow` (narrow)
- Cohort declares X, spec doesn't declare → REFUSED unless `"version-mismatch"` (broad) in allow_flags OR `<pkg>` in `version_mismatch_allow` (narrow)
- Cohort doesn't declare, spec declares → OK (cohort may be older format)

This closes adversary gap #3 — the AND-clause hole in rev 1. The narrow `version_mismatch_allow` field replaces rev 2's parameterized `version-mismatch:<pkg>` allow_flag syntax (per rev 2 adversary feedback — separate field is cleaner).

### 4.4 Content pinning

- For Form 1: `cohort.source_sha256` must match current sha256 of `cohort.derive_from`. REFUSED on drift.
- For Form 2: `cohort.expected_sha256` must match current sha256 of `cohort.path`. REFUSED on drift.
- For Form 3: not applicable.

### 4.5 Stage internal consistency

- Sub-sample seed (`sub_sample_seed`) is deterministic — same spec → same sub-sample
- Output dir doesn't already exist (or `--force` flag passed to derive script)
- Re-emit on a stage where prior emission's output dir exists: validator computes the cohort sha256 from the prior stage's metadata (recorded by the launcher) — if mismatched, REFUSED unless `--force` (closes gap #5)
- For full stage with Form 1 or Form 2: cohort must NOT be bare unless `"bare-list"` in `allow_flags`

### 4.6 Path safety in emitted bash

The emitted bash MUST:
- Start with `set -euo pipefail`
- `cd` to repo root before launch
- Verify venv path exists at runtime: `[ -x "$VENV" ] || { echo "venv missing: $VENV"; exit 2; }`
- Verify cohort path exists at runtime: `[ -f "$COHORT" ] || { echo "cohort missing"; exit 2; }`
- Verify output dir doesn't already exist: `[ -d "$OUT" ] && { echo "output dir exists: $OUT (re-emit to get a fresh timestamp)"; exit 3; }` — this catches same-second re-runs of the same emission, closing rev-3 adversary new gap #2.

(Closes adversary gap #11 from rev 1 + rev-3 new gap #2.)

## 5. Workflows

### 5.1 New experiment (write spec from scratch)

1. Write `experiments/specs/<name>.json`
2. `tools/derive_sweep_commands.py specs/<name>.json --validate` — runs §4 checks
3. `tools/derive_sweep_commands.py specs/<name>.json --stage gate --emit` — prints exact bash
4. Copy-paste, run gate
5. If green: `--stage sample --emit`, run
6. If green: `--stage full --emit`, run

### 5.2 Reproduce a prior run

1. `tools/derive_spec_from_prior.py sweep_results/experiments/<prior-run>/ --venv <venv-path>` → emits a spec file
   - Re-uses `tools/run_experiment.py`'s argparse for parsing (DRY)
   - REQUIRES explicit `--venv` because `sweep_state.json` doesn't record resolved venv path (sweep_state records `versions.python` which is a fingerprint string, not a path) — closes adversary gap #8
2. Edit (e.g., bump versions, change cohort)
3. Continue from §5.1 step 2

### 5.3 Bump a library version

1. Copy prior spec to `experiments/specs/<name>-<new-version>.json`
2. Edit `modellibs.<pkg>` and/or `torch` to new version
3. Validate — refuses if cohort `source_versions` drift (forces explicit `version-mismatch:<pkg>` allow)

### 5.4 Smoke regression test (no specific launch planned)

Use the existing `test-sweep-changes` SKILL.md "canary fallback" gate. The spec system is for launches.

## 6. Concrete examples

### Example 1: M2 launch spec (the one that, IF IT HAD EXISTED at 17:05 ET today, would have prevented today's failure)

```json
{
  "spec_version": 1,
  "name": "ngb-verify-2026-05-07",
  "description": "NGB verify on the regenerated 190-model cohort + right stack (NGB=nested_graph_breaks)",
  "venv": "~/envs/torch-nightly-cu126/bin/python",
  "torch": "2.13.0.dev20260502+cu126",
  "modellibs": {"transformers": "5.6.2", "diffusers": "0.38.0"},
  "cohort": {
    "derive_from": "sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json",
    "filter": "status == ok",
    "source_sha256": "<sha256 captured at spec-creation time>"
  },
  "modes": ["eval", "train"],
  "workers": 2,
  "timeout_s": 180,
  "passes": ["identify"],
  "compile_kwargs": {"fullgraph": false},
  "dynamo_config": {"nested_graph_breaks": true},
  "wake_cron_eta_min": 30
}
```

### Example 2: Gate stage emission (for example 1's spec)

```bash
#!/usr/bin/env bash
set -euo pipefail

VENV=/home/pengwu/envs/torch-nightly-cu126/bin/python
SOURCE=sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json
OUT=sweep_results/experiments/ngb-verify-2026-05-07-gate-2026-05-07-180000

cd /home/pengwu/projects/oss-model-graph-break-corpus
[ -x "$VENV" ] || { echo "venv missing: $VENV"; exit 2; }
[ -f "$SOURCE" ] || { echo "source missing: $SOURCE"; exit 2; }

# 1. Sub-sample 5 models (deterministic, preserves _metadata)
"$VENV" tools/sample_cohort.py "$SOURCE" \
  --filter 'status == ok' \
  --n 5 --seed 42 \
  --output /tmp/ngb-verify-2026-05-07-gate.json --force

mkdir -p "$OUT"

# 2. Launch identify sweep (gate: foreground, no watchdog)
"$VENV" tools/run_experiment.py sweep \
  --models /tmp/ngb-verify-2026-05-07-gate.json \
  --transformers 5.6.2 --diffusers 0.38.0 \
  --strict-modellibs \
  --modes eval train --workers 2 --timeout 180 --device cuda \
  --identify-only \
  --compile-kwargs '{"fullgraph": false}' \
  --dynamo-config nested_graph_breaks=true \
  --output-dir "$OUT" --run-name ngb-verify-2026-05-07-gate

# 3. Post-sweep check
"$VENV" tools/check_cohort_invariants.py --post-sweep "$OUT/identify_results.json"
```

### Example 3: Full stage emission (for example 1's spec)

```bash
#!/usr/bin/env bash
set -euo pipefail

VENV=/home/pengwu/envs/torch-nightly-cu126/bin/python
SOURCE=sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json
OUT=sweep_results/experiments/ngb-verify-2026-05-07-full-2026-05-07-180500

cd /home/pengwu/projects/oss-model-graph-break-corpus
[ -x "$VENV" ] || { echo "venv missing: $VENV"; exit 2; }
[ -f "$SOURCE" ] || { echo "source missing: $SOURCE"; exit 2; }

mkdir -p "$OUT"

# Form 1: derivation → use --models-from + --filter (canonical)
nohup "$VENV" tools/run_experiment.py sweep \
  --models-from "$SOURCE" --filter 'status == ok' \
  --transformers 5.6.2 --diffusers 0.38.0 \
  --strict-modellibs \
  --modes eval train --workers 2 --timeout 180 --device cuda \
  --identify-only \
  --compile-kwargs '{"fullgraph": false}' \
  --dynamo-config nested_graph_breaks=true \
  --output-dir "$OUT" --run-name ngb-verify-2026-05-07-full \
  > "$OUT/launch.log" 2>&1 &
echo "PID=$!" > "$OUT/launch.pid"
echo "Launched. PID=$(cat $OUT/launch.pid). Monitor: tail -f $OUT/launch.log"

# Watchdog cron (mandatory for full stage)
( crontab -l ; echo "*/10 * * * * $VENV ~/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog.py $PWD/$OUT/ --interval-min 10 --post-to spaces/AAQANraxXE4 >> /tmp/sweep-watchdog.log 2>&1" ) | crontab -

# Wake-cron at +30 min for invariant check (per spec.wake_cron_eta_min).
# Emitted as a LITERAL myclaw-cron registration command — copy-paste-and-run, no English prose.
# (Per rev-3 adversary feedback: English advisory re-opens the wake-cron-misfire failure mode
# because the human re-types the prompt and may drop a flag.)
WAKE_AT_EPOCH=$(($(date +%s) + 30*60))
echo "Register wake-cron via:"
echo "  meta cron one-shot --at $WAKE_AT_EPOCH --prompt \"Run: $VENV tools/check_cohort_invariants.py --post-sweep $OUT/identify_results.json. Reply: PASS or FAIL: <details>. Post to spaces/AAQANraxXE4 via --as-bot.\""
```

### Example 4: derive_spec_from_prior.py round-trip

```bash
$ tools/derive_spec_from_prior.py sweep_results/experiments/ngb-verify-2026-05-06/ \
    --venv ~/envs/torch-nightly-cu126/bin/python \
    --output specs/ngb-verify-repro-2026-05-07.json

Reading sweep_state.json → args field
Reading versions block → torch=2.13.0.dev20260502+cu126
Re-using tools/run_experiment.py argparse for token parsing
Generating spec with all flags from prior run + provided --venv
Validating generated spec...

Wrote specs/ngb-verify-repro-2026-05-07.json:
  modellibs.transformers = 5.6.2
  modellibs.diffusers = 0.38.0
  cohort.path = experiments/configs/nested_gb_cohort_2026-05-06.json
  compile_kwargs.fullgraph = false
  dynamo_config.nested_graph_breaks = true
  ...

Diff against prior run's args (sanity check):
  ✓ all flags from prior recovered into spec form
  ✓ no spec fields invented (every value sourced from sweep_state.json)
```

## 7. Reference workflows for the script

```bash
# Validate without emitting
tools/derive_sweep_commands.py specs/<name>.json --validate

# Print the gate command
tools/derive_sweep_commands.py specs/<name>.json --stage gate --emit

# Print the sample command
tools/derive_sweep_commands.py specs/<name>.json --stage sample --emit

# Print the full canonical command
tools/derive_sweep_commands.py specs/<name>.json --stage full --emit

# Show the parity report (table comparing spec / cohort metadata / planned launch)
tools/derive_sweep_commands.py specs/<name>.json --report

# Generate a spec from a prior run's recorded args
tools/derive_spec_from_prior.py sweep_results/experiments/<prior-name>/ \
    --venv <venv-path> --output specs/<new-name>.json

# Generate the canonical wake-cron line for a stage
tools/derive_wake_cron.py specs/<name>.json --stage full
```

## 8. CLI flag coverage (v1 vs v2)

Maps every flag from `tools/run_experiment.py sweep` argparse → spec field. **A flag missing from the spec re-introduces drift on that axis.**

| CLI flag | v1 spec field | Notes |
|---|---|---|
| `--models <path>` | `cohort.path` (Form 2) | Validator pins via `expected_sha256` |
| `--models-from <path>` | `cohort.derive_from` (Form 1) | + `cohort.filter` |
| `--filter <expr>` | `cohort.filter` (Form 1) | |
| `--transformers <ver>` | `modellibs.transformers` | |
| `--diffusers <ver>` | `modellibs.diffusers` | |
| `--timm <ver>` | `modellibs.timm` | |
| `--torch <spec>` | (NOT in v1; venv path is the source of truth; v2 may add for build-the-venv workflows) | |
| `--cuda-variant {cu128,cu126}` | `cuda_variant` | |
| `--strict-modellibs` | `strict_modellibs` (default `true`) | Always emitted unless `non-strict-modellibs` in `allow_flags` |
| `--modes <eval/train>` | `modes` | |
| `--workers <N>` | `workers` | |
| `--timeout <S>` | `timeout_s` | |
| `--device {cpu,cuda}` | `device` | |
| `--identify-only` | `passes: ["identify"]` | Inferred: emitted if `"explain" not in passes` |
| `--compile-kwargs <json>` | `compile_kwargs` | Required in spec |
| `--dynamo-config <key=val>` | `dynamo_config` | Required in spec; emitted as repeated flag |
| `--inductor-config <key=val>` | `inductor_config` | Repeated flag |
| `--dynamic-dim {batch,all}` | `dynamic_dim` | |
| `--limit <N>` | `limit` | |
| `--stability {stable,unstable}` | `stability` | |
| `--setup-script <path>` | `setup_script` | |
| `--allow-version-mismatch` | `allow_flags: ["version-mismatch"]` | |
| `--allow-bare-cohort` | `allow_flags: ["bare-list"]` | **CLI flag stays `--allow-bare-cohort`** — only spec vocabulary uses `bare-list` (matches `cohort_validator.py` `BARE_LIST_REJECTED` error code). No breaking change to argparse; `derive_spec_from_prior` (when added in v1.1) can re-parse old `sweep_state.json` files. |
| `--allow-empty-versions` | `allow_flags: ["empty-versions"]` | |
| `--allow-partial-versions` | `allow_flags: ["partial-versions"]` | |
| `--allow-stale-cohort` | `allow_flags: ["stale-cohort"]` | |
| `--no-auto-retry` | `auto_retry: false` | |
| `--source <list>` | (NOT in v1; nightly canonical sweep uses this — see §9 nightly carve-out. v1.1 may add Form 4 (`cohort.sources`). For now, nightly retains direct-CLI invocation.) | |
| `--save-cohort <path>` | (NOT in v1; output side-effect; v1.1 may add for repro inspection) | |
| `--output-dir <path>` | derived from `output_dir_template` | |
| `--run-name <slug>` | derived from `run_name_template` | |
| `--resume` | NOT in spec (per-run operational; spec is launch-time) | |

v1.1+ deferrals: `--torch <spec>` (build-the-venv workflows), `--source <list>` (source-enumeration mode — nightly retains direct-CLI), `--save-cohort <path>` (output inspection).

## 9. Spec ↔ skill migration (Phase 4 — load-bearing)

The `skills/sweep.md` and `skills/test-sweep-changes/SKILL.md` skills are partially superseded by the spec system. Without explicit migration, future-Otter will read the skills, ignore the spec system, and repeat today's failures.

**Required Phase 4 changes:**

1. **Top of `skills/sweep.md`** — add a header above § 1:
   > **STOP.** If you are about to launch a curated sweep (verify, correctness, explain follow-up, hand-built cohort, or any sweep where the cohort is a specific list of models): you should be invoking `tools/derive_sweep_commands.py` against a spec at `experiments/specs/`. The skill below is the historical workflow plus operational details (watchdog, post-sweep diligence). DO NOT hand-type launch flags for curated sweeps.
   >
   > **Carve-out — nightly:** the canonical nightly sweep (`tools/run_experiment.py nightly`) uses `--source hf diffusers custom` for source enumeration, which is NOT yet covered by the v1 spec system. Nightly retains direct-CLI invocation until v1.1 adds Form 4 (`cohort.sources`).

2. **Top of `skills/test-sweep-changes/SKILL.md`** — replace the current "5 gates" framing with:
   > **STOP.** If you are about to launch a sweep: use `tools/derive_sweep_commands.py specs/<name>.json --stage gate` instead of the canary 5-gate. The canary fallback below is for harness regression after code edits with NO specific launch planned.

3. **Replace `skills/test-sweep-changes/SKILL.md` Gate 5 body** with: "Use `derive_sweep_commands.py --stage gate --emit` to get the launch-customized gate command. Run it. The body is mechanically derived from the spec — no manual flag construction."

4. **`skills/sweep.md` § 4 sample-sweep gate body** — replace with: "Use `derive_sweep_commands.py --stage sample --emit`. Run it. APPLY-A from `sweep_sanity_check.md` runs via `tools/check_cohort_invariants.py` per spec.post_sweep_check."

5. **Both skills' revision logs** — entries dated to the spec-system commit referencing this design doc.

6. **Mechanical enforcement** (Phase 0 test): `tools/test_skill_headers.py` greps both skill files for the verbatim STOP header strings AND the nightly carve-out paragraph. Both must be present. Test fails CI if either is absent or modified. Per rev-3 adversary new gap #5: explicit grep scope is "the entire STOP block including the nightly carve-out paragraph." Editorial changes to either trigger a CI failure that requires explicit re-acknowledgment.

Without these changes (especially #6), the spec system lands but is silently bypassed.

## 10. What's NOT in v1 (intentional non-goals — "start simple" per Peng)

**Deferred to v1.1:**
- **`derive_wake_cron.py` as a separate tool** — v1 emits the LITERAL myclaw-cron registration command inline (per rev-3 adversary feedback; no separate tool needed). v1.1 may extract this into a helper if it grows.

(`derive_spec_from_prior.py` was deferred in rev 3 but moved BACK INTO v1 in rev 4 per Peng — the NGB verify re-sweep needs it, and deferring re-opens the 2026-05-07 afternoon failure mode that motivated this entire effort.)
- **`--refresh-content-hashes` helper** — v1 spec author runs `sha256sum <file>` manually; v1.1 adds the helper to reduce friction.
- **`cohort.cache`** (Form 1 optional sub-field) — was in rev 2; dropped from v1 because it's documentation-only and creates a stale-cache failure mode.
- **Form 4 (`cohort.sources`)** for nightly — nightly retains direct-CLI per §9 carve-out.
- **`--save-cohort` field** — output side-effect; v1.1 if needed.

**Deferred to v2:**
- **Spec inheritance / templates** — v1 specs are flat.
- **Per-stage flag overrides beyond §3.3** — intentionally rigid.
- **Auto-execution** — script EMITS commands; humans run them. No `--execute` flag.
- **Watchdog auto-install** — v1 emits the bash; humans run.
- **Form 3 (inline cohort)** — rare; v2.
- **`--torch <spec>`** — build-the-venv workflows.

**Out of scope entirely:**
- **Result-file deeper analysis** (analyze_sweep, file_issues) — separate concern from launch.

## 11. Implementation phases (simplified for v1 per Peng's "start simple")

**Strict ordering — no concurrent phases.** Each phase ends with adversary review of THAT phase's diffs.

**Phase 0:** Write the test contract. `tools/test_sweep_spec.py` containing all implementation tests (16 from rev 1 review + 20 from rev 2 review = 36 tests). Each test has SETUP / ACTION / EXPECTED / DETECTS. Use mock cohort + mock venv via monkeypatch where possible. Tests fail until Phase 1 lands. PLUS `tools/test_skill_headers.py` for §9 mechanical enforcement.

**Phase 1:** Spec schema + parser + validator (`tools/sweep_spec.py`). All Phase 0 tests pass after this phase.

**Phase 1.5:** Upgrade `tools/sample_cohort.py` to preserve _metadata + add sub-sample provenance (per §3.4). Add tests. **STRICT prereq of Phase 2** (rev 2 had this as "concurrent" — adversary flagged that "concurrent ≠ before"; the §6 emissions assume preservation has landed).

**Phase 2:** `tools/derive_sweep_commands.py` — uses Phase 1 module, emits bash for each stage. Includes `--validate`, `--report`, `--stage X --emit` modes. Adds tests.

**Phase 3:** `tools/derive_spec_from_prior.py` — re-uses `tools/run_experiment.py` argparse (DRY), reads `sweep_state.json`'s `args` field, emits a v1 spec. REQUIRES explicit `--venv <path>` (sweep_state.json doesn't record resolved venv path). Adds tests including round-trip fidelity (re-emit-from-derived-spec → diff against original args). **In v1 per Peng directive 2026-05-07 18:26: directly supports the NGB verify re-sweep use case and closes the 2026-05-07 afternoon memory-vs-record failure mode.**

**Phase 4:** Skill updates (per §9). Update local CLAUDE.md trigger entry. Run `tools/test_skill_headers.py` to verify mechanical enforcement passes.

**Phase 5:** Use `derive_spec_from_prior.py` against the abandoned 2026-05-06 NGB verify run to produce the v1 spec for the new NGB verify launch. This validates the system end-to-end and IS the spec we use for the next launch.

**Deferred to v1.1 (separate workstream):** `derive_wake_cron.py` (extracted from inline emission), `--refresh-content-hashes`, Form 4 (sources for nightly), `cohort.cache`. Each gets its own design + adversary review when scheduled.

**Estimated v1 effort:** ~5 hours total (rev 3 was ~4h before adding `derive_spec_from_prior.py` back).

---

## Revision log

| Rev | Date | Changes |
|---|---|---|
| 1 | 2026-05-07 | Initial draft — captures Peng's proposal + my proposed extensions. |
| 2 | 2026-05-07 | Addresses adversary review of rev 1 (14 gaps + 16 tests). Major changes: (a) `cohort` field replaced by block with three forms (Form 1 derivation preferred, per Peng question); (b) added required `spec_version`, `torch` fields; (c) made `compile_kwargs` and `dynamo_config` REQUIRED (no defaulting); (d) version-coherence check made asymmetric (cohort-declares-but-spec-doesn't = REJECTED); (e) cohort content pinning via sha256 (Form 1: `source_sha256`; Form 2: `expected_sha256`); (f) `strict_modellibs` default ON; (g) added optional fields covering ALL relevant CLI flags (limit, dynamic_dim, stability, setup_script, cuda_variant, etc.) per §8 coverage table; (h) `output_dir_template` default includes HHMMSS suffix; (i) sub-sample preservation via §3.4 (sample_cohort.py upgrade); (j) emitted bash includes `set -euo pipefail` + path safety net per §4.6; (k) `post_sweep_check:` field added (per adversary opinion on open Q3); (l) Phase 4 skill migration spelled out concretely (per gap #9); (m) Phase 0 (test contract) added before Phase 1; (n) `--allow-bare-cohort` renamed to `bare-list` to match validator code; (o) Form 3 (inline) deferred to v2. |
| 3 | 2026-05-07 | Addresses adversary review of rev 2 (3 HIGH new gaps + 7 MED + 3 LOW + 20 additional tests) + Peng's "start simple" directive. Autonomous changes: (a) replaced parameterized `version-mismatch:<pkg>` with separate `version_mismatch_allow` field — cleaner; (b) `sub_sample_seed` default changed from `42` to `null` (cohort-anchored via `_seed_from_cohort`) — fixes regression; (c) clarified CLI flag stays `--allow-bare-cohort`; only spec vocabulary uses `bare-list`; (d) `post_sweep_check` made passes-aware (defaults to `identify` or `explain` results based on `passes`); (e) Phase 6 (sample_cohort.py upgrade) renamed Phase 1.5 + made strict prereq of Phase 2; (f) added `test_skill_headers.py` to Phase 0 for §9 mechanical enforcement; (g) dropped `cohort.cache` from v1 (defer to v1.1); (h) added nightly carve-out to §9 STOP header; (i) v1 simplified — `derive_spec_from_prior.py`, `derive_wake_cron.py`, `--refresh-content-hashes`, Form 4 all deferred to v1.1 separate workstream. Implementation effort estimate dropped from 6-8 hours to ~4 hours. Pending Peng approval before implementation. |
| 4 | 2026-05-07 | Addresses adversary review of rev 3 (2 RECONSIDER deferrals + 6 new gaps + 8 additional tests). Autonomous changes: (a) sha256 placeholder validator behavior: detects literal `<...>` pattern + emits help with `sha256sum <path>` command (closes new gap #1); (b) emitted bash adds `[ -d "$OUT" ] && exit 3` to safety preamble (closes new gap #2 — same-second collision); (c) §3.3 cleaned up — `--allow-bare-cohort` NOT emitted by spec system after Phase 1.5 (sub-samples are canonical) (closes new gaps #3, #4); (d) §3.4 example renamed `sub_sample_seed` field to `sub_sample_seed_resolved` for clarity (closes new gap #4); (e) §6 Example 3 wake-cron advisory becomes literal myclaw-cron registration command (no English prose) — closes wake-cron RECONSIDER without needing separate `derive_wake_cron.py` tool; (f) §9 #6 `test_skill_headers.py` scope explicitly defined as "STOP block + nightly carve-out paragraph verbatim"; (g) **`derive_spec_from_prior.py` moved BACK into v1** per Peng directive: directly supports the NGB verify re-sweep + closes the 2026-05-07 afternoon memory-vs-record failure mode that motivated this entire effort. Phase added: Phase 3 builds the tool, Phase 5 uses it to derive the actual NGB verify spec from the abandoned 2026-05-06 run. v1 effort estimate: ~4h → ~5h. |
