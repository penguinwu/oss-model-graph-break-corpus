# Sweep Experiment Spec — Design Doc

**Revision:** 2
**Owner:** otter (with Peng-in-the-loop)
**Created:** 2026-05-07
**Status:** draft — pending adversary re-review and Peng approval before implementation

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
  "source_sha256": "7f3a...",
  "cache": "experiments/configs/<name>.json"
}
```

| Sub-field | Required | Description |
|---|---|---|
| `derive_from` | yes | Path to a prior run's results file (typically `identify_results.json` or `explain_results.json`) |
| `filter` | yes | Filter expression (see §3.1.6) |
| `source_sha256` | yes | SHA256 of the source file at spec-creation time. Validator REFUSES if source's current sha256 doesn't match. Forces awareness when prior run is regenerated. |
| `cache` | optional | If set, the resolved cohort is written here for inspection / reproducibility. The CLI emission still uses `--models-from + --filter` (regenerates inline), so the cache is documentation-only. |

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
| `allow_flags` | array<string> | `[]` | Subset of `["bare-list", "empty-versions", "partial-versions", "stale-cohort", "version-mismatch", "non-strict-modellibs"]`. **Note:** `bare-list` is auto-added for gate/sample only if sub-sample preservation is disabled (see §3.4). For Form 3 (inline), MUST be set explicitly. |
| `output_dir_template` | string | `"sweep_results/experiments/{name}-{stage}-{date}-{time}"` | Format vars: `{name}`, `{stage}`, `{date}` (YYYY-MM-DD), `{time}` (HHMMSS). HHMMSS prevents same-day collision. |
| `run_name_template` | string | `"{name}-{stage}"` | |
| `gate_size` | int | `5` | Models in gate stage |
| `sample_size` | int | `20` | Models in sample stage |
| `sub_sample_seed` | int | `42` | Deterministic seed for sub-sample selection |
| `watchdog` | object | `{"enabled": true, "interval_min": 10, "post_to": "spaces/AAQANraxXE4"}` | Watchdog cron config |
| `wake_cron_eta_min` | int\|null | `null` | If set, schedule wake at +eta minutes |
| `post_sweep_check` | string | `"tools/check_cohort_invariants.py --post-sweep {output_dir}/identify_results.json"` | Canonical post-sweep check command. Wake-cron uses this. Format vars: `{output_dir}`, `{venv}`, `{name}`, `{stage}`. |
| `nohup` | bool\|null | `null` | `null` = auto (foreground for gate/sample, nohup for full); explicit override applies to all stages |

### 3.3 Stage-specific derivations

Three stages: `gate`, `sample`, `full`. Each derives a launch command from the same spec, with these stage-specific transformations:

| Aspect | Gate | Sample | Full |
|---|---|---|---|
| Sub-sample size | `min(gate_size, limit)` | `min(sample_size, limit)` | `limit` if set, else full cohort |
| Sub-sample written to | `/tmp/<name>-gate.json` | `/tmp/<name>-sample.json` | N/A |
| `--models` / `--models-from` | Sub-sample → `--models` (with `--allow-bare-cohort` only if sub-sample preservation disabled) | Same as gate | If Form 1: `--models-from` + `--filter`; if Form 2: `--models <path>`; if Form 3: inline materialized to `/tmp/<name>-full.json` + `--allow-bare-cohort` |
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
  "sub_sample_seed": 42,
  "sub_sample_parent_sha256": "<parent cohort sha256 at sample time>"
}
```

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
- Cohort declares X, spec declares ≠X → REFUSED unless `"version-mismatch"` in allow_flags
- Cohort declares X, spec doesn't declare → REFUSED unless `"version-mismatch:<pkg>"` in allow_flags
- Cohort doesn't declare, spec declares → OK (cohort may be older format)

This closes adversary gap #3 — the AND-clause hole in rev 1.

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

(Closes adversary gap #11.)

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

# Wake-cron at +30 min for invariant check (per spec.wake_cron_eta_min)
echo "Schedule wake-cron via myclaw-cron with prompt:"
echo "  Run: $VENV tools/check_cohort_invariants.py --post-sweep $OUT/identify_results.json"
echo "  Reply: \"PASS\" or \"FAIL: <details>\""
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
| `--allow-bare-cohort` | `allow_flags: ["bare-list"]` | Renamed to align with `cohort_validator.py` `BARE_LIST_REJECTED` error code |
| `--allow-empty-versions` | `allow_flags: ["empty-versions"]` | |
| `--allow-partial-versions` | `allow_flags: ["partial-versions"]` | |
| `--allow-stale-cohort` | `allow_flags: ["stale-cohort"]` | |
| `--no-auto-retry` | `auto_retry: false` | |
| `--source <list>` | (NOT in v1; spec uses `cohort` block instead. v2 may add for source-enumeration workflows.) | |
| `--save-cohort <path>` | (NOT in v1; output side-effect; spec's `cohort.cache` covers Form 1) | |
| `--output-dir <path>` | derived from `output_dir_template` | |
| `--run-name <slug>` | derived from `run_name_template` | |
| `--resume` | NOT in spec (per-run operational; spec is launch-time) | |

v2 deferrals: `--torch <spec>` (build-the-venv workflows), `--source <list>` (source-enumeration mode), `--save-cohort <path>` (covered by cohort.cache for Form 1).

## 9. Spec ↔ skill migration (Phase 4 — load-bearing)

The `skills/sweep.md` and `skills/test-sweep-changes/SKILL.md` skills are partially superseded by the spec system. Without explicit migration, future-Otter will read the skills, ignore the spec system, and repeat today's failures.

**Required Phase 4 changes:**

1. **Top of `skills/sweep.md`** — add a header above § 1:
   > **STOP.** If you are about to launch a sweep, you should be invoking `tools/derive_sweep_commands.py` against a spec at `experiments/specs/`. The skill below is the historical workflow plus operational details (watchdog, post-sweep diligence). DO NOT hand-type launch flags. If no spec exists for your experiment, write one (or `derive_spec_from_prior.py` from the closest reference run) FIRST.

2. **Top of `skills/test-sweep-changes/SKILL.md`** — replace the current "5 gates" framing with:
   > **STOP.** If you are about to launch a sweep: use `tools/derive_sweep_commands.py specs/<name>.json --stage gate` instead of the canary 5-gate. The canary fallback below is for harness regression after code edits with NO specific launch planned.

3. **Replace `skills/test-sweep-changes/SKILL.md` Gate 5 body** with: "Use `derive_sweep_commands.py --stage gate --emit` to get the launch-customized gate command. Run it. The body is mechanically derived from the spec — no manual flag construction."

4. **`skills/sweep.md` § 4 sample-sweep gate body** — replace with: "Use `derive_sweep_commands.py --stage sample --emit`. Run it. APPLY-A from `sweep_sanity_check.md` runs via `tools/check_cohort_invariants.py` per spec.post_sweep_check."

5. **Both skills' revision logs** — entries dated to the spec-system commit referencing this design doc.

Without these changes, the spec system lands but is silently bypassed.

## 10. What's NOT in v1 (intentional non-goals)

- **Spec inheritance / templates:** v1 specs are flat. v2 if needed.
- **Per-stage flag overrides beyond §3.3:** intentionally rigid.
- **Auto-execution:** the script EMITS commands; humans (or wake-crons) RUN them. No `--execute` flag in v1.
- **Watchdog auto-install / wake-cron auto-schedule:** v1 emits the bash for these as advisory output; humans run them. v2 may automate.
- **Spec for explain-only / correctness / non-NGB experiments:** v1 covers the same axes as today's NGB verify. Other sweep types may need additional fields; we'll add as we encounter them.
- **Result-file analysis flags:** the spec covers the launch and the post-sweep check (via `post_sweep_check`), not deeper analysis (analyze_sweep, file_issues).
- **`--torch <spec>`, `--source <list>`, `--save-cohort <path>`:** see §8 v2 deferrals.
- **Form 3 (inline cohort):** rare; deferred to v2.
- **`derive_wake_cron.py`:** the canonical wake-cron emission — v1 ships an advisory line; v2 emits the full myclaw-cron registration command.

## 11. Implementation phases

**Phase 0 (NEW):** Write the test contract. `tools/test_sweep_spec.py` containing all the implementation tests (the 16 from adversary review of rev 1, plus any new tests from rev 2's additions). Each test has SETUP / ACTION / EXPECTED / DETECTS. Use mock cohort + mock venv via monkeypatch where possible. Tests fail until Phase 1 lands.

**Phase 1:** Spec schema + parser + validator (`tools/sweep_spec.py`). All Phase 0 tests pass after this phase.

**Phase 2:** `tools/derive_sweep_commands.py` (uses Phase 1 module, emits bash for each stage). Adds tests.

**Phase 3:** `tools/derive_spec_from_prior.py`. Adds tests.

**Phase 4:** Skill updates (per §9). Update local CLAUDE.md trigger entry.

**Phase 5:** Migrate existing/recent specs to the new format (a spec for the abandoned 2026-05-06 NGB verify, the M2 plan, the explain pass — for reference and tests).

**Phase 6 (concurrent with Phase 1+):** Upgrade `tools/sample_cohort.py` to preserve _metadata + add sub-sample provenance (per §3.4). Add tests. This is a small change but unblocks Phase 2's emission of clean sub-sample bash.

Each phase ends with adversary review of THAT phase's diffs.

---

## Revision log

| Rev | Date | Changes |
|---|---|---|
| 1 | 2026-05-07 | Initial draft — captures Peng's proposal + my proposed extensions. |
| 2 | 2026-05-07 | Addresses adversary review of rev 1 (14 gaps + 16 tests). Major changes: (a) `cohort` field replaced by block with three forms (Form 1 derivation preferred, per Peng question); (b) added required `spec_version`, `torch` fields; (c) made `compile_kwargs` and `dynamo_config` REQUIRED (no defaulting); (d) version-coherence check made asymmetric (cohort-declares-but-spec-doesn't = REJECTED); (e) cohort content pinning via sha256 (Form 1: `source_sha256`; Form 2: `expected_sha256`); (f) `strict_modellibs` default ON; (g) added optional fields covering ALL relevant CLI flags (limit, dynamic_dim, stability, setup_script, cuda_variant, etc.) per §8 coverage table; (h) `output_dir_template` default includes HHMMSS suffix; (i) sub-sample preservation via §3.4 (sample_cohort.py upgrade); (j) emitted bash includes `set -euo pipefail` + path safety net per §4.6; (k) `post_sweep_check:` field added (per adversary opinion on open Q3); (l) Phase 4 skill migration spelled out concretely (per gap #9); (m) Phase 0 (test contract) added before Phase 1; (n) `--allow-bare-cohort` renamed to `bare-list` to match validator code; (o) Form 3 (inline) deferred to v2. |
