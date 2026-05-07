# Sweep-Spec Test Contract

**Phase 0 deliverable.** Companion to `design/sweep-spec-design.md` rev 4.

This is the explicit, enumerated test inventory for the sweep-spec system. Every test has SETUP / ACTION / EXPECTED / DETECTS. Each is tagged with v1 or v1.1 (which phase it belongs to).

**Phase 0 = make every v1 test in this file PASS as a failing-stub-with-assertion in `tools/test_sweep_spec.py` (and its peers). Phase 1+ = make them turn green.**

| Status legend | Meaning |
|---|---|
| 🟡 stub | Test written, asserts the implementation exists; failing because Phase 1+ hasn't landed |
| 🟢 passing | Test passing |
| 🔴 failing | Test failing for a reason other than missing implementation (regression) |
| ⏸️ v1.1 | Test reserved; implementation deferred |

## Test inventory

### Group A — Spec self-consistency (§4.1)

**A1: spec_version unknown** [v1] [🟡 stub]
- SETUP: spec with `spec_version: 99`
- ACTION: `derive_sweep_commands.py specs/x.json --validate`
- EXPECTED: exit non-zero; error names `spec_version`
- DETECTS: validator silently accepts unknown spec versions

**A2: name field != filename** [v1] [🟡 stub]
- SETUP: file `experiments/specs/foo.json` with `"name": "bar"`
- ACTION: `--validate`
- EXPECTED: exit non-zero naming both filename and field value
- DETECTS: drift between filename and `name`

**A3: required field missing (`compile_kwargs`)** [v1] [🟡 stub]
- SETUP: spec without `compile_kwargs` field
- ACTION: `--validate`
- EXPECTED: exit non-zero naming the missing required field; NOT silently defaulted
- DETECTS: selective trust / "fill in defaults" anti-pattern

**A4: required field missing (`dynamo_config`)** [v1] [🟡 stub]
- Same as A3 for `dynamo_config`

**A5: invalid mode in spec** [v1] [🟡 stub]
- SETUP: spec with `modes: ["eval", "explain"]`
- ACTION: `--validate`
- EXPECTED: exit non-zero naming invalid mode
- DETECTS: enum violation; tests strict subset, not just presence

**A6: workers out of range** [v1] [🟡 stub]
- SETUP: spec with `workers: 99`
- ACTION: `--validate`
- EXPECTED: exit non-zero with valid range
- DETECTS: lenient bounds

**A7: passes order wrong** [v1] [🟡 stub]
- SETUP: spec with `passes: ["explain", "identify"]`
- ACTION: `--validate`
- EXPECTED: exit non-zero with canonical order
- DETECTS: order-independence bug

**A8: limit < gate_size** [v1] [🟡 stub]
- SETUP: spec with `limit: 3` and `gate_size: 5`
- ACTION: `--validate`
- EXPECTED: exit non-zero with explicit message
- DETECTS: stage-internal-consistency hole around `--limit`

### Group B — Modellibs (§3.1, §4.1, §4.2)

**B1: modellibs declared package not installed** [v1] [🟡 stub]
- SETUP: spec with `modellibs: {transformers: "9.9.9-not-installed"}`
- ACTION: `--validate`
- EXPECTED: exit non-zero naming missing `~/envs/modellibs/<pkg>-<ver>/` path
- DETECTS: late-binding install failures

**B2: cohort source uses transformers but spec.modellibs missing transformers** [v1] [🟡 stub]
- SETUP: cohort source has `metadata.versions.transformers`; spec has empty `modellibs: {}`
- ACTION: `--validate`
- EXPECTED: exit non-zero with "spec.modellibs missing transformers required by cohort source"
- DETECTS: silent fall-through to venv-bundled copy (the 2026-05-06 wrong-stack failure mode on a new axis)

**B3: spec torch != venv actual torch** [v1] [🟡 stub]
- SETUP: spec with `venv: ~/envs/torch211/bin/python` (torch 2.11) but `torch: "2.13.0.dev..."`
- ACTION: `--validate`
- EXPECTED: exit non-zero; error includes both versions and which side is spec/venv
- DETECTS: spec-vs-venv torch drift

### Group C — Cohort block (§3.1.5)

**C1: cohort block has both Form 1 + Form 2 keys** [v1] [🟡 stub]
- SETUP: spec with both `cohort.derive_from` AND `cohort.path` set
- ACTION: `--validate`
- EXPECTED: exit non-zero with "cohort block must have EXACTLY ONE of derive_from/path/inline"
- DETECTS: form ambiguity

**C2: bare list cohort without `bare-list` allow** [v1] [🟡 stub]
- SETUP: cohort Form 2 pointing at a flat-list .json
- ACTION: `--validate`
- EXPECTED: exit non-zero with code matching `cohort_validator.BARE_LIST_REJECTED`
- DETECTS: spec-vs-runtime validator alignment

**C3: bare list cohort with `bare-list` allow** [v1] [🟡 stub]
- SETUP: same as C2 but spec has `allow_flags: ["bare-list"]`
- ACTION: `--validate`
- EXPECTED: pass; emission for full stage includes `--allow-bare-cohort`
- DETECTS: override mechanism end-to-end

**C4: Form 3 (inline) not in v1 → REJECTED** [v1.1] [⏸️ v1.1]
- SETUP: spec with `cohort.inline: [{name: "X"}]`
- ACTION: `--validate`
- EXPECTED: exit non-zero "Form 3 (inline) deferred to v1.1"
- DETECTS: premature Form 3 acceptance

**C5: source_sha256 placeholder detected** [v1] [🟡 stub]
- SETUP: spec with `cohort.source_sha256: "<sha256 captured at spec-creation time>"` (literal placeholder)
- ACTION: `--validate`
- EXPECTED: exit non-zero with help message containing literal `sha256sum <derive_from-path>` command
- DETECTS: silent failure where user pastes placeholder; per rev-3 adversary new gap #1

**C6: source_sha256 drift detected** [v1] [🟡 stub]
- SETUP: spec with `cohort.derive_from + source_sha256: <hash X>`; modify source file in place
- ACTION: `--validate`
- EXPECTED: exit non-zero with "source sha256 drift" error
- DETECTS: §4.4 enforcement

**C7: filter expression invalid** [v1] [🟡 stub]
- SETUP: spec with `cohort.filter: "name == X"` (unsupported predicate)
- ACTION: `--validate`
- EXPECTED: exit non-zero naming the bad filter
- DETECTS: filter grammar enforcement

### Group D — Version coherence (§4.3)

**D1: cohort declares X, spec declares ≠X, no allow** [v1] [🟡 stub]
- SETUP: cohort source declares `transformers=5.6.2`; spec declares `5.5.3`
- ACTION: `--validate`
- EXPECTED: exit non-zero VERSION_MISMATCH
- DETECTS: §4.3 case 2

**D2: cohort declares X, spec doesn't declare** [v1] [🟡 stub]
- SETUP: cohort declares `diffusers=0.38.0`; spec has no `diffusers` in modellibs
- ACTION: `--validate`
- EXPECTED: exit non-zero with "cohort declares diffusers but spec doesn't"
- DETECTS: rev-1 gap #3 closure (the AND-clause hole)

**D3: cohort declares + narrow allow** [v1] [🟡 stub]
- SETUP: D2 setup but spec has `version_mismatch_allow: ["diffusers"]`
- ACTION: `--validate`
- EXPECTED: pass
- DETECTS: narrow allow per-package mechanism

**D4: cohort declares + broad allow** [v1] [🟡 stub]
- SETUP: D2 setup but spec has `allow_flags: ["version-mismatch"]`
- ACTION: `--validate`
- EXPECTED: pass (broad covers all)
- DETECTS: broad allow disjunction with narrow

**D5: neither narrow nor broad allow** [v1] [🟡 stub]
- SETUP: D2 setup with empty allow flags
- ACTION: `--validate`
- EXPECTED: exit non-zero
- DETECTS: regression of D2 if allows are bug-implemented as conjunction

### Group E — Stage derivation (§3.3)

**E1: gate/sample/full emissions byte-identical except for documented deltas** [v1] [🟡 stub]
- SETUP: clean spec
- ACTION: emit gate, sample, full into three files; `diff` them
- EXPECTED: only deltas: `--models-from`/`--models` path, `--output-dir`, `--run-name`. Every other token bit-identical, in same order.
- DETECTS: stage drift (§3.3 load-bearing invariant)

**E2: NGB flags propagate to all stages** [v1] [🟡 stub]
- SETUP: §6 Example 1 spec verbatim
- ACTION: emit all three stages
- EXPECTED: each emission contains `--compile-kwargs '{"fullgraph": false}'` AND `--dynamo-config nested_graph_breaks=true`
- DETECTS: 2026-05-07 afternoon failure mode (missing flags) — the load-bearing test

**E3: dynamo_config repeated flag emission** [v1] [🟡 stub]
- SETUP: spec with `dynamo_config: {nested_graph_breaks: true, capture_scalar_outputs: true}`
- ACTION: capture emitted bash; grep `--dynamo-config` count
- EXPECTED: two `--dynamo-config key=value` flags (not single JSON blob)
- DETECTS: emission templating bug; argparse uses `action="append"`

**E4: compile_kwargs JSON quoting safe** [v1] [🟡 stub]
- SETUP: spec with `compile_kwargs: {"backend": "inductor", "options": {"key": "val with spaces"}}`
- ACTION: capture emitted bash; literally execute the `--compile-kwargs` token; assert downstream parses as same dict
- EXPECTED: round-trip identity
- DETECTS: shell quoting bugs in JSON emission

**E5: --identify-only emitted iff passes == ["identify"]** [v1] [🟡 stub]
- SETUP: spec A with `passes: ["identify"]`; spec B with `passes: ["identify", "explain"]`
- ACTION: emit full; check for `--identify-only` flag
- EXPECTED: A has it, B doesn't
- DETECTS: passes-aware emission

### Group F — Sub-sample (§3.4)

**F1: sub-sample preserves _metadata** [v1] [🟡 stub]
- SETUP: spec with cohort Form 2 (canonical); `gate_size: 5`
- ACTION: emit + run gate stage's `sample_cohort.py` invocation; load resulting sub-sample JSON
- EXPECTED: sub-sample contains `_metadata` block with `source_versions`, `sub_sampled_from`, `sub_sample_size`, `sub_sample_seed_resolved`, `sub_sample_parent_sha256`
- DETECTS: Phase 1.5 didn't land OR `--allow-bare-cohort` regression

**F2: no `--allow-bare-cohort` in sub-sample emission** [v1] [🟡 stub]
- SETUP: same spec as F1, post-Phase-1.5
- ACTION: capture emitted bash; grep for `--allow-bare-cohort`
- EXPECTED: flag absent (sub-sample is canonical, validator handles)
- DETECTS: emission template forgot to remove the bare-cohort allow

**F3: sub_sample_seed determinism with default null** [v1] [🟡 stub]
- SETUP: spec with `sub_sample_seed: null`; emit gate stage twice (separate processes)
- ACTION: compare `/tmp/<name>-gate.json` byte-by-byte
- EXPECTED: identical files
- DETECTS: regression where `null` accidentally passes nothing to `--seed`

**F4: sub_sample_seed override works** [v1] [🟡 stub]
- SETUP: spec with `sub_sample_seed: 42`; emit twice
- ACTION: compare files
- EXPECTED: identical (seed is determined by spec value)
- DETECTS: override mechanism

### Group G — Path safety in emitted bash (§4.6)

**G1: bash starts with set -euo pipefail** [v1] [🟡 stub]
- SETUP: any valid spec
- ACTION: capture emitted bash; check first non-shebang line
- EXPECTED: starts with `set -euo pipefail`
- DETECTS: emission template regression

**G2: bash exits 2 on missing venv at runtime** [v1] [🟡 stub]
- SETUP: spec with non-existent venv path; capture emitted bash
- ACTION: `bash -n` (syntax) then `bash` (runtime)
- EXPECTED: `bash -n` passes; `bash` exits 2 with "venv missing" error
- DETECTS: runtime path checks fire

**G3: bash exits 3 on output dir collision** [v1] [🟡 stub]
- SETUP: spec; emit gate; create the output dir; run emitted bash
- ACTION: bash
- EXPECTED: exits 3 with "output dir exists" before any model loads
- DETECTS: same-second collision protection (rev-3 adversary new gap #2)

### Group H — derive_spec_from_prior.py (§5.2)

**H1: round-trip fidelity** [v1] [🟡 stub]
- SETUP: real `sweep_state.json` from `ngb-verify-2026-05-06-2026-05-06`
- ACTION: derive_spec → re-emit full stage → diff against original `args` field
- EXPECTED: every flag in original args present in re-emission; `--compile-kwargs '{"fullgraph": false}'` AND `--dynamo-config nested_graph_breaks=true` both present (these are the two flags missed in 2026-05-07 afternoon failure)
- DETECTS: the EXACT 2026-05-07 afternoon memory-vs-record failure mode

**H2: requires explicit --venv** [v1] [🟡 stub]
- SETUP: any prior run dir
- ACTION: invoke without `--venv`
- EXPECTED: exit non-zero with explicit "sweep_state.json doesn't record resolved venv path; pass --venv"
- DETECTS: silent venv-from-memory failure mode

**H3: argparse re-use (DRY) catches old flag names** [v1] [🟡 stub]
- SETUP: synthesized `sweep_state.json` with `args: ["--allow-bare-cohort"]` (current name); CLI argparse intact
- ACTION: derive_spec
- EXPECTED: maps to `allow_flags: ["bare-list"]` correctly
- DETECTS: incorrect arg parsing

**H4: dynamo_config repeats** [v1] [🟡 stub]
- SETUP: synthesized `sweep_state.json` with `args: [..., "--dynamo-config", "a=1", "--dynamo-config", "b=2"]`
- ACTION: derive_spec
- EXPECTED: spec has `dynamo_config: {"a": "1", "b": "2"}`
- DETECTS: `action="append"` flag parsing

### Group I — Skill enforcement (§9)

**I1: STOP header present in skills/sweep.md** [v1] [🟡 stub]
- SETUP: working tree
- ACTION: `tools/test_skill_headers.py`
- EXPECTED: exit 0
- DETECTS: §9 silently dropped

**I2: STOP header present in skills/test-sweep-changes/SKILL.md** [v1] [🟡 stub]
- Same as I1 for the other skill

**I3: nightly carve-out paragraph present** [v1] [🟡 stub]
- SETUP: working tree
- ACTION: `tools/test_skill_headers.py`
- EXPECTED: exit 0; nightly carve-out paragraph found verbatim
- DETECTS: carve-out edit detected

**I4: STOP header drift detection** [v1] [🟡 stub]
- SETUP: edit `skills/sweep.md` to remove a word from STOP header
- ACTION: `tools/test_skill_headers.py`
- EXPECTED: exit non-zero identifying which header is missing/modified
- DETECTS: silent skill-doc drift

### Group J — Cross-cutting integration (Phase 4-5)

**J1: NGB verify spec validates clean** [v1] [🟡 stub]
- SETUP: `experiments/specs/ngb-verify-2026-05-07.json` (the spec we'll write in Phase 5)
- ACTION: `--validate`
- EXPECTED: pass
- DETECTS: any v1 design rule that the real NGB verify spec can't satisfy

**J2: NGB verify gate emission runs (foreground, ~5 min)** [v1] [🟡 stub]
- SETUP: NGB verify spec
- ACTION: emit gate; run; capture exit
- EXPECTED: gate runs to completion; post_sweep_check passes
- DETECTS: end-to-end real-stack emission failure

**J3: NGB verify sample emission runs (~15 min)** [v1] [🟡 stub]
- SETUP: NGB verify spec
- ACTION: emit sample; run; capture exit
- EXPECTED: sample runs to completion; post_sweep_check passes
- DETECTS: integration of sub-sample preservation + post_sweep_check

(J4 = full canonical NGB verify launch; tracked outside the test contract because it's the end goal, not a unit test.)

### Group K — v1.1 reserved

**K1: derive_wake_cron.py extracted from inline emission** [v1.1] [⏸️]
**K2: --refresh-content-hashes helper** [v1.1] [⏸️]
**K3: Form 4 (cohort.sources) for nightly** [v1.1] [⏸️]
**K4: cohort.cache field re-introduced** [v1.1 if there's demand] [⏸️]

---

## Test count

- v1: 36 tests (Groups A-J)
- v1.1: 5 tests (C4 + Group K)
- Total: 41

(The "36 tests" estimate in `design/sweep-spec-design.md` rev 3 was approximate — this contract is the actual count.)

## Phase 0 deliverable definition

Phase 0 is COMPLETE when:
1. `tools/test_sweep_spec.py` (or appropriate split into multiple files) exists with all v1 tests above as failing stubs
2. `tools/test_skill_headers.py` exists and passes (Group I)
3. Each test's docstring/comment references this file's test name (e.g., `# Test C5: source_sha256 placeholder detected`) for cross-reference

Phase 1 is COMPLETE when all v1 tests turn green.

## Process discipline

- Each test added during implementation gets a row added here
- Each test marked v1.1 cannot be promoted to v1 without an adversary review of WHY (per "start simple")
- This file is committed alongside Phase 0 code and updated whenever the contract changes
