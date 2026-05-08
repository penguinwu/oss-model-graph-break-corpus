---
name: test-sweep-changes
description: Use BEFORE editing any corpus sweep harness file (sweep/worker.py, sweep/orchestrator.py, sweep/models.py, sweep/explain.py, sweep/run_sweep.py, sweep/venv_setup.py, tools/run_experiment.py) AND BEFORE proposing or launching a full sweep. Walks through 5 testing gates (unit → smoke → single-trial → reproducibility → mini-sweep) to catch script bugs at low compute cost rather than discovering them in a multi-hour full sweep.
---

# test-sweep-changes

## When to use

**Trigger conditions (any one):**
- About to edit `sweep/worker.py`, `sweep/orchestrator.py`, `sweep/models.py`, `sweep/explain.py`, `sweep/run_sweep.py`, `sweep/venv_setup.py`, or `tools/run_experiment.py` in the corpus repo
- About to propose, schedule, or launch a full sweep (any pass: identify, explain, correctness, validate)
- About to push a commit that touches any of the above files

**Do NOT use for:** changes to `tools/analyze_*.py`, `tools/check_*.py`, docs, or test files themselves — those are Tier C ad-hoc.

## Why this exists

A full sweep is multi-hour. Discovering script bugs in a full sweep wastes wall-clock and erodes trust in headline numbers. Each gate below catches a different bug class at low compute cost. The recipe was failed once (2026-05-01) when worker.py shipped on direct-worker smoke tests alone, bypassing the orchestrator pipeline — caught only after Peng prompted for testing methodology. This skill exists so the recipe is harder to miss.

## CRITICAL: Gates 2-5 must be CUSTOMIZED to the planned launch

**Encoded 2026-05-07 (Peng directive: "I think the gated runs w/o customizing to the actual run we are launching is a serious anti-pattern we should prevent for the future. It gives us the false sense of confidence. Very bad!")**

Generic gates that use a different stack, different worker count, different mode set, or different model corpus from the planned launch DO NOT VALIDATE THE LAUNCH. They give false confidence. Every observed instance:

- 2026-05-07: M2 pre-flight ran Gates 2-5 on torch211 + canary models + 2 workers. The actual M2 launch was torch-nightly-cu126 + 20 random cohort models + 4 workers + train mode. The standard gates passed; the M2-customized gate caught a JSONL-parsing bug in `tools/check_cohort_invariants.py` that would have silently failed the post-sweep wake.
- (Future instances of this anti-pattern get appended here as they occur.)

**Therefore, BEFORE walking gates 2-5:** state the planned launch configuration explicitly (stack, worker count, modes, cohort source, key flags). Then customize EVERY following gate to that configuration.

If you find yourself thinking "let me just run the standard gates first," STOP — that's the false-confidence anti-pattern. The standard fixed-model gates are a HARNESS REGRESSION CHECK ONLY (Gate 1 unit tests + Gates 2-4 as harness smoke). They do NOT substitute for the launch-customized gate.

The launch-customized gate (replacing the old "Gate 5") MUST use:
- The EXACT venv + python interpreter the launch will use
- The EXACT modellib versions (--transformers, --diffusers, --timm) the launch will pass
- A SUB-SAMPLED subset of the EXACT cohort the launch will use (use `tools/sample_cohort.py` with a smaller --n)
- The EXACT modes (--modes eval train if launch uses both)
- The EXACT worker count (--workers N matching the launch)
- The EXACT auto-retry setting (do NOT pass --no-auto-retry just to make the gate "cleaner" if the launch keeps it on)
- The same `--allow-*` flags the launch will use

This gate proves the launch path itself works on a microcosm of the launch, before committing to the full launch's compute.

## The gates (5 + customized; strict order, no skipping)

Skipping a gate requires Peng's explicit approval, in writing, in the conversation. Otherwise: walk all gates IN ORDER.

State your gate progression explicitly in the conversation as you go. Example: "Gate 1 ✅ — 9/9 pass."

### Gate 1 — Unit tests (~1 min)

```bash
cd ~/projects/oss-model-graph-break-corpus
~/envs/torch211/bin/python sweep/test_explain.py
# (and any other sweep/test_*.py files)
```

**Pass criterion:** all tests exit 0. Failures here mean a static regression — fix BEFORE any other gate.

### Gate 2 — Smoke test (~2 min)

```bash
cd ~/projects/oss-model-graph-break-corpus
~/envs/torch211/bin/python tools/smoke_test.py --python ~/envs/torch211/bin/python
```

8 fixed models with expected statuses (3 full_graph, 4 graph_break, 1 create_error). **Pass criterion:** 9/9 pass (8 models + 1 enumeration count check).

### Gate 3 — Single-trial field enumeration (~30s)

```bash
cd ~/projects/oss-model-graph-break-corpus
~/envs/torch211/bin/python sweep/worker.py \
  --model-json '{"name":"BayesianDetectorModel","source":"hf"}' \
  --pass-num 1 --device cuda --mode eval 2>/dev/null | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
print('=== FIELDS ===')
for k in sorted(r.keys()):
    s = repr(r[k])
    print(f'  {k:35s} = {s[:80]}')
# CUSTOMIZE these assertions for what your change introduces:
assert r['status'] == 'full_graph'
assert r['numeric_status'] == 'match'
print('PASS')
"
```

**Pass criterion:** every expected field present, no unexpected `null`/`NaN`/missing-required-field. If your change adds new fields, assert their presence + correct values here.

### Gate 4 — 3-trial reproducibility (~1 min)

```bash
cd ~/projects/oss-model-graph-break-corpus
for i in 1 2 3; do
  echo -n "Trial $i: "
  ~/envs/torch211/bin/python sweep/worker.py \
    --model-json '{"name":"BayesianDetectorModel","source":"hf"}' \
    --pass-num 1 --device cuda --mode eval 2>/dev/null | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
# CUSTOMIZE: print fields where reproducibility matters for your change
print(f'numeric_max_diff={r.get(\"numeric_max_diff\")} status={r.get(\"status\")}')"
done
```

**Pass criterion:** values are bit-identical across the 3 trials (or explicitly noise-bounded — and you state the bound). Drift here means RNG leakage or state contamination.

### Gate 5 — Launch-Customized Mini-Sweep (~3-5 min)

**This is the gate that actually validates the launch.** Gates 1-4 are harness regression checks only — they do not substitute for this gate. Per the "CRITICAL: Gates 2-5 must be CUSTOMIZED" section above.

**If no specific launch is planned** (you're just doing a harness sanity check after a code edit, with no specific sweep launch coming up): you may use the canary fallback below.

**If a specific launch IS planned** (you intend to invoke `tools/run_experiment.py sweep` with specific flags / cohort / stack within this session or the next): you MUST customize this gate to match.

**Preferred path: `tools/derive_sweep_commands.py --stage gate --run`.** If the launch has a canonical experiment config with `settings.python_bin` + `settings.modellib_pins`, the derive tool mechanically guarantees the gate matches the full launch — same flags, same stack, deterministic 5-model sub-sample. After the derived gate passes (recorded in `/tmp/derive_sweep_state/`), `--stage sample --run` becomes the next step (matches §Sample-sweep gate of `skills/sweep.md`).

The hand-rolled procedure below is the FALLBACK for ad-hoc launches that don't have a canonical config (e.g., one-off `tools/run_experiment.py sweep --compile-kwargs ...` invocations).

#### Customized gate (when launch is planned)

```bash
cd ~/projects/oss-model-graph-break-corpus
# 1. State the planned launch configuration explicitly (paste in conversation):
#    - venv:        e.g. ~/envs/torch-nightly-cu126/bin/python
#    - --transformers VER   (e.g. 5.6.2)
#    - --diffusers VER      (e.g. 0.38.0)
#    - --workers N          (e.g. 4)
#    - --modes ...          (e.g. eval train)
#    - --timeout S          (e.g. 180)
#    - --identify-only?     (yes/no)
#    - --auto-retry?        (yes/no — default is on; --no-auto-retry disables)
#    - cohort source:       e.g. experiments/configs/<name>.json
#    - any --allow-* flags  (e.g. --allow-bare-cohort)

# 2. Sub-sample 5 models from the planned launch's cohort (deterministic)
~/envs/<launch-venv>/bin/python tools/sample_cohort.py \
  <planned-launch-cohort.json> --n 5 --seed 42 \
  --output /tmp/launch-customized-gate.json --force

# 3. Run with the EXACT planned-launch flags (just substituting --models for the sub-sample)
OUT=sweep_results/experiments/launch-customized-gate-$(date +%Y-%m-%d-%H%M%S)
mkdir -p "$OUT"
~/envs/<launch-venv>/bin/python tools/run_experiment.py sweep \
  --models /tmp/launch-customized-gate.json \
  --transformers <VER> --diffusers <VER> \
  --modes <eval/train> --workers <N> --timeout <S> --device cuda \
  --identify-only \
  --output-dir "$OUT" --run-name launch-customized-gate \
  --allow-bare-cohort   # since the sample is bare; preserve other --allow-* matching the launch
  # DO NOT add --no-auto-retry unless the launch will also pass it

# 4. Run the post-sweep invariant check on the result (proves the wake-cron path works too)
~/envs/<launch-venv>/bin/python tools/check_cohort_invariants.py --post-sweep "$OUT/identify_results.json"
```

**Pass criterion:**
- All sub-sample models complete (no orchestrator crash, no unexplained timeouts)
- `identify_results.json` parses cleanly via `check_cohort_invariants.py --post-sweep`
- Mix of statuses observed (graph_break and/or full_graph), no create_error / eager_error / worker_error / timeout
- modellibs PYTHONPATH injection visible in launch output (`[run_sweep] modellibs: transformers==X (flag)` line)
- Both modes exercised if launch uses both
- Multi-worker concurrency exercised if launch uses N>1 workers

**Fail handling:** any failure means the launch path has a bug. STOP — do not launch. Diagnose the failure. The whole point of this gate is to catch launch-path bugs at 5-min cost instead of N-hour cost.

#### Canary fallback (when no specific launch is planned)

(Only when you're doing a harness regression check after a code edit, with no immediate sweep launch.)

```bash
cd ~/projects/oss-model-graph-break-corpus
cat > /tmp/mini_sweep_models.json <<'EOF'
[
  {"name": "BayesianDetectorModel", "source": "hf"},
  {"name": "BeitModel", "source": "hf"},
  {"name": "Blip2VisionModel", "source": "hf"},
  {"name": "ConvBertModel", "source": "hf"},
  {"name": "HubertModel", "source": "hf"},
  {"name": "Data2VecAudioModel", "source": "hf"},
  {"name": "FSMTModel", "source": "hf"},
  {"name": "DistilBertModel", "source": "hf"},
  {"name": "AlbertModel", "source": "hf"},
  {"name": "MobileViTModel", "source": "hf"}
]
EOF
OUT=sweep_results/experiments/test-sweep-changes-$(date +%Y-%m-%d-%H%M%S)
mkdir -p "$OUT"
SWEEP_PYTHON=~/envs/torch211/bin/python ~/envs/torch211/bin/python tools/run_experiment.py sweep \
  --models /tmp/mini_sweep_models.json --modes eval --workers 2 \
  --identify-only --output-dir "$OUT" --no-auto-retry --allow-bare-cohort
~/envs/torch211/bin/python tools/analyze_sweep.py "$OUT/identify_results.json"
```

**Pass criterion:**
- All 10 models run (no timeouts, no orchestrator crash)
- `identify_results.json` aggregates correctly
- Schema consistency: `status` ↔ field-presence consistent
- `analyze_sweep.py` renders without error

If your change adds new result fields, **add a Python script that reads `identify_results.json` and asserts schema consistency for those fields across all 10 models** — paste the output in the conversation.

Note: this canary set runs on torch211 + transformers 5.5.3 + diffusers 0.37.1 (defaults_per_venv). It validates harness wiring but DOES NOT validate any launch on a different stack — see "CRITICAL" section above.

### Optional — Forced-edge-case test

If your change has a code path that the 10 mini-sweep models won't naturally exercise (e.g., a retry on divergence when divergence is rare), write a focused test that monkey-patches or otherwise forces that path. Without this, you cannot claim the path is validated. Pattern: see `/tmp/test_retry_path.py` from 2026-05-01 (forced-divergence test of the less_flaky retry).

## After all gates pass

State the gate-progression summary in the commit message or launch report. Recommended template:

```
Validation per test-sweep-changes:
- Gate 1 (sweep/test_explain.py): N/N pass
- Gate 2 (tools/smoke_test.py): N/N pass on <env>
- Gate 3 (single-trial field enumeration on <model>): <result summary>
- Gate 4 (3-trial reproducibility): bit-identical / <bound>
- Gate 5 (LAUNCH-CUSTOMIZED mini-sweep — venv=<X>, modellibs=transformers=<V>+diffusers=<V>,
          workers=<N>, modes=<M>, sub-sample of <cohort.json>):
  <result summary including: which models, all completed, status mix,
   modellibs PYTHONPATH injection confirmed in launch output>
- [Forced-edge-case test if applicable]: <result>

Confirmed: gate matches the planned launch's stack + flags (no anti-pattern; per
the CRITICAL section of skills/test-sweep-changes/SKILL.md).
```

## Failure modes

- **Gate fails after a fix attempt:** stop. Diagnose. Don't loop "fix → re-run" without inspecting the failure.
- **Mini-sweep too slow:** the 10-model set above is the floor. Going smaller defeats the purpose. If wall-clock is a problem, run on a faster venv but don't drop the gate.
- **No torch211 venv available:** use whatever stable venv you have but state which env in the gate report. Re-run on the target sweep env BEFORE the full sweep.
