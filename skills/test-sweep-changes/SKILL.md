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

## The 5 gates (strict order, no skipping)

Skipping a gate requires Peng's explicit approval, in writing, in the conversation. Otherwise: walk all 5 IN ORDER.

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

### Gate 5 — Orchestrated mini-sweep (~3 min)

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
  --identify-only --output-dir "$OUT" --no-auto-retry
~/envs/torch211/bin/python tools/analyze_sweep.py "$OUT/identify_results.json"
```

**Pass criterion:**
- All 10 models run (no timeouts, no orchestrator crash)
- `identify_results.json` aggregates correctly
- Schema consistency: `status` ↔ field-presence consistent (e.g., full_graph models have expected fields; graph_break models have skip semantics)
- `analyze_sweep.py` renders without error and shows reasonable counts

If your change adds new result fields, **add a Python script that reads `identify_results.json` and asserts schema consistency for those fields across all 10 models** — paste the output in the conversation.

### Optional — Forced-edge-case test

If your change has a code path that the 10 mini-sweep models won't naturally exercise (e.g., a retry on divergence when divergence is rare), write a focused test that monkey-patches or otherwise forces that path. Without this, you cannot claim the path is validated. Pattern: see `/tmp/test_retry_path.py` from 2026-05-01 (forced-divergence test of the less_flaky retry).

## After all gates pass

State the gate-progression summary in the commit message. Recommended template:

```
Validation per test-sweep-changes (5 gates):
- Gate 1 (sweep/test_explain.py): N/N pass
- Gate 2 (tools/smoke_test.py): N/N pass on <env>
- Gate 3 (single-trial field enumeration on <model>): <result summary>
- Gate 4 (3-trial reproducibility): bit-identical / <bound>
- Gate 5 (mini-sweep, 10 models eval/identify-only): <result summary>
- [Forced-edge-case test if applicable]: <result>
```

## Failure modes

- **Gate fails after a fix attempt:** stop. Diagnose. Don't loop "fix → re-run" without inspecting the failure.
- **Mini-sweep too slow:** the 10-model set above is the floor. Going smaller defeats the purpose. If wall-clock is a problem, run on a faster venv but don't drop the gate.
- **No torch211 venv available:** use whatever stable venv you have but state which env in the gate report. Re-run on the target sweep env BEFORE the full sweep.
