# Discovered-Strategies Artifact Design

**Status:** Proposal, awaiting Peng review
**Author:** Otter
**Created:** 2026-04-28

## The use case

The corpus's *product* is the catalog of graph-break-fixing strategies that real models + real agents discover. The downstream consumer is a skillwatch-style evaluation: hand a broken model to an agent under test, watch which strategy it reaches for, score it.

For that to work, each catalog entry must be **runnable end-to-end without depending on our discovery harness**. The agent under test should be able to:

1. Receive the broken model + canonical inputs as a self-contained package
2. Edit it (possibly with a skill loaded)
3. Run a verify script that returns a clean verdict (compile-clean? canonical_gb=0? output within tolerance?)
4. Be compared against known-correct reference fixes (so the evaluator can classify which strategy was applied)

## What we have today

Per discovery trial we keep: `agent_diff.patch`, `stream.jsonl`, `prompt.txt`, `result.json`, `validation_*.log`, `perf_*.log`, `sandbox/` (full snapshot of agent's edited files), `claude_stderr.log`, `run_config.log`. These are *raw evidence*, not curated artifacts.

What we do *not* have:
- A self-contained "broken case" package (skillwatch-runnable)
- A first-class strategy index (each pattern named, described, located)
- Reference fixes (known-correct end-to-end model files for each strategy)
- Strategy applicability metadata (where the pattern can be reused across cases)
- Apparent-fix vs correct-fix distinction (we have `max_diff_compiled_vs_eager` but no derived "fix is numerically correct" verdict)

## Proposed layout

A new top-level directory `discovered_strategies/` (sibling to `discovery/`):

```
discovered_strategies/
  README.md                          # what this catalog is, how skillwatch consumes it
  SCHEMA.md                          # canonical schema for each artifact below
  <case_id>/                         # one dir per case (e.g. vits_model_train)
    case.md                          # case summary, link to discovery/cases/<case_id>.py
    broken/                          # the BROKEN form — what skillwatch gives agents under test
      <model_file>.py                # full file at the broken state (e.g. modeling_vits.py)
      baseline.py                    # standalone repro script (NO discovery harness deps)
      canonical_inputs.json          # test inputs that produce the break
      requirements.txt               # pinned deps (torch, transformers, ...)
      README.md                      # what break to expect, how to run
    strategies/
      <strategy_id>/                 # e.g. S1_remove_jit_script, S7_static_max_frames_per_token
        strategy.md                  # name, problem statement, generalization rule, status
        diff.patch                   # minimal diff applying ONLY this strategy
        applies_at.json              # file:line + applicability conditions (machine-readable)
        reference_trials.txt         # links to trial dirs that demonstrate it
        verified_correct.json        # apparent-fix + correct-fix verdict, with evidence
    fixed_reference/                 # known-correct end-to-end fixes (verified)
      <fix_label>/                   # e.g. with_S6_declared_overrides, with_S7_static_cap
        <model_file>.py              # full file at fixed state
        baseline.py                  # may equal broken/baseline.py (V9-class fix) or differ
        verify.sh                    # one-shot: runs both files, asserts canonical_gb=0
        result_expected.json         # what verify.sh should produce (gb=0, max_diff<tol)
        applies_strategies.txt       # which strategy_ids this fix combines
        provenance.txt               # which trial first produced this fix
```

## Schema details

### `strategy.md` (per-strategy entry)

Markdown with required headers:
- **Strategy ID** (e.g. `S7_static_max_frames_per_token`)
- **Name** (human-readable)
- **Problem** (what break does this address?)
- **Generalization rule** (when does this pattern apply outside this case?)
- **Status** (draft / verified / retired)
- **First surfaced** (date + trial ID)
- **Convergence** (count + description)
- **Trade-offs** (any soft constraints, perf cost, correctness caveats)
- **Code excerpt** (representative diff)

### `applies_at.json` (machine-readable applicability)

```json
{
  "strategy_id": "S7_static_max_frames_per_token",
  "case_id": "vits_model_train",
  "edit_sites": [
    {"file": "modeling_vits.py", "line_range": [1370, 1410], "function": "VitsModel.forward"}
  ],
  "applicability_pattern": {
    "ast_signature": "torch.arange(<tensor>.max())",
    "preconditions": [
      "input has a static dim that bounds the data-dep result",
      "downstream consumer accepts a padded result with mask"
    ]
  }
}
```

### `verified_correct.json`

```json
{
  "strategy_id": "S7_static_max_frames_per_token",
  "verified_at": "2026-04-28T11:00:00Z",
  "torch_version": "2.13.0a0+gitf8d66d2",
  "transformers_version": "5.6.2",
  "apparent_fix": {
    "compile_ok": true,
    "gb_under_canonical_inputs": 0
  },
  "correct_fix": {
    "max_diff_compiled_vs_eager": 2.0,
    "noise_floor": 2.0,
    "within_noise_floor": true,
    "verdict": "correct"
  },
  "perf": {
    "speedup_t1": 1.53,
    "speedup_t2": 1.39,
    "fix_survives_perf": true
  }
}
```

### `result_expected.json`

What `verify.sh` should produce when an agent under test correctly applies the strategy. Skillwatch diffs the agent's output against this:

```json
{
  "graph_break_count": 0,
  "max_diff_compiled_vs_eager": {"max": 2.0, "tolerance": "noise_floor"},
  "compile_ok": true
}
```

## Skillwatch consumption pattern

```
1. Skillwatch picks <case_id>/broken/ → hands files + canonical_inputs.json to agent under test
2. Agent edits files (with or without skill loaded)
3. Skillwatch runs <case_id>/broken/verify.sh against agent's edited state
4. Skillwatch compares agent's diff to <case_id>/strategies/*/diff.patch
   → classifies which strategy_id(s) the agent reached for
5. Skillwatch scores: did verify pass? which strategy? did the agent invent a NEW strategy
   not in the catalog (high-value signal — feed it back into the catalog)?
```

## Open design questions for Peng

1. **One broken form, or several?** A single canonical `broken/` per case keeps things simple. But for some cases we may want a "minimal-repro" variant (stripped to just the breaking call site) alongside the "full case" variant. Adds complexity; defer until skillwatch consumer asks for it?

2. **Multi-strategy reference fixes** — when a complete fix combines S1+S2+S6, do we keep:
   - (a) one `fixed_reference/with_S1_S2_S6/` with all three applied?
   - (b) per-strategy reference fixes that each apply ONE strategy?
   - I lean *both* — per-strategy `strategies/<S>/diff.patch` is minimal/teaching; `fixed_reference/<combo>/` is the runnable end-to-end. Per-strategy isolation may not always be possible (some strategies require others).

3. **Pin version, or track HEAD?** Pin tightly (exact torch SHA + transformers version) makes verification reproducible but ages fast. Track HEAD makes the catalog stay fresh but reproducibility weakens. I lean *pin at first verification, refresh when sweep schedule re-validates*.

4. **Verification gate before publishing** — require a strategy entry to pass *both* apparent-fix (canonical_gb=0) AND correct-fix (max_diff within noise floor) before adding to the catalog? I lean yes; only verified-correct strategies ship.

5. **Agent-trial provenance retention** — should `reference_trials.txt` link to the trial dirs (currently in `/tmp/runs/...`, not durable)? Or should each catalog entry copy the relevant trial files into its own subdir for archival? Lean copy — `/tmp/` is not stable.

6. **Negative discoveries** — what the agent tried and abandoned. Worth a `negative_attempts/` subdirectory per case? Or aggregated into a separate analysis layer? Lean *separate analysis layer* (out of scope for this design — proposed as a follow-up).

7. **Cross-case strategy reuse** — when S7's pattern (static-cap on data-dep arange.size) applies to a future non-VITS case, do we:
   - (a) symlink/reference the original strategy entry from the new case?
   - (b) create a new per-case entry that points back to S7's source-of-truth?
   - (c) elevate S7 to a top-level `discovered_strategies/_patterns/S7_static_arange_cap/` with cross-case applicability?
   I lean *(c)*: once a strategy demonstrates cross-case reuse, promote it to a top-level pattern. Per-case entries become specializations.

## What this design is NOT

- NOT a skill catalog (that lives elsewhere; this feeds skill catalog growth)
- NOT a benchmark (no perf-comparison framework)
- NOT a replacement for the trial-level raw evidence (those stay; this is the curated layer above them)

## Sequencing

If you bless this design, sequence is:
1. Create `discovered_strategies/SCHEMA.md` + `README.md` (the contract)
2. Build `discovered_strategies/vits_model_train/` end-to-end as the prototype (first case populated)
3. Write extractor tooling: `discovery/extract_strategy.py <trial_dir> <strategy_id>` — turns a trial into a catalog entry (semi-automated; human still authors strategy.md)
4. Write the skillwatch consumer-facing README explaining how to run a case
5. Backfill `mistral3_data_dep` (case 3a) when bandwidth allows

Estimated effort:
- Steps 1–2 for VITS: ~2 hours after VITS report wraps
- Step 3 (extractor tooling): ~3 hours
- Steps 4–5: as encountered

## Revision log

- 2026-04-28: created
