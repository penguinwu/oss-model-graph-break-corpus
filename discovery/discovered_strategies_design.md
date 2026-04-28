# Discovered-Strategies Artifact Design

**Status:** Proposal v0.2 — Peng-blessed Q4/Q5/Q6/Q7; Q2 = store both forms (don't choose); Q1 explained + scoped (full first, minimal opportunistic); Q3 awaiting confirmation on pin policy
**Author:** Otter
**Created:** 2026-04-28
**Updated:** 2026-04-28 (post-Peng feedback)

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

## Design decisions (post Peng review 2026-04-28)

### Q1 — broken form variants: BOTH, full first

A *full* form per case = the model as shipped (e.g. transformers' modeling_vits.py + a baseline.py). Realistic eval surface; agents must navigate ~1500-line files.

A *minimal* form per case = a stripped-down standalone Python file (~50 lines) that isolates ONE break shape. Focused pattern-recognition eval; agents see the break in isolation.

**Decision:** support both. Author the *full* form first per case (high value, comes from the discovery trial). Add *minimal* repros opportunistically when a strategy proves cross-case-generalizable (e.g. when S7's static-cap appears in a 2nd case, factor it into a minimal repro).

Layout:
```
<case_id>/
  broken/           # full form (model file + baseline + canonical_inputs)
  broken_minimal/   # OPTIONAL minimal-repro form, when authored
```

### Q2 — multi-strategy reference fixes: store all info, don't pre-decide

Don't choose between "per-strategy isolated diffs" vs "combined end-to-end fixes" — store *both kinds of evidence*, let consumers extract what they need.

**What gets stored per strategy** (`strategies/<S_id>/`):
- `diff.patch` — the minimal diff applying ONLY this strategy where possible (teaching artifact)
- `applies_at.json` — file:line + applicability conditions (machine-readable)
- `requires.txt` — IDs of other strategies this one depends on (e.g. S6 requires S1's `@torch.jit.script` removal? — author-provided based on what the trial showed)

**What gets stored per reference fix** (`fixed_reference/<combo>/`):
- `<model_file>.py` — full file at fixed state
- `baseline.py` — full baseline (may equal broken's, may differ)
- `applies_strategies.txt` — which strategy_ids this fix combines
- `verify.sh` + `result_expected.json` — runnable verification
- `provenance.txt` — which trial first produced this combination

A consumer wanting just S7's diff goes to `strategies/S7/diff.patch`. A consumer wanting an end-to-end runnable fix goes to `fixed_reference/with_S7/`. A consumer wanting "all strategies the agent applied in trial X" reconstructs from `applies_strategies.txt`.

### Q3 — version pinning: AWAITING PENG CLARIFICATION

I conflated two pins in v1. They're independent:
- **PyTorch pin** — torch SHA used for verification.
- **HuggingFace transformers pin** — transformers version the broken file was sliced from.

**My lean for both:** pin at first verification, periodically re-verify on the cron sweep schedule. When re-verification fails (PyTorch evolved away the break, or HF refactored), update entry status `verified` → `superseded` rather than delete (preserves historical record).

**Awaiting confirmation:** is "pin both + periodic re-verify" the right policy, or do you want different policies for each? *(See message thread.)*

### Q4 — verification gate before publishing: YES (per Peng)

A strategy entry ships only after passing BOTH:
- **Apparent-fix verdict:** `gb_under_canonical_inputs == 0`
- **Correct-fix verdict:** `max_diff_compiled_vs_eager` within case-specific noise floor (declared in `case.md`)

Only verified-correct strategies enter the catalog. Apparent-fix-only entries (e.g. S10 `torch._check` for VITS — passes apparent but Inductor still fails to lower) get a `partial-verified` status with explicit notes about what's missing.

### Q5 — trial provenance: COPY (per Peng)

Each catalog entry copies the source trial files into a `provenance/` subdir (durable archive) rather than linking to `/tmp/runs/...` (volatile).

```
strategies/<S_id>/
  provenance/
    trial_id.txt          # e.g. "vits-r5v9-smoke/noskill_V9_1"
    agent_diff.patch      # copy of agent's full diff (not just the S-strategy diff)
    stream.jsonl.gz       # gzipped agent transcript
    result.json           # the trial's full result
```

Cost: ~5 MB per strategy (mostly stream.jsonl). Acceptable.

### Q6 — negative discoveries: INCLUDE IN SCOPE (per Peng)

What the agent tried and *abandoned* is valuable steering signal — tells us what to dissuade future agents from. Promoted from "out of scope" to first-class artifact.

Layout per case:
```
<case_id>/
  negative_discoveries/          # NEW per Peng 2026-04-28
    README.md                    # how to read this directory
    <attempt_id>/                # one dir per discovered-then-abandoned attempt
      attempt.md                 # what the agent tried, why it failed, why it was abandoned
      attempt_diff.patch         # the abandoned edit (extracted from stream.jsonl)
      failure_evidence.txt       # output that surfaced the failure (compile error, perf regression, max_diff blowup)
      provenance/
        trial_id.txt
        relevant_stream_excerpt.jsonl   # the turns where the attempt happened
```

**Examples we already see in the corpus** (from smoke + wave1a transcripts):
- **NA1 — `torch.compile(model, fullgraph=False)`** — tried implicitly in early agent turns, abandoned because it doesn't produce graph-break-count signal we need.
- **NA2 — `torch._dynamo.disable` decorator** — agent considers; usually abandons because it dodges rather than fixes.
- **NA3 — `lru_cache` on the offending function** — V0 SKILL agent attempted; abandoned when it didn't help (the function's args aren't hashable).
- **NA4 — Removing the data-dep call entirely (just delete the `predicted_lengths.max()` line)** — agent considers; abandons because output mask construction breaks.

Extraction tooling: a `discovery/extract_negative_discoveries.py <trial_dir> <out_dir>` that scans `stream.jsonl` for `Edit`/`Bash` tool calls followed by failure (compile error, validation regression, agent-stated rollback).

**Note on novelty:** a "negative discovery" is itself a discovery — sometimes the agent's abandoned attempt is the right answer for a *different* break shape. Cross-reference with the strategy catalog when this happens.

### Q7 — cross-case strategy promotion: YES, top-level `_patterns/` (per Peng's lean)

When a strategy demonstrates cross-case reuse, promote it to a top-level pattern entry. Per-case entries become specializations.

```
discovered_strategies/
  _patterns/                                      # NEW
    S7_static_max_arange/
      pattern.md                                   # the abstract pattern, applicability rules
      cases.txt                                    # which cases use this pattern
      generalization_notes.md                      # what changes case-to-case, what stays
  vits_model_train/
    strategies/
      S7_static_max_frames_per_token/             # specialization of _patterns/S7_static_max_arange/
        specializes: ../../_patterns/S7_static_max_arange/
        ...
```

**Promotion trigger:** a strategy is promoted to a top-level `_pattern` after it appears in 2+ cases. Until then, it lives only as a per-case entry.

## Updated layout (post-decisions)

```
discovered_strategies/
  README.md                           # what this catalog is, how skillwatch consumes it
  SCHEMA.md                           # canonical schema spec for each artifact below
  _patterns/                          # cross-case promoted patterns (Q7)
    <pattern_id>/
      pattern.md
      cases.txt
      generalization_notes.md
  <case_id>/                          # one dir per case
    case.md
    broken/                           # full-form broken state
      <model_file>.py
      baseline.py
      canonical_inputs.json
      requirements.txt
      README.md
    broken_minimal/                   # OPTIONAL minimal-repro variant
      ...
    strategies/<strategy_id>/         # per-pattern entry
      strategy.md
      diff.patch                      # minimal diff for this strategy
      applies_at.json                 # machine-readable applicability
      requires.txt                    # other strategies this depends on
      verified_correct.json           # apparent + correct-fix verdict
      provenance/                     # COPY (not link) of source trial
        trial_id.txt
        agent_diff.patch
        stream.jsonl.gz
        result.json
    fixed_reference/<combo_label>/    # known-correct end-to-end fix
      <model_file>.py
      baseline.py
      verify.sh
      result_expected.json
      applies_strategies.txt
      provenance.txt
    negative_discoveries/             # NEW per Q6 — abandoned attempts
      README.md
      <attempt_id>/
        attempt.md
        attempt_diff.patch
        failure_evidence.txt
        provenance/
          trial_id.txt
          relevant_stream_excerpt.jsonl
```

## What this design is NOT

- NOT a skill catalog (that lives elsewhere; this feeds skill catalog growth)
- NOT a benchmark (no perf-comparison framework)
- NOT a replacement for the trial-level raw evidence (those stay; this is the curated layer above them)

## Sequencing

If you bless this design (pending Q3 confirmation), sequence is:
1. Create `discovered_strategies/SCHEMA.md` + `README.md` (the contract)
2. Build `discovered_strategies/vits_model_train/` end-to-end as the prototype: broken/, strategies/S1–S11, fixed_reference/{with_S7, with_S6_declared}, negative_discoveries/{NA1–NA4}
3. Write extractor tooling:
   - `discovery/extract_strategy.py <trial_dir> <strategy_id>` — turns a trial into a strategy entry (semi-automated; human still authors strategy.md)
   - `discovery/extract_negative_discoveries.py <trial_dir> <out_dir>` — scans stream.jsonl for tool-call-then-rollback patterns
4. Write the skillwatch consumer-facing README explaining how to run a case
5. Backfill `mistral3_data_dep` (case 3a) when bandwidth allows

Estimated effort:
- Steps 1–2 for VITS: ~3 hours after VITS report wraps (added negative_discoveries adds ~1 hour)
- Step 3 (extractor tooling): ~4 hours (was 3 — extra for negative-discoveries extractor)
- Steps 4–5: as encountered

## Revision log

- 2026-04-28 v0.1: created with 7 open questions
- 2026-04-28 v0.2: post-Peng feedback — Q4/Q5/Q7 confirmed; Q2 = store both, don't pre-decide; Q1 = both forms (full first); Q6 promoted to in-scope (negative discoveries are valuable steering signal); Q3 = awaiting clarification on PyTorch vs HF pin policy
