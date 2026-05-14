# SKIP_LIST — GB patterns deliberately not filed as issues

Purpose: track GB patterns we identify but consciously decide NOT to file as separate issues, with the reason. Lets us see "what we deliberately ignored" when revisiting backlog.

When to add an entry:
- A triage surfaces a GB pattern (in sweep results)
- We decide not to file (WONTFIX-class / duplicate / too-scattered / tracked-elsewhere / too-small)

Each entry: signature + cited source line + scope at last triage + skip reason + date + revisit-trigger.

If a skipped pattern's scope grows past its revisit threshold, the next triage should reconsider.

---

## Entries (most recent first)

### 2026-05-14 — "Observed exception" cluster (scattered across 5 source lines)

- **Signature:** `Observed exception` (Failed to handle graph break gracefully; falls back to eager)
- **Cited source lines (5 distinct):**
  - `transformers/models/longt5/modeling_longt5.py:74` — 3 classes × 6 breaks
  - `transformers/utils/output_capturing.py:251` — 3 classes × 6 breaks (same helper as #11/#23/#24/#96 but different code path)
  - `transformers/models/switch_transformers/modeling_switch_transformers.py:704` — 3 classes × 6 breaks
  - `transformers/models/udop/modeling_udop.py:1173` — 2 classes × 4 breaks
  - `transformers/models/pop2piano/modeling_pop2piano.py:687` — 1 class × 2 breaks
- **Scope at triage:** 9 classes × 24 breaks aggregate; 1-3 classes per source line
- **Skip reason:** scattered across 5 distinct source lines with 1-3 classes each. No concentration. Filing 5 separate issues for 24 total breaks doesn't move the dial. Each subset is too small to be its own actionable issue.
- **Revisit trigger:** if any single source-line subset grows to ≥6 model classes OR aggregate scope grows past 50 breaks.

### 2026-05-14 — "Failed to trace builtin operator" residual (44/56 already tracked by #24)

- **Signature:** `Failed to trace builtin operator` — `Dynamo does not know how to trace builtin operator setattr with argument types ['type', 'str', 'int']`
- **Cited source lines:**
  - `torch/utils/hooks.py:27` — 11 classes × 44 breaks → **TRACKED-by-#24** (same `register_*_hook` setattr root)
  - `transformers/models/ernie4_5_vl_moe/modeling_ernie4_5_vl_moe.py:1029` — 2 classes × 8 breaks (residual)
  - `transformers/models/vits/modeling_vits.py:792` — 1 class × 2 breaks (residual)
  - `transformers/models/vits/modeling_vits.py:594` — 1 class × 2 breaks (residual)
- **Scope at triage:** 14 classes × 56 breaks aggregate; 11/14 classes (44/56 breaks) are #24's existing scope; residual 3 classes × 12 breaks across 3 source lines.
- **Skip reason:** main concentration tracked by #24. Residual is 3 classes × 12 breaks across 3 separate source lines — too scattered to file individually.
- **Revisit trigger:** if any residual source-line subset grows to ≥4 model classes.

### 2026-05-14 — Awareness/WONTFIX-class candidates (4 patterns, batch decision)

Identified during 2026-05-13 dynamo triage. After cost-benefit (WONTFIX-class, full subagent walk per filing, low signal to PT2 team), batch-skipped 2026-05-14 13:39 ET per Peng directive.

- **Reformer `torch.Generator.seed`**
  - Source: `transformers/models/reformer/modeling_reformer.py:1501`
  - Scope: 1 class × 8 breaks (`ReformerModel|train`)
  - Skip reason: dynamo's own hint says "fundamental — unlikely Dynamo will ever trace through this." WONTFIX-class.
  - Revisit trigger: if dynamo upstream announces a fix path for `torch.Generator` method tracing.
- **Reformer `manual_seed` skipped**
  - Source: `transformers/models/reformer/modeling_reformer.py:1506`
  - Scope: 1 class × 4 breaks (`ReformerModel|train`)
  - Skip reason: dynamo intentionally skips `torch.manual_seed` (function marked as skipped). WONTFIX-class by design.
  - Revisit trigger: if dynamo's skip-list policy changes for manual_seed.
- **Vits missing `tp_iter` (reversed)**
  - Source: `transformers/models/vits/modeling_vits.py:792`
  - Scope: 1 class × 4 breaks (`VitsModel|eval+train`)
  - Skip reason: 1-class scope on a Python iterator-protocol limitation. Too small.
  - Revisit trigger: if scope grows to ≥3 model classes.
- **Encodec RNN/GRU/LSTM wrap**
  - Source: `transformers/models/encodec/modeling_encodec.py:247`
  - Scope: 1 class × 2 breaks (`EncodecModel|eval+train`)
  - Skip reason: documented opt-in workaround (`torch._dynamo.config.allow_rnn=True`). Maintainer is aware; user has a knob.
  - Revisit trigger: if a fix changes the default OR scope grows to ≥3 model classes.

### 2026-05-13 — Encodec recompile-limit (TRACKED-by-#98)

- **Signature:** `recompile_limit (8)` errors on `ParametrizedConv1d` type churn
- **Source:** `transformers/models/encodec/modeling_encodec.py:171`
- **Scope at triage:** 1 class × 4 breaks (`EncodecModel|eval+train`)
- **Skip reason:** TRACKED-by-#98 — same `parametrize.weight_norm` / `___check_type_id` churn pattern at the exact same source line. Not a new issue.
- **Revisit trigger:** N/A (covered).

### 2026-05-13 — BigBirdPegasus Uninitialized nn.Module (downstream of #27)

- **Signature:** `Attempted to trace an uninitialized nn.Module of type BigBirdPegasusSelfAttention`
- **Source:** `transformers/models/bigbird_pegasus/modeling_bigbird_pegasus.py:114`
- **Scope at triage:** 2 classes × 4 breaks (`BigBirdPegasusForConditionalGeneration|eval+train`, `BigBirdPegasusModel|eval+train`)
- **Skip reason:** structurally inseparable from #27 in any synthetic MRE form. The break is the downstream consequence of #27's `nn.Parameter()` ctor break in the same compile-trace; under `fullgraph=True` the MRE halts on #27 before reaching this break. Cross-ref filing dropped per Peng Q3 decision 2026-05-14 13:39 ET (option (c): drop entirely; maintainer will discover when investigating #27 — sweep evidence #27 references shows it).
- **Revisit trigger:** if a future SKILL extension provides a `--comment` mode for the file-issue subagent (then can post the cross-ref as a comment on #27). OR if the downstream break is observed in models OTHER than the BigBirdPegasus pair (would indicate it's not just-downstream-of-#27).

---

## Maintenance

When adding an entry:
1. Run `tools/dedup_source_lines.py` against the cited source line(s) — confirm no existing tracking issue covers it.
2. Note scope at triage time (model_classes × breaks, dedup-suppressed-filtered per R12).
3. State revisit trigger explicitly (when does this entry need re-evaluation?).
4. Date the entry.

Periodic review: when running a triage, scan the SKIP_LIST first — has any entry's scope grown past its revisit trigger? If so, re-evaluate.
