# USE_CASES.md

How the corpus is used, by whom, and how new use cases plug in. This is the consumer-facing companion to the [design doc](design/design-doc.md), which describes what the corpus IS.

## Why this doc exists

The corpus produces a stable signal — per-model compiler-quality data (graph breaks, correctness divergences, reproducers) refreshed nightly. Different audiences consume that signal in different ways. As consumers multiply, this catalog keeps each use case explicit instead of letting them sprawl across READMEs and ad-hoc scripts.

The design doc and this doc change at different rates. Methodology changes are rare; new audiences arrive often. Keeping them separate prevents every new consumer from dirtying the methodology section.

---

## Use Cases

### 1. Issue filing → PT2 component owners
- **Status:** LIVE (graph breaks); LIVE for correctness as of 2026-04-21
- **Audience:** PT2 dynamo, dispatcher, inductor, decomposition teams
- **Signal consumed:** graph break records (pattern, model list, reproducer); correctness divergence records (max_diff, severity, divergent field, reproducer)
- **Output format:** GitHub issues on `penguinwu/oss-model-graph-break-corpus`, labeled by component, with model list and one-command repro
- **Code path:** `tools/file_issues.py`, `tools/github_issue_monitor.py`
- **Notes:** Per-component issue budgets are negotiated with maintainers to keep the queue actionable. Correctness-mode filing in active development.

### 2. Cross-version regression detection
- **Status:** LIVE
- **Audience:** PT2 release team, working-group leads
- **Signal consumed:** per-version sweep results (full_graph / graph_break / error counts)
- **Output format:** version reports (`results/pt2.X.md`), nightly diffs (`results/nightly/`), trend analysis
- **Code path:** `tools/compare_results.py`, `tools/analyze_trend.py`
- **Notes:** Zero full_graph → graph_break regressions across 2.8 → 2.11. The corpus *is* the regression detector for these models.

### 3. Per-pattern fixture set for PT2 diagnostic-skill evaluation (niche)
- **Status:** ACTIVE as of 2026-04-22 (niche framing — see Scope below)
- **Audience:** Maintainers of compiler-diagnostic skills who need clean, isolated, single-model reproducers for one specific pattern at a time.
- **Scope (read this before consuming).** The corpus is **not** a general skill-eval source. For broad skill capability evaluation — multi-step reasoning, ambiguous diagnosis, richer Q&A — use the doc-eval project's Q&A corpus instead. The corpus's niche is the opposite end: clean, isolated, single-model reproducers tied to known root causes, suitable for unit-test-style checks of diagnostic accuracy on individual graph breaks in a controlled environment.
- **Signal consumed:** graph break records (with reproducers and pattern labels) for graph-break skills; correctness divergence records for accuracy-debugging skills.
- **Output format:** test fixture export (JSONL), one entry per real-model finding with a self-contained code snippet, expected classification, and metadata about the originating model.
- **Code path:** TBD — likely a new `tools/export_skill_eval.py` parameterized by signal type.
- **Known consumers:**
  - Arsh Zahed's `debug-graph-breaks` skill (D99943226, currently Unpublished draft) consumes `corpus.json` directly via `generate_oss_corpus_evals.py` to auto-derive a SkillWatch eval suite. Three open TODOs intersect with the corpus: pegged env (consumer wants the latest stable PT), hardcoded local paths, and mid-execution agent stops.
- **Failure modes are deliverables.** When a consumer pilots a skill against the corpus, the failure catalog (what the skill missed, where the schema friction shows up, where the reproducer was insufficient) is itself a project output — not a side experiment. Consumer-driven contract feedback feeds the [AI-native maintenance](design/ai-native-maintenance.md) loop as a "Skill integration request" classification.
- **Notes:** Synthetic test suites built from registry stubs or curated examples can drift from real-world distributions. Real-model cases — produced as a side effect of the nightly sweep — give controlled-environment coverage on a per-pattern basis. They do not substitute for richer skill evaluation that requires natural-language Q&A or multi-step reasoning.

### 4. Auto-fix application (skill-as-benchmark loop)
- **Status:** PROPOSED (2026-04-21)
- **Audience:** PT2 users blocked on compiler issues while waiting for upstream fixes; skill maintainers measuring real-world uplift
- **Signal consumed:** graph break records + reproducers (initial scope); extensible to correctness records
- **Workflow:** apply a fix-suggestion skill to each affected model → re-sweep → measure pass-rate uplift and which patterns the skill resolves
- **Output format:** uplift report (per-skill, per-pattern); fixed-model artifacts
- **Code path:** TBD — likely `tools/apply_skill.py` + reuse `sweep/run_sweep.py`
- **Notes:** Turns the corpus from bug-surface into benchmark. Closes the loop between detection and remediation while upstream fixes are pending.

### 5. Registry liveness audit
- **Status:** PROPOSED
- **Audience:** PyTorch Graph Break Registry maintainers
- **Signal consumed:** graph break patterns from sweep results, joined against Registry entries
- **Output format:** audit report — which registry entries fire on modern HF models, which are dead, which need clarification
- **Code path:** TBD — extension of `tools/analyze_explain.py`
- **Notes:** Distinguishes living documentation from stale entries by grounding the registry in real-model behavior.

### 6. Non-strict tracer soundness validation
- **Status:** ACTIVE as of 2026-04-22 (named consumer; multi-input infra gated on consumer signal)
- **Audience:** PyTorch export / non-strict tracer team. **Named stakeholder:** Animesh Jain (anijain) — wants comparative data on whether traced graphs generalize across input distributions, in the context of the active non-strict-tracer vs Dynamo debate.
- **Protocol:** Trace each model on input A; run the resulting graph on inputs B, C, D; compare per-input outputs against eager (or against each other). Soundness violations show up as input-dependent divergence in a graph that was assumed input-independent.
- **Signal consumed:** traced graph + per-model multi-input set + per-input outputs.
- **Output format:** per-model soundness report — which inputs pass, which diverge, divergence pattern, repro for each divergent input.
- **Code path:** TBD — `tools/check_tracer_soundness.py`. Gated on multi-input infrastructure (today corpus is one-input-per-model).
- **Notes:** Multi-input MVP scoping is intentionally deferred until we have a concrete soundness signal Animesh wants to see — don't pre-build the wrong thing.

---

## Stable signals consumers can rely on

These are the primary outputs any consumer can build against. Treat them as the corpus's API surface — changes need a deprecation path.

- **Sweep results JSON** — per-run JSON; markdown roll-ups at `results/pt2.X.md`. Per model: status, mode, error info, wall time, GPU memory.
- **Correctness results JSON** — `correctness/correctness_results.json`. Per model: status, max_diff, severity_ratio, tolerance, compared_fields.
- **Reproducers** — `python3 tools/reproduce.py <ModelName>` reproduces any sweep finding with one command.
- **Pattern classification** — `tools/analyze_explain.py` outputs per-model break patterns derived from TORCH_LOGS.
- **Input set** — *forthcoming.* Versioned per-model JSON describing the canonical input plus optional variations. Today the corpus has one input per model implicitly; multi-input infra (required by use case #6, useful for #3, #4, and for `optimum-benchmark` parity) is in scoping. See [charter delta 2026-04-22](design/charter-delta-2026-04-22.md).

If a new consumer needs a new stable export, add it both to the relevant code path and to this list.

---

## Adding a new use case

1. **Catalog first.** Add an entry above with audience, signal, output format, status. This forces clarity before code.
2. **Reuse existing exports** when possible. Most use cases differ in filtering/transformation, not data source.
3. **Build the consumer in `tools/`** if it's one script; promote to a subdirectory only when it outgrows that. Premature directory churn breaks cron jobs and integrations.
4. **Update this doc** as status moves through proposed → prototype → live.

---

## Explicitly out of scope (today)

Considered but not on the roadmap:
- Latency / memory regression sweeps
- Kernel-quality scoring
- External research-team self-service (would require platform-mode investment — see [open loops](design/open-loops.md))
