# Sweep Workflow Skill

Accumulated knowledge for running, analyzing, and maintaining the model graph break sweep.
Living document — update after every sweep cycle.

## When to use this skill

Read this skill BEFORE doing any of the following — no exceptions:

- About to launch ANY sweep — full sweep, cohort sweep, experimental sweep, ad-hoc batch model run. Anything that invokes `tools/run_experiment.py sweep` or otherwise runs models in batch via `sweep/worker.py`. Size doesn't matter — overnight cohorts and 30-min experiments both qualify, because the failure modes (forgetting the watchdog, picking wrong worker count, skipping baseline-compat check) are identical.
- Auditing or reproducing a prior sweep's results.
- Adding new models to the corpus.
- Investigating why a sweep produced unexpected status counts.

Do NOT skim. Read at minimum:

- §1 Default Sweep Scope (so you state scope explicitly)
- §2 Proven Baseline Settings (so you don't drift)
- §4 Sweep Execution Strategy — **including the Watchdog subsection** (mandatory pre-flight install)
- §7 Known Pitfalls (so you don't repeat them)

If your work touches the sweep harness code itself (`sweep/worker.py` etc.), `skills/test-sweep-changes/SKILL.md` is a separate, complementary trigger — both apply.

**Failure mode this trigger exists to prevent:** 2026-05-06 nested-gb correctness launch. Otter went straight from "Peng approved the cohort run" to drafting the launch command, never read this skill, never installed the canonical `sweep/sweep_watchdog.py`, and rolled a custom myclaw-cron one-shot wake instead. Peng caught it. Cost: ~20 min and a sharp prompt.

---

## 1. Default Sweep Scope

**Default (no special request needed):**
- HuggingFace Transformers (all 3 variants: base, ForCausalLM, ForConditionalGeneration)
- HuggingFace Diffusers
- Custom models
- Static shapes only
- Both eval and train modes

**Only include if Peng explicitly requests:**
- TIMM models
- Dynamic shapes

**Before every sweep, state the scope explicitly:**
> "Running N models: X HF (Y base + Z ForCausalLM + W ForCondGen) + A Diffusers + B Custom.
> Static shapes, eval+train = M work items. Estimated ~T minutes."

Never just say "sweeping all."

## 2. Proven Baseline Settings

These settings have been validated across pt2.8, pt2.9, pt2.10, pt2.11 sweeps:

| Setting | Value | Rationale |
|---------|-------|-----------|
| Workers | 4 | Proven stable; higher counts cause GPU memory contention |
| Timeout | 180s | Covers 95%+ of models; shorter causes false timeouts |
| Large timeout | 600s | For models in large_models.json registry |
| Device | cuda | Standard |

**DO NOT change these without:**
1. Running the testing workflow (Section 3) first
2. Comparing error rates against baseline
3. Getting Peng's approval

## 3. Script Change Testing Workflow

When making changes to sweep scripts (run_sweep.py, worker.py, models.py):

1. **Small test first:** Run on 5-10 known models (mix of full_graph, graph_break, eager_error) with both eval and train modes
2. **Compare results:** Same models should produce same statuses as the last known-good sweep
3. **Check error rates:** Any new errors or timeouts = regression, investigate before proceeding
4. **Scaling test:** If changing workers/timeout/parallelism, run on ~50 models and compare error rates against 4-worker baseline
5. **Only then:** Run the full sweep

For parameter changes (workers, timeouts):
- Always compare against the 4-worker/180s baseline
- If error rate increases > 1%, revert the change
- Document what you tested and the results

## 4. Sweep Execution Strategy

### Pre-flight
- [ ] Enumerate scope and state it explicitly
- [ ] Confirm settings match baseline (4 workers, 180s/600s)
- [ ] Check large_models.json is clean (no spurious entries)
- [ ] Verify torch/transformers versions
- [ ] **If your cohort came from a different sweep**, use `--models-from <source.json> --filter '<expr>'` (NOT `--models <flat.json>`). The launcher reads the source's torch/transformers/diffusers versions and refuses launch on mismatch — protects against the "I derived from torch-nightly-cu126 baseline but launched on torch211" failure mode (cautionary tale: 2026-05-06, this caused 25 spurious "regressions" because transformers 5.6.2 → 5.5.3 doesn't have the newer model `*Config` classes). Override only with `--allow-version-mismatch` and only when you mean it.
- [ ] **If using --models <flat.json>** (no provenance, e.g. a hand-built cohort file), explicitly identify the venv + version stack the cohort was derived from, and verify your launch flags match. The harness emits a `WARNING: --models was a flat list; cannot detect version mismatches` line — read it and act on it. Do NOT rationalize past it.
- [ ] **Run the sample-sweep gate** (see "Sample-sweep gate — mandatory before any full launch" below). Skipping requires Peng's written approval.
- [ ] **Install the watchdog cron** (see "Watchdog — mandatory for any non-trivial sweep" below)
- [ ] **If you are bumping a model library version, installing a new library, or adding new corpus sources** — read `skills/sweep_sanity_check.md` (Cohort expansion runs section) before launching. This is a cohort-expansion run, not a regular sweep. The triage step is mandatory output: the run isn't done when the sweep finishes — it's done when known_errors.json + skip_models.json are updated and a re-run sample shows 0 untriaged errors. **Pre-existing models that regress under the new library version are NOT triage-able — they are real bugs and must be fixed or filed.**

### Canonical multi-stage launcher: `tools/derive_sweep_commands.py`

When the experiment has a single canonical config (e.g. `experiments/configs/<name>.json` with `settings.python_bin` + `settings.modellib_pins`), use derive_sweep_commands instead of hand-typing gate/sample/full bash:

```bash
tools/derive_sweep_commands.py <config> --stage gate --run     # 5 models, ~5 min
tools/derive_sweep_commands.py <config> --stage sample --run   # 20 models, ~15 min
tools/derive_sweep_commands.py <config> --stage full --run     # entire cohort
```

Mechanical guarantees:
- Every stage uses the same flags as the full launch (only sub-sample size + output dir vary)
- Recursive sha256 validation on inner cohort `from` blocks before each stage
- Pinned interpreter + pinned modellibs from the source config (no PATH-based `python3` ambiguity)
- `gate` and `sample` success recorded in `/tmp/derive_sweep_state/<name>-<sha8>.json`
- `--stage full --run` REFUSED unless gate AND sample have passed for current source-config sha (override: `--allow-skip-gate`, logged loud)

This is the post-2026-05-07 canonical launch path for any experiment that fits the gate→sample→full pattern. The hand-rolled "Sample-sweep gate" instructions below remain valid for ad-hoc launches without a canonical config (e.g. one-off `tools/run_experiment.py sweep --compile-kwargs ...` invocations).

### Sample-sweep gate — mandatory before any full launch

Before any full sweep, run a sample sweep on **20 random models from the planned full cohort** with **identical flags** to the planned full launch (same venv, modellibs, workers, timeouts, compile-kwargs, dynamo-config). Suffix the sample's `--run-name` with `-sample` and write the sampled cohort to `/tmp/<run_name>-sample.json`.

When the sample completes (~10-15 min for 20 models × 2 modes at 2 workers), apply `skills/sweep_sanity_check.md` in **Pre-launch sample mode** against the sample's output. **Any STRICT invariant failure → HALT the full launch and investigate.**

If sample finds cohort contamination, the cohort generator is broken — fix the generator, regenerate the cohort, re-sample. Do NOT silently exclude individual contaminated models and proceed; that lets the generator bug ship.

If sample passes → launch full sweep.

**Cautionary tale (2026-05-06):** NGB correctness rerun launched at 21:33 ET with no sample gate. ~2h later the post-completion review found 22 create_error + 4 eager_error: 6 cohort-contamination rows (models removed in the target transformers version), 14 custom-model loader regressions (pre-existing models broke at create), 2 pre-existing diffusers bug, 4 input-gen failures. A 20-random sample on the actual cohort + actual flags would have hit at least one contaminated model and aborted the full launch in ~15 min instead of 3 hours.

A separate, earlier 6-fixed-model "smoke" (replaced by this gate) caught version-stack issues but missed all of the above because the fixed picks didn't overlap the contamination. Random 20-from-cohort is the right design.

### Execution order
1. **Run eval mode first** — 50% of work items, gives early signal
2. **Check eval results** — look at error rates, any surprises?
3. **If errors > 2%**, investigate and fix before proceeding
4. **Run train mode** — remaining 50%
5. **Merge and analyze**

### Watchdog — mandatory for any non-trivial sweep

Any sweep that will outlive your interactive session (> ~30 min wall-clock, anything launched via `nohup`, any overnight / cohort / experimental run) MUST have `sweep/sweep_watchdog.py` running as a recurring system cron BEFORE you close out of the launching session. The watchdog reports `{completed}/{total}` progress, phase changes, stalls, and DEAD events to GChat — without it, a silent crash mid-run gets discovered hours late.

**Canonical install (do this in the SAME session as `nohup ... &`):**

```bash
# Append to crontab; replace OUT_DIR with your sweep's --run-name dir.
( crontab -l ; echo "*/10 * * * * /home/USER/envs/torch211/bin/python /home/USER/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog.py /home/USER/projects/oss-model-graph-break-corpus/sweep_results/experiments/OUT_DIR/ --interval-min 10 --post-to spaces/AAQANraxXE4 >> /tmp/sweep-watchdog.log 2>&1" ) | crontab -
```

Then sanity-run it ONCE manually to verify it can read sweep_state.json + posts to GChat — don't wait for the first cron tick to discover a typo.

**What the watchdog is NOT:** auto-restart. When it reports DEAD, a human (or wake-fired agent) investigates and decides whether to `--resume`. Silent auto-restart hides root causes (spring 2026 source-build venv parity issues would have looped invisibly).

**Don't roll your own.** If you find yourself drafting a `myclaw-cron` one-shot wake or a custom progress-checker for a sweep launch, stop — `sweep/sweep_watchdog.py` already does that work, talks the right state-file schema, and tracks per-phase progress. **Common failure mode:** reading the file-header comment "NOT auto-restart" and dismissing the script without reading the body. Don't. (Cautionary tale: 2026-05-06 nested-gb correctness launch — Otter spent 20 min building a custom myclaw-cron wake before Peng pointed out the existing watchdog. See HANDOFF.md.)

User-facing version: [docs/running-sweeps.md § Monitoring long sweeps](../docs/running-sweeps.md#monitoring-long-sweeps).

### Post-sweep
- [ ] Error/timeout rate should be < 2% (in-scope models)
- [ ] Compare status distribution against previous sweep
- [ ] If errors exist, re-run those specific models with baseline settings before analyzing
- [ ] Clean up errors FIRST, then do analysis — never analyze dirty data
- [ ] Remove the watchdog crontab line (it self-silences but clutters `crontab -l`)

## 5. Large Model Registry Management

### Current problem
The auto-retry system defines "large" as "timed out at regular timeout." This is relative — if timeout=90s, a 100s model becomes "large." That's wrong.

### Proposed criteria (absolute)
A model is "large" if its wall_time_s exceeds **120 seconds** at the baseline settings (4 workers, 180s timeout). This is an absolute threshold independent of timeout configuration.

Tiers:
- **regular**: wall_time_s ≤ 120s — uses standard 180s timeout
- **large**: 120s < wall_time_s ≤ 480s — uses 600s timeout
- **very_large**: wall_time_s > 480s or never completes at 600s — needs investigation

### Registry update workflow
1. Auto-retry may discover new large models — this is fine for discovery
2. But **do not auto-commit** changes to large_models.json
3. After each sweep, review any new entries:
   - Is the wall_time_s genuinely > 120s at baseline settings?
   - Or was it caused by changed settings (fewer workers, lower timeout)?
4. Peng reviews new large model entries before committing

### When adding new models to the corpus
- Run the new model individually first with baseline settings
- Record wall_time_s
- If > 120s, add to large_models.json with actual wall_time_s
- Document why it's slow (large architecture, complex attention, etc.)

## 6. Adding New Models to the Corpus

When adding new models (new HF release, custom models, etc.):

1. **Test individually first** with baseline settings
2. **Record:** name, source, wall_time_s, status in both eval and train
3. **If large (>120s):** add to large_models.json with wall_time_s and discovery date
4. **If eager_error or create_error:** investigate — is it a real model issue or env issue?
5. **Update model count** in sweep scope documentation
6. **Run a comparison sweep** to establish baseline for the new models

## 7. Known Pitfalls

| Pitfall | What happens | Prevention |
|---------|-------------|------------|
| Increasing workers beyond 4 | GPU memory contention → worker_error, zombie | Stick to 4 unless tested |
| Reducing timeout below 180s | False timeouts on 90-180s models | Stick to 180s baseline |
| Running `--source all` | Includes TIMM (1284 models), wastes hours | Use `--source hf+diffusers` or explicit model list |
| Analyzing before cleaning errors | Misleading percentages, wrong conclusions | Always triage errors first |
| Changing large_models.json criteria | Corrupts registry, wrong models get long timeout | Use absolute 120s threshold |
| Running both modes at once | Doubles wait time before first signal | Run eval first, check, then train |
| Incomplete model specs in re-run | Missing hf_config/variant causes config resolution to fail silently | Always build re-run specs from `enumerate_all()`, never from checkpoint data |

## 8. Post-Sweep Due Diligence

After every sweep completes, do this BEFORE reporting results to Peng:

### Step 0: Run the sanity check (Post-completion mode)
- [ ] Apply `skills/sweep_sanity_check.md` in **Post-completion mode** to the full output dir
- [ ] Classify every FAIL as `accepted` (record reason in plan.md or the run's notes) or `blocking`
- [ ] **Untriaged errors (INV-G1 fail) → HALT.** Either triage them (file a fix, add a known_errors entry, or skip-list with reason) BEFORE proceeding to Step 1, or open a cohort-expansion sub-task to process them as a batch
- [ ] Do NOT analyze, file issues, or publish results while blocking fails exist

### Step 1: Error/timeout triage
- [ ] Count errors + timeouts. If > 2% of in-scope models, investigate before proceeding.
- [ ] Compare error list against previous sweep — any NEW errors?
- [ ] For new errors: is it a real model issue or a sweep config issue?
- [ ] Re-run any config-caused errors with baseline settings

### Step 2: Diff against previous sweep
- [ ] Load both current and previous sweep results
- [ ] Find models that **changed status** between sweeps:
  - full_graph → graph_break (regression — new graph break, high priority)
  - graph_break → full_graph (fix — someone fixed it upstream)
  - any status → error/timeout (sweep issue, not real signal)
- [ ] Document what changed and hypothesize why

### Step 3: New model check
- [ ] Any models in current sweep that weren't in previous? (new HF releases, etc.)
- [ ] Record their statuses as new baselines
- [ ] Check if any new models are large (>120s) — update registry if so

### Step 4: Report
Report should include:
1. **Scope:** what was swept (model counts, versions, settings)
2. **Overall numbers:** full_graph %, graph_break %, error %
3. **What changed:** regressions, fixes, new models since last sweep
4. **Action items:** new issues to file, things to investigate

Don't just dump numbers. Lead with what's **different** from last time.

### Step 5: Update documentation and artifacts
After the sweep is validated and results are final:
- [ ] Update `results/v2.X.md` with current model counts, status breakdown, and changes
- [ ] Update `corpus/summary.md` with refreshed stats
- [ ] Update `README.md` model counts and results-at-a-glance section
- [ ] Update `design/error-models.md` if error models changed
- [ ] Regenerate `docs/index.html` dashboard: `python3 tools/generate_index.py`
- [ ] Commit all doc updates together
- [ ] Verify actual git state (`git log origin/main..HEAD`) before reporting push counts

This is NOT optional — stale docs erode trust in the corpus.

### Step 6: Reflect on error signals
Before closing a sweep cycle, review the error patterns collectively:
- [ ] What do the errors tell us as a group? Are there common root causes?
- [ ] Are there consolidation opportunities in the model-specific handlers? (e.g., 12 similar blocks → 1 table)
- [ ] Did we add any code that smells like brute force / special-casing? Refactor it.
- [ ] Are there hygiene cleanups needed? (deduplication, stale entries, dead code)
- [ ] What would prevent these errors in the next sweep? (better defaults, smarter input generation)

**Anti-pattern:** fixing models one-by-one without stopping to see the bigger picture. Speed without reflection turns into brute force. Long sessions without checkpoints lead to rabbit holes.

## 9. Interpreting Results (Reference)

### Status meanings
- **full_graph**: torch.compile(fullgraph=True) succeeds — no graph breaks
- **graph_break**: compilation succeeds but with graph breaks — these are the signal
- **eager_error**: model fails even without compile (broken model config, missing deps)
- **create_error**: model can't be instantiated (bad config, missing weights)
- **timeout**: didn't complete in time — re-run or classify as large
- **worker_error**: worker process crashed — usually GPU OOM from too many workers
- **zombie**: worker didn't respond — GPU contention or system issue

### Acceptable error rates
- eager_error + create_error: ~5% is normal (genuinely broken model configs)
- timeout + worker_error + zombie: should be < 1% with baseline settings
- If timeout+worker+zombie > 2%: settings are wrong, fix before analyzing

### Analysis order
1. Overall: full_graph % and graph_break %
2. By source: HF vs Diffusers vs Custom
3. By HF variant: base vs ForCausalLM vs ForConditionalGeneration
4. By mode: eval vs train differences
5. Graph break model list: group by architecture family for root cause clustering

## 10. Post-Mortem: Improving This Skill

After every sweep cycle that includes feedback from Peng, do this:

### Step 1: Collect feedback
- Review all corrections, questions, and concerns Peng raised during the sweep
- Note anything that surprised you or that you got wrong

### Step 2: Gap analysis
For each piece of feedback, ask:
- **Is this already covered in this skill document?** If yes, why didn't I follow it? (execution gap — might need stronger language or checklist enforcement)
- **Is this NOT covered?** Then it's a knowledge gap — add it.
- **Does this contradict something in the doc?** Update the doc.

### Step 3: Propose changes
- Draft specific additions/edits to this skill document
- Include the lesson source: what happened, what the feedback was, what we learned
- Add to the Revision Log (below)

### Step 4: Notify Peng
- **Always tell Peng what you learned and what you're changing** before committing
- Format: "Lesson from this sweep: [what happened]. Updating sweep skill: [specific change]."
- Peng wants visibility into skill evolution — never silently update

### Step 5: Commit
- After Peng has seen the proposed changes, commit the updated skill document
- The updated version is automatically available for next sweep

### Why this matters
Without this step, feedback only lives in Otter's session memory (which expires) or in scattered memory files (which may not get loaded at the right time). By writing lessons directly into the skill document, they become part of the workflow itself — automatically injected into context every time a sweep starts.

---

## Revision Log

| Date | What changed | Lesson source |
|------|-------------|---------------|
| 2026-04-14 | Initial draft | v2.10_full sweep: scope ambiguity (TIMM included by mistake), worker/timeout regression (16 workers + 90s timeout caused false errors), serial polling blocked conversation |
| | Added post-sweep due diligence | Peng feedback: "you always ask me what to do after sweep — minimum is compare against previous" |
| | Added post-mortem skill improvement | Peng feedback: "need a skill for improving the skill system — capture contextual learning into the file" |
| 2026-04-14 | Added pitfall: incomplete re-run specs | Re-run with only name+source caused 30+ false create_errors. Config name derivation fails for ForCausalLM variants without hf_config field |
| | Added Tier 2 autonomy for merging/re-running | Tiger autonomy tiers doc: merge + re-run after sweep is obvious next step, don't ask |
| 2026-05-06 | Added smoke pre-flight subsection (§4) + flat-list provenance check | NGB correctness sweeps #1 and #2 same day burned 2h41m + 11min on wrong version stack (torch211 + transformers 5.5.3) when explain pass had run on torch-nightly + transformers 5.6.2. Smoke including BartModel + BlenderbotModel + BambaModel would have aborted both before launch. Also includes the "provenance vs status" pitfall in criterion design (initial criterion-5 confused cohort provenance with result status). |
| 2026-05-07 | Replaced 6-fixed smoke with 20-random sample-sweep gate; added APPLY-D cohort-expansion trigger; added §8 Step 0 sanity-check gate. New skill `skills/sweep_sanity_check.md` v2.1 holds the invariants (families A-G) + four apply contexts. Library-bump rule encoded: pre-existing models that regress are real bugs, not triage-able. *(Note: sanity-check skill v3 simplified the four apply contexts to three modes — Pre-launch sample / Mid-sweep peek / Post-completion + Cohort expansion runs. Cross-refs in this skill updated 2026-05-08.)* | NGB verify 2026-05-06: 6-fixed smoke passed but full sweep surfaced 22 create_error + 4 eager_error from cohort contamination + custom-model loader regression that the fixed picks didn't overlap. Random 20-from-cohort is the right design. Peng's directive: "infra solid > moving fast" + skill-form-not-script (different sweep types have different invariants — judgment over enforcement). |
| 2026-05-08 | Updated APPLY-A/C/D cross-references to v3 mode names (Pre-launch sample / Post-completion / Cohort expansion runs section); 4 stale refs fixed in §4 Pre-flight, §4 Sample-sweep gate, §8 Step 0. | Adversary review case 2026-05-07-190947-doc-vs-impl, gap #1 — `skills/sweep_sanity_check.md` v3 deleted the four named apply contexts but cross-refs in `skills/sweep.md` weren't updated atomically. |

---

*This document is loaded at the start of every sweep-related task.*
*To invoke: read `docs/sweep-skill.md` before starting any sweep work.*
