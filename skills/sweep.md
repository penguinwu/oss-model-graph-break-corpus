# Sweep Workflow Skill

Accumulated knowledge for running, analyzing, and maintaining the model graph break sweep.
Living document — update after every sweep cycle.

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

### Execution order
1. **Run eval mode first** — 50% of work items, gives early signal
2. **Check eval results** — look at error rates, any surprises?
3. **If errors > 2%**, investigate and fix before proceeding
4. **Run train mode** — remaining 50%
5. **Merge and analyze**

### Post-sweep
- [ ] Error/timeout rate should be < 2% (in-scope models)
- [ ] Compare status distribution against previous sweep
- [ ] If errors exist, re-run those specific models with baseline settings before analyzing
- [ ] Clean up errors FIRST, then do analysis — never analyze dirty data

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

---

*This document is loaded at the start of every sweep-related task.*
*To invoke: read `docs/sweep-skill.md` before starting any sweep work.*
