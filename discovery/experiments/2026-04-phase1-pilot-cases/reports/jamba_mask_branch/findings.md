# Jamba — Findings (Pilot 2, N=6 trials)

**Source:** Migrated from Drive doc [Pilot 2 Jamba — Findings](https://docs.google.com/document/d/1I3XRLdQfomDFOo7jLyHNB9gx6L-f-_rT4xmM69amRvw/edit) (2026-04-23).
**Case:** `jamba_mask_branch`
**Model:** `JambaForCausalLM` (HuggingFace `transformers`)
**Skill under test:** `debug-graph-breaks` (Arsh Zahed)
**Setup:** 6 trials = 3 with_skill + 3 no_skill (bare Claude, no plugin, slash commands disabled)
**Run harness:** `/tmp/pilot2-jamba/run_trial.sh`
**Trial artifacts:** `/tmp/pilot2-jamba/trials/{with,no}_skill_{1,2,3}/`

> **Note:** This is a **Phase 1 pilot** report, written before the discovery harness existed. Methodology and fingerprint axes here predate the per-case-analysis SKILL — see `2026-04-cross-case-skill-discovery/plan.md` for the current methodology.

---

## The Jamba break

Single data-dependent branch in `_update_mamba_mask` (transformers/models/jamba/modeling_jamba.py:754):

```python
if (past_key_values is not None and past_key_values.has_previous_state()) or (
    attention_mask is not None and torch.all(attention_mask == 1)
):
    mamba_mask = None
```

`torch.all(attention_mask == 1)` evaluated inside a Python `if` is data-dependent — Dynamo can't statically resolve it, produces 3 graph breaks (one per Python boolean operator that tries to materialize the tensor).

The optimization the branch encodes: when attention_mask is all 1s, masking is a no-op multiply downstream in `JambaMambaMixer` (`hidden_states = hidden_states * attention_mask.unsqueeze(1)`), so set `mamba_mask = None` to skip the multiply entirely.

## Outcome (all 6 trials)

All 6 trials, both arms, converged on the same fix: **delete the data-dep branch entirely**, return `attention_mask` unconditionally (except the cached-forward early-exit, which is structure-dependent and Dynamo can trace through).

```python
def _update_mamba_mask(self, attention_mask, past_key_values):
    if past_key_values is not None and past_key_values.has_previous_state():
        return None
    return attention_mask
```

This is mathematically equivalent in all cases (no-op multiply by all-ones). It loses the perf optimization (skip the multiply when mask is all-ones) but preserves correctness end-to-end. All 6 trials passed validation: graph_break_count 3→0, max_diff vs eager = 1.19e-07.

## Variance scan (N=6)

| trial | turns | dur(s) | cost | reads | bash | edits | mentioned cond? | fullgraph recheck? |
|---|---|---|---|---|---|---|---|---|
| with_skill_1 | 13 | 128 | $0.43 | 6 | 5 | 1 | n | n |
| with_skill_2 | 9 | 181 | $0.38 | 2 | 5 | 1 | n | Y |
| with_skill_3 | 13 | 135 | $0.43 | 6 | 5 | 1 | n | n |
| no_skill_1 | 16 | 220 | $0.51 | 7 | 7 | 1 | n | Y |
| no_skill_2 | 14 | 146 | $0.34 | 5 | 7 | 1 | n | n |
| no_skill_3 | 14 | 132 | $0.28 | 6 | 6 | 1 | n | n |

**with_skill mean:** 11.7 turns, $0.41
**no_skill mean:** 14.7 turns, $0.38

## What survives N=6

1. **Identical fix-shape across all trials.** Single-edit deletion in 6/6 trials, both arms. Zero exceptions.
2. **Zero `torch.cond` mention in any trial, either arm.** The skill explicitly lists `torch.cond` as the textbook fix for data-dependent branches; with-skill agents had it in their prompt context but did not surface it in reasoning. Bare Claude likewise didn't reach for it.
3. **Skill saves turns on average (~20%).** with_skill arm averages 11.7 turns; no_skill arm averages 14.7. Same direction across all 3 paired trials, modest magnitude.
4. **Same diagnostic strategy in both arms:** Detect → trace consumer → identify safe simplification → fix → verify. The skill did not introduce a new diagnostic *shape*; it pruned a shape Claude already had.
5. **Same fix-space breadth in both arms (= 1 fix).** Neither arm enumerated alternatives. The skill did not unlock options Claude lacked, but also did not enforce option-enumeration.

## What was a single-trial artifact (retracted)

After analyzing only the with_skill_1 / no_skill_1 pair, I claimed *"the skill reduced verification paranoia"* (no_skill_1 did a separate fullgraph recheck; with_skill_1 didn't). Looking at all 6 trials, that claim doesn't generalize: 1 with-skill trial (with_skill_2) and 1 no-skill trial (no_skill_1) did fullgraph rechecks. The other 4 didn't. Verification depth is uncorrelated with the skill in this dataset.

## Provisional implications for the mental model (N=6, one model, one break-shape)

Caveats first: this is a single model, with a single break shape (data-dep `if` that has a provably-equivalent deletion fix). The findings probably generalize to *similar* break-shapes; whether they generalize to break-shapes where deletion is unsafe (Pilot 3 Dbrx target) is open.

With those caveats:

- **The skill in this case acted as a speed boost, not a capability boost.** With-skill agents reached the same fix faster and with less code reading. The destination didn't change.
- **The skill did not broaden the fix-space.** Both arms produced one fix. The skill's fix vocabulary (torch.cond, torch.where, torch._check, torch.compiler.disable, restructure) did not surface in agent reasoning even though the prompt context contained it. *Open question: is this because the deletion fix was so obviously safe that alternative-enumeration was unnecessary, or because the skill doesn't enforce enumeration as a step?* Pilot 3 Dbrx (where deletion isn't safe) will test this.
- **Bare Claude is a competent baseline for this break-shape.** Without any skill, Claude traces the consumer, builds a safety proof, and chooses the same fix. The skill's incremental value here is efficiency, not correctness.

## Open questions for the mental model

- **Does the skill's value scale with break-complexity?** Jamba is a 1-line break with a provably-safe deletion fix. If the skill's effect is "speed up the obvious," its value should grow when the answer is *not* obvious. Pilot 3 Dbrx will probe this.
- **What conditions make agents enumerate alternatives vs single-fix?** The skill lists alternatives but did not trigger enumeration on Jamba. Hypothesis: agents enumerate only when their first candidate fails verification. Untested.
- **What does bare Claude *not* know that the skill teaches?** This dataset doesn't surface a capability gap. Either the dataset is too easy, or the skill's value is genuinely "shape-of-workflow + speed" rather than "new techniques."

## Method note

The mental-model framing for this work treats **what the skill changes about agent reasoning** as the primary signal — not pass/fail or leaderboard rank. We are NOT comparing skill-A to skill-B (Olga's SkillsWatch shape). We compare with-skill to no-skill and read the trace.
