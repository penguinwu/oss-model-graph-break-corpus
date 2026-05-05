# Weekly Sweep Brief — methodology rules

These are the rules that distinguish a defensible brief from a misleading one. Every rule was learned from a specific mistake in 2026-04 / 2026-05 weekly briefs. **Skip a rule = ship a wrong number.**

## Hard rules (mechanical guards exist; bypass requires Peng's written approval)

### R1 — Single source of truth: `tools/sweep_compare.py`

Every count in the brief must derive from `sweep_compare.py` output (markdown report, JSON output, or `--pattern` query). NO ad-hoc one-liner Python scripts that compute corpus-wide aggregates. (Reason: the 2026-05-04 hand-aggregated `_local_scalar_dense` "+41 regression" finding mixed cat 3 with cat 1/cat 4 and was wrong by sign.)

### R2 — Apple-to-apple only for "regression" / "improvement" claims

The only category where a delta means "Dynamo regressed" or "Dynamo improved" is **cat 3** (compile-success in BOTH baseline and current). Cat 1 / cat 4 deltas are EXPOSURE — patterns becoming newly observable, not Dynamo getting worse.

Use `python3 tools/sweep_compare.py --pattern "<substring>"` for any per-pattern question. The tool's output is segmented; copy the structure as-is into the brief, do not collapse to one number.

Forbidden phrasing: "X regressed by Y" or "X improved by Y" without naming the population. Required phrasing: "X regressed by Y on common compile-success pairs" or "X exposure +Y from newly compile-testable models."

### R3 — Attribution must be verified on at least 2 flipped models before claiming "Dynamo win"

For any model that flipped `graph_break → full_graph`, run the model with current harness + OLDER torch + baseline transformers. If it reproduces the original break, attribution is torch (Dynamo). If it compiles cleanly, attribution is harness or transformers — investigate further.

Verify the involved transformers source files are byte-identical between baseline and current. Same caller + same callee + same call path + only torch differs = torch attribution.

If attribution can't be verified before brief deadline, mark "### Attribution status: UNVERIFIED" and propose the test. Don't claim "Dynamo win" without verification.

### R4 — Umbrella-split policy applied before any issue claims

Before referring to a tracked issue as "improving" or "closeable", verify it's not an umbrella (multiple distinct root causes bundled). Apply the corpus CLAUDE.md "Umbrella-split policy" if so.

The "trace CALL" / "Encountered graph break when attempting to trace CALL" / "Failed to handle graph break gracefully" / "Cannot resume from graph break" patterns are bytecode-instruction wrappers, NOT root causes. Always classify by the underlying explanation/operator beneath.

### R5 — Search existing issues BEFORE filing new ones

Before filing any new dynamo / transformers issue, grep open + closed issues for the break-reason substring, the operator name, OR the source location. (Reason: 2026-05-04 split of #8 created 3 duplicates of #11, #23, #24 because I didn't search first.)

```bash
gh issue list --repo penguinwu/oss-model-graph-break-corpus --state all --search "<substring>" --limit 20
```

### R6 — Newly compile-testable = cat 1 + cat 4 successes combined

Per Peng (2026-05-04): any flip from error to eager-success/compile-success counts as a "new model" for brief purposes. Don't report cat 4 alone as "new models added" — it under-reports the actual newly-testable surface.

### R7 — Broadcast posts to feedback space go through the gated wrapper

For weekly briefs / daily standups / any broadcast announcement to `spaces/AAQABmB_3Is`, the signal-boost step uses `python3 ~/.myclaw/spaces/AAQANraxXE4/tools/post_to_feedback.py send "..."` so it picks up dedup + audit logging + weekend gates.

Direct `gchat send spaces/AAQABmB_3Is "..."` is fine for thread replies (someone asked a question, you're answering — match the conversational tone). The discipline check before any send is "is this internal sweep ops state?" If yes, wrong space. (Reason: 2026-05-03 watchdog status leak. Discipline-only — Peng confirmed normal interaction is approved.)

### R7.5 — Wrapper-only buckets are usually classifier gaps, not explain-pass bugs

When the umbrella-split bucketing puts entries into a `failed_graceful_only` / `cannot_resume_only` / similar catch-all bucket, the FIRST check is "does my classifier have a rule for the underlying reason?" — sample the actual `break_reasons` text from `explain_checkpoint.jsonl` and look for inner explanations the classifier missed.

Filing an issue against the explain pass (claiming serialization drops the inner reason) requires verifying that the inner reason is GENUINELY missing in the serialized text — not just unclassified by your script. (Reason: 2026-05-04 issue #123 was filed against the explain pass, then closed when verification showed the inner reasons WERE captured — my classifier just didn't have rules for `torch.nn.Parameter() constructor`, `torch.* op returned non-Tensor`, `missing tp_iter`, etc.)

### R8 — Verify the destination group's actual name before referring to it by name

`meta workplace.group details --group-id=<ID>` returns the canonical name. Always run it and use the output, NOT a name from memory or a USER.md group list (which can have similar-sounding groups with different IDs).

(Reason: 2026-05-04 signal boost called the brief's destination "PT2 Compile Q&A" — wrong; the actual group is "PyTorch Compiler AI-native Working Group" (`1251481819891072`). The PT2 Compile Q&A group exists separately at `1075192433118967`. Peng caught and corrected. The Workflow Step 7b now mechanically captures the name into a shell variable before composing the signal-boost.)

## Soft rules (judgment, no mechanical guard)

### S1 — Numbers in the headline must reconcile with body sections

The headline says "Net −124 GBs" — the cat 3 sub-totals in section 3 must add up to that. Run a self-check: `(sum of GB-improved deltas) + (sum of GB-regressed deltas) + (sum of full-flip eliminations) = headline net`. If they don't reconcile, fix one or the other.

### S2 — Forbidden vague language

- "Likely closeable" → either VERIFIED-closeable (with checked closure criterion) or NOT-closeable. No "likely."
- "Substantially advanced" → if the cat 3 delta is positive, just say "improved by N." If you can't quantify, you don't know.
- "Some models" / "a few models" → name N. If too many to name, give the number.

### S3 — Audience-adapted detail

The audience is PT2 dynamo team — technical, but not in our internal discussion threads. Forbidden references: bare issue numbers without 1-line description; internal tool names (`sweep_compare.py`, `post_to_feedback.py`) without enough context for an outsider to find them; internal jargon ("cat 3", "cat 4") used as load-bearing without first defining them in the headline framing.

### S4 — Actionable items must name the leverage

Each actionable bullet states "if you fix X, Y breaks across Z models clear." If you can't quantify leverage, the item shouldn't be in the actionable list — move it to a footnote or drop it.

## Self-check checklist (run before posting)

Walk through every item. Each must pass.

- [ ] Step 1: `tools/sweep_compare.py --check-only` returned exit code 0
- [ ] R1: every number in the brief comes from sweep_compare output (markdown, JSON, or `--pattern` query) — no ad-hoc scripts
- [ ] R2: every "regression" or "improvement" claim names cat 3 explicitly (or is qualified as "exposure" / "newly observable" for cat 1/cat 4)
- [ ] R3: attribution verified on ≥2 flipped models, OR explicitly marked UNVERIFIED
- [ ] R4: every umbrella issue mentioned has been split (or marked for split this cycle)
- [ ] R5: every new issue filed today was preceded by a `gh issue list --search` against existing
- [ ] R6: "newly compile-testable" includes cat 1 + cat 4, not just cat 4
- [ ] R7: signal-boost message goes through `post_to_feedback.py`, not raw `gchat send`
- [ ] R8: destination group name in the signal-boost matches `meta workplace.group details` output (verified, not from memory)
- [ ] S1: headline numbers reconcile with body sub-totals
- [ ] S2: no "likely" / "substantially" / "some" / "a few" in load-bearing claims
- [ ] S3: audience can read this without our internal context (no bare #N, no "cat 3" load-bearing without intro)
- [ ] S4: each actionable bullet names the leverage in numbers

If any item fails: fix it, then re-run the checklist (don't selectively re-check). The whole list must pass before Step 7 (post).

## Lessons baked in (changelog of why these rules exist)

- **R2** added 2026-05-04 — `_local_scalar_dense` "+41 regression" finding was wrong because hand-aggregated across cats. `tools/sweep_compare.py --pattern` added as the mechanical guard.
- **R3** added 2026-05-04 — initial brief claimed "Dynamo improvement" without testing; SmolVLM + LwDetrModel attribution test confirmed torch source.
- **R4** added 2026-05-04 — Issue #8 (DETR proxy CALL) had 8+ underlying root causes, not one. Same applied to #18, #102, #103.
- **R5** added 2026-05-04 — split of #8 created 3 duplicates of #11/#23/#24. Settings deny rule + grep workflow added.
- **R6** added 2026-05-04 — initial brief reported only cat 4 as "new"; missed Qwen3VL/Qwen3.5 vision encoders that flipped from eager_error.
- **R7** added 2026-05-04 — watchdog status leaked to user group via raw `gchat send --as-user`. Settings deny + this rule added.
- **R8** added 2026-05-04 — signal-boost called destination "PT2 Compile Q&A" (wrong); actual group is "PyTorch Compiler AI-native Working Group". Workflow Step 7b now captures name into shell variable mechanically.
- **S1, S2, S3, S4** consolidated 2026-05-04 from iterations during the 2026-05-03 brief composition.
