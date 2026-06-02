---
case_id: adv-2026-06-02-134721-reformer-m3-cached-dropout
subagent: adversary-review
date_utc: 2026-06-02T13:56:21Z
trigger: validator-code
files: sweep/hf_bitwise_patch.py, sweep/worker.py
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 82ac3f9a47aa1d351803d2f288d115604daa42ea0fe8ead79e76f842eda355e5
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: medium
CONFIDENCE_RATIONALE: Read the full patch module, the real Reformer source (LSH _hash_vectors, all cached self.dropout sites incl. line 2553 module case), worker call sites, and prior adversary review; cannot run the harness to confirm dynamo honors the runtime torch.randn rebind on the compiled path.
FILES_READ:
- sweep/hf_bitwise_patch.py
- transformers/models/reformer/modeling_reformer.py
- sweep/worker.py
- subagents/adversary-review/invocations/adv-2026-05-29-230737-hf-bitwise-patch.md
GAPS_FOUND:
1. [SEVERITY: high] M3 Python-level torch.randn rebind validated ONLY under backend=eager; whether the COMPILED path actually routes the rotation draw through det_randn is unverified (esp. inductor). If not, eager(seed42) vs compiled(dynamo RNG) diverge -> patch doesn't deliver equivalence (silent-fake risk). FIX: instrument det_randn with a counter; assert it fires on the compiled path; optionally test backend=inductor.
2. [SEVERITY: medium] orig() still runs conditional torch.manual_seed(self.hash_seed) (~L715) when config sets hash_seed; det_randn ignores global RNG for rotations but the manual_seed mutates GLOBAL RNG, leaking to downstream ops differently eager-vs-compile. FIX: test with hash_seed set; save/restore global RNG or neutralize hash_seed.
3. [SEVERITY: medium] cached-dropout float zeroing matches only attr literally named 'dropout'; would MISS a model caching a dropout prob under another name -> latent residual divergence; "cached_dropout_floats_zeroed=10" gives false completeness confidence. FIX: document name-specificity or broaden with guards; pin with a test.
4. [SEVERITY: low] torch.randn swap is process-global, not thread-safe; concurrent thread calling torch.randn during the window gets the seeded generator. FIX: document single-thread assumption.
SUGGESTED_ADDITIONAL_TESTS:
1. det_randn_fires_on_compiled_path -> counter>0 on compiled forward, bitwise 0.0.
2. reformer_inductor_bitwise -> backend=inductor bitwise 0.0.
3. reformer_with_hash_seed_set -> hash_seed non-None, bitwise 0.0 + global RNG unchanged.
4. float_dropout_skips_module_and_bool -> nn.Dropout module zeroed via .p branch, float zeroed via cached branch, bool untouched.
5. m2_eval_noop_for_reformer -> eval mode mutates nothing.
NOTES:
Line 2553 self.dropout = nn.Dropout(...) correctly skipped by float check (Module is not int/float). not-bool guard correct (isinstance(True,int) is True). Idempotency holds two ways. Re-seed-42-every-call degrades LSH hash quality but correct for equivalence measurement. Highest-leverage: gap 1 — M3 rests on the compiled path executing the rebind, validated only under backend=eager.
```

## My disposition

- gap 1 (M3 may no-op on compiled path / silent-fake) → **addressed**: added `_M3_FIRE_COUNT` + `m3_reformer_rotation_fired` in the stamp. Verified ReformerModel/train ON → fired=2 AND bitwise 0.0, vs OFF → divergence 4.62 — the OFF→ON contrast + nonzero count prove equivalence is attributable to M3. backend=inductor deferred: the corpus bitwise sweep uses backend=eager only (documented in M3 docstring SCOPE).
- gap 2 (hash_seed manual_seed global-RNG leak) → **documented limitation**: attempted save/restore of global RNG state but `torch.cuda.get_rng_state_all()` is NOT traceable under dynamo (raises 'source must be provided in options') — it broke the working path, reverted. config.hash_seed defaults to None so manual_seed never runs in our sweep; residual only possible if a config sets hash_seed (none in corpus). Documented in code.
- gap 3 (cached-dropout name-specific to 'dropout') → **documented limitation**: kept conservative (no name-substring scan to avoid zeroing unrelated floats); added explicit code comment that a future model caching a dropout prob under a different attr name would be missed; extend the list if it appears. OPEN-LOOP filed.
- gap 4 (torch.randn swap not thread-safe) → **documented**: single-thread / one-model-per-process invariant noted in code next to the swap.
