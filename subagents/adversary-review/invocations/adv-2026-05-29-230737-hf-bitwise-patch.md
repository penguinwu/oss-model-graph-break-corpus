---
case_id: adv-2026-05-29-230737-hf-bitwise-patch
subagent: adversary-review
date_utc: 2026-05-29T23:13:38Z
trigger: validator-code
files: sweep/hf_bitwise_patch.py, sweep/worker.py
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: cc65cf1aed4104649426e9a69c0f3ff060e9648bb3894e91499fee4de6a047db
---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: medium
CONFIDENCE_RATIONALE: I read the full patch module, both worker.py edit regions, and the real HF source being patched, but cannot run the harness to confirm the dynamo-traceability and reentrancy behaviors empirically.

FILES_READ:
- sweep/hf_bitwise_patch.py
- sweep/worker.py (run_identify M2 region ~3630-3646; main() M1 region ~4473-4484)
- transformers/masking_utils.py
- transformers/utils/import_utils.py

GAPS_FOUND:
1. [SEVERITY: high] The M1 is_tracing swap is module-global and NOT reentrant: nested causal+bidirectional (encoder-decoder, e.g. Bart) could restore a stale neutralizer, leaving is_tracing permanently neutralized -> find_packed_sequence_indices skips its guard under eager too, silently.
2. [SEVERITY: high] Under backend=eager+fullgraph=False, is_compiling() may be False at the mask call -> M1 wrapper is a no-op, yet row still stamped hf_bitwise_patched=True (silent over-claiming / mis-attribution).
3. [SEVERITY: medium] Stamp written only in run_identify; create_error early return (line 3624) and other passes are patched-but-unstamped, violating "every patched row is self-identifying".
4. [SEVERITY: medium] M2 mutates model/config in place with no restore; cross-model leak if config cached/shared (mitigated by one-model-per-process).
5. [SEVERITY: low] _iter_configs may miss list/dict-nested sub-configs or falsely descend into to_dict objects (tokenizers/processors).

SUGGESTED_ADDITIONAL_TESTS:
1. reentrancy_restores_true_original -> after nested wrapped calls, mu.is_tracing is the genuine original (identity).
2. m1_actually_fired_on_bart -> counter > 0 AND bitwise == 0.0.
3. early_return_row_is_stamped -> create_error row still carries hf_bitwise_patched=True.
4. off_run_byte_identical -> OFF run has no hf_bitwise_* keys; originals unpatched.
5. m2_eval_is_noop -> eval mode leaves dropout/LayerDrop unchanged.

NOTES:
M1 does NOT blindly fake equality — after neutralizing is_tracing, the original still enforces query_length==1/kv==query, padding_mask.all(), local_attention_size gates. Genuine-fix claim holds CONDITIONAL on gap 2. Highest-leverage gaps (1,2) are about dynamo-state/module-global mutation, not exercised by happy-path validation. Adjacent: apply_global_patches idempotency is process-local; a masking_utils reimport would silently lose M1.
```

## My disposition

- gap 1 (reentrancy / stale-neutralizer restore) → **addressed**: capture `_TRUE_IS_TRACING` once at patch time + reentrancy depth guard (`_SWAP_DEPTH`); only the outermost wrapped call swaps, and it restores the genuine original, never a stale neutralizer.
- gap 2 (M1 silent no-op but stamped) → **addressed**: added `_M1_FIRE_COUNT` incremented on every neutralize-branch entry; surfaced as `hf_bitwise_patch_detail.m1_shortcut_fired`. Verified BartModel/eval ON → fired=3 + bitwise=0.0 (equivalence IS attributable to M1); create_error → fired=0.
- gap 3 (unstamped early-returns / other passes) → **addressed**: moved the marker + fire-count stamp to the `main()` chokepoint before `print(json.dumps(result))`, so every pass and every early-return row is stamped. Verified create_error row → hf_bitwise_patched=True.
- gap 4 (in-place mutation, no restore) → **addressed (precondition)**: documented one-model-per-worker-process invariant (orchestrator.spawn_worker) in apply_model_determinism docstring; no cross-model reuse within a process.
- gap 5 (_iter_configs list/dict-nested miss / false descend) → **deferred** (low sev): current attribute-traversal + get_text_config covers the validated audio + SpeechEncoderDecoder families; revisit if a multi-config model shows residual divergence in a patched sweep. OPEN-LOOP filed.
