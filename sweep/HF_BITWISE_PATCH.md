# HF Transformers Bitwise-Equivalence Patch — NON-STANDARD sweep environment

⚠️ **This patch makes the sweep run against a *patched* HuggingFace Transformers, not the stock release. Numbers produced with it active are NOT comparable to standard-corpus numbers.**

## What / why

Landing the two bitwise-equivalence fixes upstream in HuggingFace will take a while (filed as corpus issues **#129** SDPA causal-mask and **#130** audio LayerDrop RNG, for hand-off to a HF maintainer). Until then, `sweep/hf_bitwise_patch.py` applies the same two fixes **at sweep time, at runtime, version-agnostically** — no edits to the installed `transformers` source, works against any installed version (verified the buggy code paths are unchanged through transformers **5.9.0**, the current latest).

## How to enable

OFF by default. Enable for a run by exporting the env var before launching:

```bash
export CORPUS_HF_BITWISE_PATCH=1
python tools/run_experiment.py sweep ...   # workers inherit the env via os.environ.copy()
```

No changes to the orchestrator/run_sweep are needed — the orchestrator already spawns workers with `os.environ.copy()` (same mechanism as `SWEEP_USE_KERNEL_RESOLVER`).

## What it changes

| | Mechanism | Patch |
|---|---|---|
| **M1** (#129) | `masking_utils._ignore_causal_mask_sdpa` + sibling `_ignore_bidirectional_mask_sdpa` skip the SDPA `is_causal` shortcut under `torch.compile` (because `is_tracing()` is True), diverging from eager | Wrap both functions so under torch.compile-but-not-export they behave as if not tracing → take the eager shortcut. Does **not** reimplement the (version-varying) body — temporarily neutralizes the module-level `is_tracing` for the duration of the original call only. |
| **M2** (#130) | Train-mode audio encoders draw `torch.rand([])` per layer for LayerDrop; dynamo RNG advances out of step with eager → different layers dropped (O(1) divergence) | In train mode, zero every LayerDrop / dropout / SpecAugment probability so **neither** path drops/masks stochastically. |

### Important asymmetry (M2)

The *production* #130 fix is compile-only (it preserves eager's stochastic LayerDrop). So even after #130 lands, an **eager-vs-compile** bitwise comparison of a stochastic-LayerDrop model is **not** bitwise-equal unless eager is also made deterministic. The symmetric neutralization here is what lets the *sweep* measure post-fix equivalence; it is intentionally stronger than the upstream patch. (M1 has no such asymmetry — post-fix, eager and compile genuinely match.)

## Self-identification

Every result row produced with the patch active is stamped:
- `hf_bitwise_patched: true`
- `hf_bitwise_patch_note: "NON-STANDARD transformers: ..."`
- `hf_bitwise_patch_detail: {configs_zeroed, dropout_modules_zeroed}`

Standard (OFF) runs carry none of these fields.

## Validation (transformers 5.9.0, torch 2.13 nightly cu126)

End-to-end through `sweep/worker.py` with the NGB baseline compile-kwargs (`backend=eager, fullgraph=False`):

| Model / mode | OFF (stock) | ON (patched) |
|---|---|---|
| BartModel / eval (M1) | bitwise=False, max_diff=1.67e-6 | **bitwise=True, max_diff=0.0** |
| HubertModel / train (M1+M2) | divergence, bitwise=False, max_diff=5.02 | **match, bitwise=True, max_diff=0.0** |

OFF runs are byte-identical to standard behavior (no stamp; HubertModel divergence 5.02 matches the NGB baseline's 4.96).
