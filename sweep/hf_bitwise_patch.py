"""HF Transformers bitwise-equivalence patch (NON-STANDARD sweep environment).

⚠️  This module makes the sweep run against a *patched* HuggingFace Transformers,
    NOT the stock release. Results produced with this patch active are NOT
    comparable to standard-corpus numbers and every result row is stamped with
    `hf_bitwise_patched=True` so they are self-identifying.

Why this exists
---------------
Landing the two fixes upstream in HuggingFace will take a while (filed as corpus
issues #129 + #130 for hand-off). Until then, this applies the same two fixes at
sweep time, at runtime, **version-agnostically** (no edits to the installed
transformers source — works against any installed version, verified the buggy
paths are unchanged through transformers 5.9.0).

It is OFF by default. Enable it for a run by exporting:

    export CORPUS_HF_BITWISE_PATCH=1

The orchestrator spawns workers with `os.environ.copy()`, so setting the env var
before launching `tools/run_experiment.py sweep ...` propagates to every worker.

What it patches
---------------
M1 — SDPA causal-mask shortcut (issue #129):
    `transformers.masking_utils._ignore_causal_mask_sdpa` (and the sibling
    `_ignore_bidirectional_mask_sdpa`) skip the `is_causal` shortcut whenever
    `is_tracing()` is True — which is True under BOTH torch.export and
    torch.compile. torch.export genuinely needs the explicit mask (pytorch#108108),
    but torch.compile does not, and blocking it makes compiled output diverge from
    eager. We wrap the two functions so that under torch.compile-but-not-export
    they behave as if not tracing (take the shortcut), matching eager. The wrapper
    does NOT reimplement the (version-varying) body — it temporarily neutralizes
    the module-level `is_tracing` for the duration of the original call only.

M2 — Train-mode LayerDrop / dropout RNG determinism (issue #130):
    Audio encoders (Wav2Vec2, Hubert, UniSpeech, …) draw `torch.rand([])` per
    layer for LayerDrop; under torch.compile the captured RNG advances out of step
    with eager, so different layers are dropped (O(1) divergence). The upstream
    fix skips the draw under compile. For an eager-vs-compile *sweep* comparison
    that is not enough on its own (eager would still draw and drop while compile
    would not), so for the sweep we make BOTH paths deterministic: in train mode we
    zero every LayerDrop / dropout / SpecAugment probability so neither path
    drops/masks stochastically. This is the determinism the #130 fix delivers,
    applied symmetrically so the bitwise comparison is apples-to-apples.

    NOTE (important asymmetry, surfaced for the report): the production #130 fix is
    compile-only, so even *after* it lands, an eager-vs-compile bitwise comparison
    of a stochastic-LayerDrop model is NOT bitwise-equal unless eager is also made
    deterministic. The symmetric neutralization here is what lets the sweep
    *measure* post-fix equivalence; it is intentionally stronger than the upstream
    patch.
"""

import os
import sys

PATCH_ENV = "CORPUS_HF_BITWISE_PATCH"

# Config attributes that gate stochastic LayerDrop / SpecAugment in train mode.
_LAYERDROP_ATTRS = (
    "layerdrop", "encoder_layerdrop", "decoder_layerdrop",
    "mask_time_prob", "mask_feature_prob",
)
# nn.Dropout probability config attrs (zeroed for train-mode determinism).
_DROPOUT_CONFIG_ATTRS = (
    "dropout", "attention_dropout", "hidden_dropout", "activation_dropout",
    "feat_proj_dropout", "final_dropout", "hidden_dropout_prob",
    "attention_probs_dropout_prob", "classifier_dropout", "classifier_dropout_prob",
)

_GLOBAL_PATCHES_APPLIED = []
# The genuine `is_tracing` captured ONCE at patch time, so the wrapper always
# restores the real original (not a previously-installed neutralizer). Hardens
# against any nested/reentrant call to the wrapped helpers (adversary gap 1).
_TRUE_IS_TRACING = None
# Reentrancy depth: only the outermost wrapped call swaps/restores is_tracing.
_SWAP_DEPTH = 0
# Number of times the M1 neutralize branch actually fired (compile-not-export).
# Surfaced in the result stamp so a patched row PROVES M1 was active, rather
# than silently being a no-op when is_compiling() was False (adversary gap 2).
_M1_FIRE_COUNT = 0


def m1_fire_count() -> int:
    """How many times the M1 SDPA-shortcut neutralize branch fired this process."""
    return _M1_FIRE_COUNT


# Number of times the M3 Reformer-LSH deterministic rotation draw actually fired.
# Surfaced in the stamp so a Reformer row PROVES M3 was active (a zero count means
# the deterministic draw never ran and any equivalence is NOT attributable to M3).
_M3_FIRE_COUNT = 0


def m3_fire_count() -> int:
    """How many times the M3 Reformer-LSH deterministic rotation draw fired."""
    return _M3_FIRE_COUNT


def is_enabled() -> bool:
    """True when the bitwise patch is requested via the env var."""
    return os.environ.get(PATCH_ENV, "") not in ("", "0", "false", "False")


def _log(msg: str) -> None:
    print(f"[HF-BITWISE-PATCH] {msg}", file=sys.stderr, flush=True)


def _resolve_compile_predicates():
    """Return (is_compiling, is_exporting) callables, version-tolerant."""
    try:
        from transformers.utils.import_utils import (
            is_torchdynamo_compiling,
            is_torchdynamo_exporting,
        )
        return is_torchdynamo_compiling, is_torchdynamo_exporting
    except Exception:
        import torch

        def is_torchdynamo_compiling():
            comp = getattr(torch, "compiler", None)
            return bool(getattr(comp, "is_compiling", lambda: False)())

        def is_torchdynamo_exporting():
            comp = getattr(torch, "compiler", None)
            return bool(getattr(comp, "is_exporting", lambda: False)())

        return is_torchdynamo_compiling, is_torchdynamo_exporting


def apply_global_patches() -> list:
    """Monkeypatch the SDPA causal-mask-ignore helpers (M1). Idempotent.

    Returns the list of patch identifiers applied (empty if transformers/masking
    utils are unavailable on this version)."""
    global _GLOBAL_PATCHES_APPLIED
    if _GLOBAL_PATCHES_APPLIED:
        return _GLOBAL_PATCHES_APPLIED

    try:
        import transformers.masking_utils as mu
    except Exception as e:  # pragma: no cover - defensive
        _log(f"masking_utils unavailable ({type(e).__name__}); M1 not applied")
        return _GLOBAL_PATCHES_APPLIED

    is_compiling, is_exporting = _resolve_compile_predicates()
    applied = []

    global _TRUE_IS_TRACING
    _TRUE_IS_TRACING = mu.is_tracing  # capture the genuine original exactly once
    _neutralized = lambda *a, **k: False

    for fn_name in ("_ignore_causal_mask_sdpa", "_ignore_bidirectional_mask_sdpa"):
        orig = getattr(mu, fn_name, None)
        if orig is None or getattr(orig, "_corpus_bitwise_patched", False):
            continue

        def make_wrapper(orig_fn):
            def wrapper(*args, **kwargs):
                # Under torch.compile (but NOT torch.export), make the helper see
                # is_tracing()==False so it takes the eager causal-mask shortcut.
                global _M1_FIRE_COUNT, _SWAP_DEPTH
                if is_compiling() and not is_exporting():
                    _M1_FIRE_COUNT += 1
                    outermost = (_SWAP_DEPTH == 0)
                    if outermost:
                        mu.is_tracing = _neutralized
                    _SWAP_DEPTH += 1
                    try:
                        return orig_fn(*args, **kwargs)
                    finally:
                        _SWAP_DEPTH -= 1
                        if outermost:
                            # Restore the TRUE original, never a stale neutralizer.
                            mu.is_tracing = _TRUE_IS_TRACING
                return orig_fn(*args, **kwargs)

            wrapper._corpus_bitwise_patched = True
            wrapper.__name__ = getattr(orig_fn, "__name__", fn_name)
            return wrapper

        setattr(mu, fn_name, make_wrapper(orig))
        applied.append(f"masking_utils.{fn_name}")

    if applied:
        _log("M1 applied (SDPA causal-mask shortcut under compile): " + ", ".join(applied))

    # M3 (Reformer-gated): make LSH rotation draws deterministic.
    applied += _apply_reformer_lsh_determinism()

    _GLOBAL_PATCHES_APPLIED = applied
    return applied


# Fixed seed for Reformer LSH rotations (any constant works — the goal is for the
# eager and compiled paths to draw IDENTICAL rotations, not a specific value).
_REFORMER_HASH_SEED = 42


def _apply_reformer_lsh_determinism() -> list:
    """M3 (Reformer-only): make `LSHSelfAttention._hash_vectors` draw its random
    rotations from an explicit local `torch.Generator`, instead of the global RNG.

    Reformer's LSH hashing draws `random_rotations = torch.randn(...)` from the
    GLOBAL RNG. Under torch.compile, dynamo substitutes its own RNG, so the
    rotations — and therefore the LSH bucket assignment — diverge from eager,
    producing O(1) output divergence. HF's built-in determinism hook
    (`config.hash_seed` -> `torch.manual_seed`) does NOT help, because dynamo does
    not honor an in-graph global `manual_seed`. An explicit `torch.Generator`
    seeded with a fixed value is state-independent of the global RNG, so the same
    rotations are produced on both paths. Re-seeded per call so every call is
    deterministic. Only touches Reformer (LSHSelfAttention); no-op for any other
    model. Idempotent.

    SCOPE (adversary gap 1): validated for `backend="eager"` — the backend the
    corpus bitwise sweep uses. The mechanism relies on the compiled path executing
    the Python-level `torch.randn` rebind; the per-process `m3_reformer_rotation_fired`
    count in the result stamp proves the deterministic draw actually ran (a zero
    count means M3 was a no-op). NOT validated under `backend="inductor"` (not used
    by the sweep); inductor's functionalized RNG may behave differently.

    Re-seeding to the same value every call means every LSH layer/round draws
    IDENTICAL rotations — this degrades LSH hash quality but is correct for the
    sweep's equivalence-measurement purpose (we measure eager-vs-compile divergence,
    not model quality). Do not "fix" the re-seed without understanding this."""
    try:
        import transformers.models.reformer.modeling_reformer as rm
    except Exception:
        return []  # Reformer not present on this version — nothing to patch
    import torch

    cls = getattr(rm, "LSHSelfAttention", None)
    orig = getattr(cls, "_hash_vectors", None) if cls is not None else None
    if orig is None or getattr(orig, "_corpus_bitwise_patched", False):
        return []

    def patched_hash(self, vectors, num_hashes, attention_mask, *args, **kwargs):
        global _M3_FIRE_COUNT
        real_randn = torch.randn
        gen = torch.Generator(device=vectors.device)
        gen.manual_seed(_REFORMER_HASH_SEED)

        def det_randn(*a, **k):
            global _M3_FIRE_COUNT
            _M3_FIRE_COUNT += 1
            k.pop("generator", None)
            return real_randn(*a, generator=gen, **k)

        # NOTE: the `torch.randn` rebind is process-global and assumes a
        # single-threaded worker (one model per worker process); a concurrent
        # thread calling torch.randn during this window would get `gen` (adversary
        # gap 4 — accepted under the one-model-per-process invariant).
        #
        # gap 2 (residual global-RNG mutation from orig()'s conditional
        # `torch.manual_seed(self.hash_seed)` at modeling_reformer.py ~L715):
        # NOT mitigated by snapshotting global RNG state here, because
        # `torch.cuda.get_rng_state_all()` is not traceable under dynamo (raises
        # "source must be provided in options"). It is a non-issue in practice:
        # `config.hash_seed` defaults to None, so that manual_seed never runs in
        # our sweep. If a config DOES set hash_seed, a residual downstream-RNG
        # divergence could remain — documented limitation, revisit only if such a
        # config enters the corpus.
        torch.randn = det_randn
        try:
            return orig(self, vectors, num_hashes, attention_mask, *args, **kwargs)
        finally:
            torch.randn = real_randn

    patched_hash._corpus_bitwise_patched = True
    cls._hash_vectors = patched_hash
    _log("M3 applied (Reformer LSH rotation determinism via local Generator)")
    return ["reformer.LSHSelfAttention._hash_vectors"]


def _iter_configs(model):
    """Yield the model config plus any nested sub-configs (encoder/decoder/text/…)."""
    seen = set()
    cfg = getattr(model, "config", None)
    stack = [cfg] if cfg is not None else []
    # Pull common nested-config getters too.
    getter = getattr(model, "get_text_config", None)
    if callable(getter):
        try:
            stack.append(getter())
        except Exception:
            pass
    while stack:
        c = stack.pop()
        if c is None or id(c) in seen:
            continue
        seen.add(id(c))
        yield c
        # Walk attribute values that are themselves config-like objects.
        for attr in list(vars(c)) if hasattr(c, "__dict__") else []:
            try:
                v = getattr(c, attr)
            except Exception:
                continue
            if hasattr(v, "to_dict") and hasattr(v, "__dict__") and id(v) not in seen:
                stack.append(v)


def apply_model_determinism(model, mode: str) -> dict:
    """M2: zero stochastic LayerDrop / dropout / SpecAugment in train mode.

    No-op in eval mode (dropout/LayerDrop already inactive). Returns a small
    summary dict for result stamping.

    Mutates the live model/config objects in place with no restore. This is safe
    because the orchestrator spawns ONE worker subprocess per model
    (`orchestrator.spawn_worker`), so there is no cross-model reuse within a
    process. Do not rely on this function inside a multi-model loop in one
    process without re-creating models between calls."""
    summary = {"configs_zeroed": 0, "dropout_modules_zeroed": 0,
               "cached_dropout_floats_zeroed": 0}
    if mode != "train":
        return summary

    import torch

    for cfg in _iter_configs(model):
        touched = False
        for attr in _LAYERDROP_ATTRS + _DROPOUT_CONFIG_ATTRS:
            if hasattr(cfg, attr):
                try:
                    val = getattr(cfg, attr)
                except Exception:
                    continue
                if isinstance(val, (int, float)) and val != 0:
                    setattr(cfg, attr, 0.0)
                    touched = True
        if touched:
            summary["configs_zeroed"] += 1

    for _, m in model.named_modules():
        # (a) nn.Dropout modules
        if isinstance(m, torch.nn.Dropout) and getattr(m, "p", 0) > 0:
            m.p = 0.0
            summary["dropout_modules_zeroed"] += 1
        # (b) Cached dropout PROBABILITIES stored as float attrs at __init__ and
        # used via nn.functional.dropout (e.g. ReformerModel: self.dropout =
        # config.*_dropout_prob). These have no nn.Dropout module to zero and the
        # config attr was already copied at init, so config-zeroing above misses
        # them — they are a real RNG-divergence source in train mode under compile.
        # LIMITATION (adversary gap 3): this matches only an attr literally named
        # `dropout`. Reformer names all of them `dropout`, but a future model that
        # caches a functional-dropout prob under a different attr name (e.g.
        # `self.drop_p`) would NOT be caught here and could show residual train-mode
        # divergence under the patch. Kept conservative (no name-substring scan) to
        # avoid zeroing unrelated floats; extend the attr list if such a model
        # appears. (`self.dropout = nn.Dropout(...)` modules are correctly skipped —
        # an nn.Module is not an int/float — and handled by branch (a) above.)
        _cached = getattr(m, "dropout", None)
        if isinstance(_cached, (int, float)) and not isinstance(_cached, bool) and _cached > 0:
            m.dropout = 0.0
            summary["cached_dropout_floats_zeroed"] += 1

    return summary


def marker() -> dict:
    """The stamp written into every result row when the patch is active."""
    return {
        "hf_bitwise_patched": True,
        "hf_bitwise_patch_note": (
            "NON-STANDARD transformers: M1 SDPA causal-mask shortcut under compile "
            "(#129) + M2 train-mode LayerDrop/dropout determinism incl. cached "
            "functional-dropout floats (#130) + M3 Reformer LSH rotation "
            "determinism. Not comparable to stock-HF corpus numbers."
        ),
    }
