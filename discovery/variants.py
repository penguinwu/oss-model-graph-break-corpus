"""Constraint variant catalog for the discovery agent.

Each variant = a constraint sentence that gets composed into the per-trial
prompt. Variants steer the agent toward (or away from) particular strategy
families. The discovery harness runs the same case under multiple variants and
clusters the resulting fixes by strategy fingerprint.

See discovery/design.md §4.2 for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    id: str                # short id (V0, V1, ...)
    name: str              # human-readable name
    constraint: str        # constraint sentence appended to the prompt body
    rationale: str         # why this variant exists


V0 = Variant(
    id="V0",
    name="bare",
    constraint="",
    rationale="Baseline. No constraint. Whatever strategy the agent reaches for first.",
)

V1 = Variant(
    id="V1",
    name="sparsity_preserved",
    constraint=(
        "IMPORTANT CONSTRAINT — preserve sparse compute pattern: "
        "the original implementation uses sparse expert dispatch (each expert only "
        "processes the tokens routed to it; top_k experts per token, not all experts). "
        "Your fix MUST preserve this sparse compute pattern. Replacing it with a dense "
        "rewrite that runs every token through every expert is NOT acceptable, even if "
        "mathematically equivalent."
    ),
    rationale="Pilot 4's constraint. Steers away from masked-dense (the human Dbrx fix).",
)

V2 = Variant(
    id="V2",
    name="bitwise_equivalent",
    constraint=(
        "IMPORTANT CONSTRAINT — bitwise equivalence: "
        "the compiled output must be bitwise equal to the eager output. "
        "Any rewrite that reorders floating-point operations (even if mathematically "
        "equivalent) is NOT acceptable. Strategies that change op order — bmm fusion, "
        "masked-dense rewrites — fail this constraint."
    ),
    rationale="Forces escape-hatch family (custom_op / disable / cond) since only those preserve op order.",
)

V4 = Variant(
    id="V4",
    name="no_escape_hatches",
    constraint=(
        "IMPORTANT CONSTRAINT — no escape hatches: "
        "do not use torch.library.custom_op, torch._dynamo.disable, "
        "torch._dynamo.allow_in_graph, or torch.cond. The fix must rewrite the "
        "code so it traces cleanly under torch.compile."
    ),
    rationale="Forces tracing-friendly rewrite (range-loop, bmm, vectorized-bucketize).",
)

V6 = Variant(
    id="V6",
    name="no_config_flags",
    constraint=(
        "IMPORTANT CONSTRAINT — no dynamo config changes: "
        "do not modify torch._dynamo.config (e.g. capture_dynamic_output_shape_ops, "
        "suppress_errors). The fix must be in source code only — no runtime flag flips."
    ),
    rationale="Added v0.2 after Pilot 4 forensic. Forces a code-only fix so we can see whether the agent has a fallback when flag-set is taken away.",
)

V8 = Variant(
    id="V8",
    name="model_layer_only",
    constraint=(
        "IMPORTANT CONSTRAINT — model-layer fix only: "
        "do not edit the test/baseline script (the script that constructs the model "
        "and calls torch.compile). The fix must live entirely in the model source "
        "files. Setup-layer edits — changing input shapes, swapping numpy for torch "
        "in the runner, adjusting random-seed patterns, modifying the torch.compile "
        "call site, restructuring how the model is invoked — are NOT acceptable. "
        "The model itself must be made to trace cleanly under the existing setup."
    ),
    rationale="Added 2026-04-25 for VitsModel case 3b. V0/V2/V4 produced 12/12 setup-required (agent escapes via baseline edits). V6 (no config flags) was the only variant that surfaced model-layer fixes (4/6 general). V8 closes the setup-edit escape route directly to test whether the agent can produce model-layer fixes consistently when forced.",
)

# Recommended starting set for Phase 1.
DEFAULT_VARIANTS: tuple[Variant, ...] = (V0, V1, V2, V4, V6)

# Lookup by id.
ALL_VARIANTS: dict[str, Variant] = {v.id: v for v in (V0, V1, V2, V4, V6, V8)}


def compose_prompt(case_body: str, variant: Variant) -> str:
    """Compose a per-trial prompt = case body + variant constraint."""
    if not variant.constraint:
        return case_body
    return f"{case_body}\n\n{variant.constraint}"


if __name__ == "__main__":
    # Smoke test: print catalog
    for v in DEFAULT_VARIANTS:
        print(f"--- {v.id} ({v.name}) ---")
        print(f"  rationale: {v.rationale}")
        print(f"  constraint: {v.constraint[:80]}{'...' if len(v.constraint) > 80 else ''}")
        print()
