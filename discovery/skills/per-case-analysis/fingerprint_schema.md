# Fingerprints CSV — schema (pinned)

`fingerprints.csv` is the per-trial classification artifact produced in Phase A. It MUST conform to this schema so cross-case synthesis (master plan §"Cross-case synthesis", question Q7) is mechanical.

If you need to add a new column or rename one, update THIS document FIRST and propagate to the per-case-analysis SKILL — never let the CSV diverge silently between cases.

## Columns (in order, header included)

| # | Column | Type | Allowed values | Notes |
|---|--------|------|----------------|-------|
| 1 | `trial` | string | `<skill_arm>_<variant>_<idx>` (e.g. `noskill_V0_1`, `debug_graph_breaks_V4_2`) | Matches the trial dir name produced by `run_case.py`. |
| 2 | `fix_locus` | enum | `model-only` \| `setup-only` \| `both` \| `unclear` | Which file class(es) the agent edited. `model-only`: only `modeling_*.py`. `setup-only`: only `baseline_*.py`. `both`: both. |
| 3 | `fix_shape_family` | enum | `deletion` \| `restructure` \| `wrap` \| `config-flag-flip` \| `escape-hatch` \| `input-type-tweak` \| `mixed` \| `other` | The dominant pattern. Use `mixed` if ≥2 distinct families and no clear primary. Use `other` only as last resort and describe in `agent_claim`. |
| 4 | `op_order_preserved` | enum | `yes` \| `no` \| `unclear` | Whether the fix preserves floating-point op order. `yes` if no reordering visible in diff; `no` if explicit reorder; `unclear` if can't tell. |
| 5 | `escape_hatches` | list (semicolon-sep) | Subset of: `custom_op`, `disable`, `cond`, `allow_in_graph`, `nonstrict_trace`, `leaf_function`, `is_compiling`, `none` | The escape hatches used. `is_compiling` refers to `torch.compiler.is_compiling()` guards (not in canonical list but worth tracking). `none` if no escape hatches. |
| 6 | `breaks_attacked` | list (semicolon-sep) | Free text, but use canonical bucket names where possible: `scalar_dense/.item()`, `nonzero/boolean-indexing`, `.tolist()`, `torch_compilable_check`, `@capture_outputs decorator`, `list comprehension`, `cumsum tensor slice`, `image_sizes (tensor)`, `unfold decomp`, `sdpa is_causal`, `generate_block_attention_mask` | Which break categories the agent's diff or final-summary explicitly addresses. Don't infer; only list what's evident. |
| 7 | `files_touched` | list (semicolon-sep) | File basenames without `.py` extension where convenient (e.g. `modeling_mistral3;modeling_pixtral;baseline_mistral3`) | Which source files the agent edited. Cross-check with diff. |
| 8 | `diff_lines` | int | non-negative | Total lines in `agent_diff.patch` (rough complexity measure). |
| 9 | `turns` | int | non-negative | Number of agent turns from stream metadata. Leave blank if not parseable. |
| 10 | `agent_claim` | string (no commas, semicolons OK) | Free text, one short phrase | Agent's own one-phrase summary at end of stream (e.g. "16->0 graph breaks; fullgraph ok"). |

## Format rules

- *CSV with header.* Column order matches above.
- *List cells use semicolon (`;`) as separator* to avoid conflict with CSV commas.
- *Strings with commas* should be quoted in standard CSV style.
- *Empty values* allowed where the data is genuinely unknown (e.g. `turns` if stream metadata didn't parse).
- *No trailing comma* on rows.

## Per-case extensions

If a case has unique axes worth tracking (e.g. for VitsModel: train-mode-specific patterns), add columns AFTER column 10. Document the extension in the case's `findings.md` "Methodology notes" section. Cross-case synthesis ignores extension columns — only columns 1-10 are guaranteed comparable.

If you find yourself wanting to add the SAME extension to multiple cases, it should be promoted to this canonical schema (open a PR).

## Anti-patterns

- Don't invent values. If a field is genuinely unknown, leave blank (or `unclear` for enum types).
- Don't combine multiple distinct values into one cell without semicolons. Cross-case parsing depends on consistent delimiters.
- Don't add new enum values without updating this schema first. (`fix_shape_family=other` is the safety valve for genuinely novel patterns; document them in `agent_claim`.)
- Don't reuse columns for different meanings across cases. If a case needs a different metric, add a column.

## Enum vocabulary — reference

### `fix_shape_family` definitions

- *deletion:* removes code (a decorator, a branch, an unused import). E.g. removing `@capture_outputs`.
- *restructure:* rewrites logic to be trace-friendly. E.g. replacing per-image loop with batched unfold using static config dims.
- *wrap:* wraps existing code in a guard or condition. E.g. `if not torch.compiler.is_compiling(): ...`.
- *config-flag-flip:* modifies `torch._dynamo.config` (e.g. `capture_scalar_outputs=True`).
- *escape-hatch:* uses one of: custom_op, disable, cond, allow_in_graph, nonstrict_trace, leaf_function.
- *input-type-tweak:* changes input shapes/types in baseline_*.py (e.g. tensor `image_sizes` → Python list).
- *mixed:* ≥2 distinct families with no clear primary. Default for complex fixes.
- *other:* a pattern not in this list. Describe in `agent_claim`.
