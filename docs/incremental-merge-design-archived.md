# Incremental Merge Design (Archived)

**Status:** Archived 2026-04-16. Superseded by the experiment-based workflow
which uses scoped replacement exclusively. Kept for reference in case we
need to revive incremental merging.

## Overview

The merge tool (`update_corpus.py`) supported two modes:

### Overlay (incremental)
Partial sweep results merged on top of existing corpus. Models not in the
sweep keep their existing entries. Version and timestamp safety checks
prevent accidental cross-version merges.

### Replace (full)
Sweep results become the complete corpus. Models not in the sweep are dropped.

### Explain-only
When only `explain_results.json` exists (no `identify_results.json`), overlay
break_reasons and graph counts onto existing corpus entries without touching
identify fields or adding/removing models.

## Safety Guards

### Version safety check (overlay only)
Blocks overlay when PyTorch, transformers, or diffusers versions differ
between sweep results and corpus. Prevents mixed-version corpus where
skipped models reflect old versions but metadata claims new.

```python
version_checks = [
    ("PyTorch", meta.get("pytorch_version"), versions.get("torch").split("+")[0]),
    ("transformers", meta.get("transformers_version"), versions.get("transformers")),
    ("diffusers", meta.get("diffusers_version"), versions.get("diffusers")),
]
mismatches = [(name, corp, sweep)
              for name, corp, sweep in version_checks
              if corp and sweep and corp != sweep]
```

### Timestamp staleness check (overlay only)
Blocks merging older sweep results over newer corpus to prevent accidental
data regression. Compares sweep timestamp against corpus `last_updated`.

### Completeness marker
For `--replace` mode, the results metadata must include version info so the
corpus records which versions it was built from.

## Why It Was Archived

1. **Frankenstein corpus risk** — incremental overlays from different experiments
   create a corpus where different models were swept under different conditions.
   No single experiment "owns" the corpus state.

2. **Dead model accumulation** — overlay never removes models. Models removed
   upstream persist in the corpus forever unless someone manually cleans them.

3. **Merge ordering matters** — merging stable then unstable (two overlays)
   produces different results than merging unstable then stable if there are
   any conflicts. The final state shouldn't depend on merge order.

4. **Complexity cost** — version guards, timestamp checks, scoped replace,
   combine tools, and completeness detection all exist to make incremental
   safe. The experiment model eliminates the need for all of them.

5. **Time savings not worth it** — a full identify sweep takes ~100 min.
   The complexity of incremental mode exceeds the value of saving that time,
   especially since sweeps run unattended.

## Lessons Learned

- The explain-only merge mode was useful for adding break_reasons without
  re-running identify. In the experiment model, this becomes "run the explain
  phase in the same experiment directory."

- The version safety check caught real issues (mixing PyTorch 2.10 and 2.11
  data). In the experiment model, version consistency is guaranteed because
  all phases run in the same environment.

- The `--force` flag for bypassing safety checks was used to work around
  the staleness check when merging explain results. This was a code smell —
  the correct fix was explain-only mode, not bypassing the check.
