---
plan: Decouple model libraries (transformers, diffusers, timm) from torch venvs
status: phase-1-4-shipped
owner: Otter (drafting); Peng (reviewing)
created: 2026-05-03
revision: 2
last_check: 2026-05-04
---

## Status (2026-05-04)

Phase 1-4 shipped tonight in commit `7d3f6f4` (Otter, agent-time). Phase 5
(pip uninstall from torch venvs) deferred for Peng's morning sign-off.
HANDOFF.md has the full picture.

Verification:
- 5-gate test-sweep-changes walked + passed
- End-to-end CLI test: cu128 venv with `--transformers 5.5.3` records 5.5.3
  in sweep_state.json (overrides the venv's baked-in 5.6.2)
- Bit-identical apple-to-apple smoke on DistilBertModel (cu128 venv 5.6.2
  vs cu128 + modellib 5.6.2 PYTHONPATH)



# Decouple Model Libraries from Torch Venvs

## Problem

Today, each PyTorch venv at `~/envs/torch{N}/` or `~/envs/torch-nightly-{cu128,cu126}/` ships its own copy of transformers, diffusers, timm, etc. baked in at venv-creation time via `SWEEP_DEPS["transformers"]="5.6.2"` (a single hardcoded constant in `sweep/venv_setup.py`). This causes three concrete problems:

1. **Hardcoded coupling.** Same hardcoded transformers version (5.6.2) for all torch versions — torch 2.10, 2.11, 2.12-nightly all get whatever that constant says today. We're not following any upstream pinning (transformers and PyTorch are independent projects with independent release cadences); the value is just whichever was current the last time someone touched the constant.
2. **Hidden drift on venv reuse.** When a venv already exists matching a torch spec, `ensure_venv_ready` returns it as-is — whatever transformers it has, even if drifted from the constant via manual `pip install`. Two sweeps weeks apart against "the same" venv can use different transformers.
3. **Cannot test multiple model-lib versions against same torch.** If we want to compare "torch 2.12 against transformers 5.6.2 vs 5.7.0" we have to create two parallel torch venvs (~5GB each, mostly duplicated). And there's no CLI handle to do it — would require editing the constant.

The transformers/diffusers/timm versions ARE recorded in `sweep_state.json["versions"]` after-the-fact (verified 2026-05-03), so reports can cite them. But the *choice* of versions is made implicitly at venv-creation and not declarative.

## Goal

Make model library versions a **first-class, declarative input** to a sweep, decoupled from the torch venv they run against.

After this change:
```
tools/run_experiment.py nightly --torch 2.12 --cuda-variant cu128 \
    --transformers 5.6.2 --diffusers 0.37.1
```
should resolve a (torch venv, transformers tree, diffusers tree) triple, recorded in sweep state. Any combination should be runnable without recreating the torch venv.

## Proposed architecture

**New layout:**
```
~/envs/
  cu128/                         # pool venv (cu libs only) — unchanged
  cu126/                         # pool venv — unchanged
  torch211/                      # torch + torch_python_deps + small ABI-coupled deps
                                 # (huggingface-hub, tokenizers, safetensors)
                                 # NO transformers / diffusers / timm
  torch-nightly-cu128/           # same shape, nightly torch
  modellibs/
    transformers-5.5.3/          # standalone tree — site-packages-only layout
    transformers-5.6.2/
    transformers-5.7.0/
    diffusers-0.37.1/
    timm-0.9.2/
```

**Mechanism:** Each `modellibs/<pkg>-<ver>/` is a directory containing the package's installed tree (e.g. `modellibs/transformers-5.6.2/transformers/...`, plus its `*.dist-info/`). At sweep launch, we resolve the requested versions and inject those paths into `PYTHONPATH` ahead of the torch venv's site-packages. Python's import system picks up the modellib trees first, the torch venv supplies torch + everything else.

**Why ABI-coupled deps stay with the torch venv:** `tokenizers`, `safetensors`, `huggingface-hub` ship with C extensions (or pure-Python deps that have ABI-sensitive transitive deps). They're small, change rarely, and pinning them to the torch venv's Python is the simplest correct option. Only the *pure-Python* model-definition trees (transformers, diffusers, timm) move to standalone directories.

**At sweep launch (run_sweep.py):**
1. Resolve torch venv via existing `ensure_venv_ready(torch_spec)` → `venv_path`.
2. **NEW**: For each `--<pkg> <ver>` flag, resolve `modellibs/<pkg>-<ver>/` exists; if not, run `pip install --target ~/envs/modellibs/<pkg>-<ver>/ <pkg>==<ver>` using a pip from the torch venv (one-time bootstrap per (pkg, ver) pair).
3. **NEW**: Set `PYTHONPATH=<modellib_tree_1>:<modellib_tree_2>:...` before re-exec'ing under `venv_path/bin/python`.
4. Existing version-recording in `sweep_state.json["versions"]` works unchanged — `_log_versions` runs in the resolved env and reads `transformers.__version__` from whatever Python found, which will be the modellib tree.

**At worker launch (worker.py):**
- `PYTHONPATH` is inherited from parent → no change needed in worker. Confirm in smoke validation.

## Migration strategy

**Phase 1: Bootstrap modellibs trees (one-shot script)**
Write `tools/bootstrap_modellibs.py` that:
1. For each (pkg, ver) we currently care about: `pip install --target ~/envs/modellibs/<pkg>-<ver>/ <pkg>==<ver>`
2. Verify importable: `PYTHONPATH=<tree> python -c "import <pkg>; print(<pkg>.__version__)"`
3. Idempotent — skip if tree already present and version matches.

Initial run installs:
- transformers: 5.4.0, 5.5.3, 5.6.2 (the pairs we have today)
- diffusers: 0.37.1
- timm: 0.9.2 (current sweep uses)

**Phase 2: New CLI flags + PYTHONPATH plumbing in run_sweep.py**
Add `--transformers`, `--diffusers`, `--timm` flags. Resolution logic. PYTHONPATH injection before re-exec.

**Phase 3: Smoke validation**
Run a 3-model smoke sweep (1 transformers model, 1 diffusers, 1 timm) under:
- (a) old layout (transformers in venv) — record results
- (b) new layout (transformers from modellibs tree) — record results
- Diff: graph_break_count, fullgraph status, error messages must match exactly

**Phase 4: Default behavior — DECIDED**
`--transformers` is **required** when the sweep uses transformers-backed models. No silent fallback. Same for `--diffusers` and `--timm` when those sources are enabled. This forces declarative selection from day one.

**Phase 5: Cleanup — DECIDED**
Once modellibs is proven (Phase 3 smoke validates), `pip uninstall transformers diffusers timm` from each torch venv. Single source of truth — no safety-net ambiguity about which copy got imported.

## Validation plan

Per `~/projects/pt2-skill-discovery/design/experiment_lifecycle.md` 5-gate model:

- **Gate 0 (Intent):** Decouple model libraries from torch venvs to make version selection declarative + enable cross-version testing without venv duplication.
- **Gate 1 (Hypothesis):** Setting `PYTHONPATH=<modellib_tree>:...` before re-exec causes Python to pick up the modellib tree's transformers; transformers can then `import torch` from the venv's site-packages without conflict; sweep results are bit-identical to the venv-installed-transformers baseline.
- **Gate 2 (Smoke):** 1 model, 1 mode, eval. Confirm import path + result equivalence.
- **Gate 3 (Multi-model):** 3 models (timm/hf/diffusers each), 2 modes. Confirm no cross-package interference.
- **Gate 4 (Conclusion):** Compare smoke results bit-for-bit to most recent baseline; if identical, modellibs is safe; if not, debug PYTHONPATH ordering / transitive dep issues.

## Risks

1. **Transitive dep version mismatch.** transformers 5.6.2 might transitively want huggingface-hub>=0.20, but the torch venv has 0.18. With `pip install --target` and no `--no-deps`, the modellibs tree will install its own copy of huggingface-hub alongside. PYTHONPATH ordering then determines which one wins.
   - **Mitigation:** Smoke test catches this. If it bites, install modellibs with `--no-deps` and rely on the torch venv's ABI-coupled deps; document explicit version requirements.
2. **C extension surprise.** If a transformers release silently introduces a new C-extension transitive dep, modellibs tree will have it but it might be Python-version-incompatible with the torch venv.
   - **Mitigation:** Bootstrap script verifies `import transformers` works under the target torch venv's python BEFORE marking the tree ready.
3. **PYTHONPATH leakage.** If we set PYTHONPATH at sweep launch and re-exec, child processes (workers) inherit it. Good for our case. But if a worker spawns its own subprocess with `subprocess.run(..., env={})`, the modellib path is lost.
   - **Mitigation:** Audit worker.py for subprocess calls; ensure env passthrough or explicit re-injection.
4. **Disk + bootstrap time.** Each modellibs tree is ~200MB. 5 transformers versions + 2 diffusers + 2 timm = ~2GB. One-time bootstrap takes ~5min over fwdproxy.
   - **Acceptable.** Smaller than the duplication we have today across torch venvs.

## Agent-time estimate

- Phase 1 (bootstrap script): 1 iteration
- Phase 2 (run_sweep.py changes): 1 iteration
- Phase 3 (smoke validation): 1 iteration + ~30min wall clock for sweep
- Phase 4 (default behavior): half iteration

**Total: ~3.5 active agent iterations + 1 smoke sweep wall-clock.** Realistic: half a session.

## What this does NOT do

- Doesn't rename or restructure existing torch venvs (only adds the modellibs sibling)
- Doesn't break any existing sweep — old `--torch` + `--cuda-variant` invocation still works (modellibs is opt-in via new flags)
- Doesn't enforce a transformers/torch compatibility table — that's the separate `check_compat` gate landed in `df3fd7c` and continues to apply
- Doesn't address the question "which transformers SHOULD we test against this torch" — that's a curation question, not a tooling question

## Decisions (recorded 2026-05-03)

1. **Phase 5 cleanup: yes** — strip transformers/diffusers/timm from torch venvs once modellibs proven. Single source of truth.
2. **Phase 4 default: explicit required** — no silent fallback; sweep refuses to launch without `--transformers` (or `--diffusers` / `--timm` for those sources).
3. **Bootstrap config: declarative file** at `sweep/modellibs.json`:
   ```json
   {
     "transformers": ["5.4.0", "5.5.3", "5.6.2"],
     "diffusers": ["0.37.1"],
     "timm": ["0.9.2"]
   }
   ```
   Why: version-controlled, reviewable in PRs, single source of "what we maintain". Bootstrap script (`tools/bootstrap_modellibs.py`) reads this and ensures every listed (pkg, ver) tree exists. Adding a new version = edit JSON + run bootstrap. CLI `--add pkg ver` reserved for ad-hoc one-offs that don't justify a checked-in entry.
