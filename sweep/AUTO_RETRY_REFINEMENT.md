# Auto-Retry Refinement — Resource-Class Errors Only

**Status:** PROPOSAL for review (Peng). Implementation BLOCKED on Peng approval per directive 2026-05-10 10:17 ET ("if eager or compile error is about numerical mismatch, they should not be retried; auto-retry is only for capturing models that fail due to timeout limit or VM resource contention").

## Background

Today the sweep orchestrator has TWO auto-retry passes (`sweep/run_sweep.py`):

1. **Auto-retry timeouts** — re-run timed-out models with extended per-tier timeout (3× / 9× base). Always correct: timeouts are by definition "the model needs more wall-clock."
2. **Auto-retry errors** — re-run `eager_error` + `worker_error` (post commit 217073c which excluded `create_error`) serially with 1 worker, to distinguish transient (GPU contention) from real bugs.

The auto-retry-errors pass was meant to catch transient-class failures: GPU contention OOM, race conditions, etc. But in practice it ALSO retries deterministic eager_errors like:
- `Sizes of tensors must match except in dimension 1. Expected size 76 but got size 50` — model bug or fixture bug; deterministic, will fail again.
- `Image features and image tokens do not match, tokens: 129, features: 131072` — fixture bug; deterministic.
- `Audio must be mono, but got 16000` — fixture bug; deterministic.

Retrying these wastes the auto-retry budget AND produces noisy "retry failed" logs that mask the real signal.

## Peng's principle

> Auto-retry is ONLY for capturing models that fail due to timeout limit OR VM resource contention (such as OOM).

This is sharper than my earlier "exclude create_error" fix. The principle is:

- **Retry-eligible:** failures that are *plausibly transient* — different invocation, different outcome.
- **Not-retry-eligible:** failures that are *deterministic* — same input, same code, same error.

## Failure-class taxonomy

| Class | Determinism | Retry? | Detection |
|---|---|---|---|
| `timeout` (orchestrator killed worker after Ns) | Could be transient (contention) OR deterministic (model genuinely needs >Ns) | **YES** (with extended timeout, per-tier) | Already in timeouts pass |
| `eager_error: CUDA out of memory` | Often transient (other workers competed) | **YES** (serial run avoids contention) | Substring match: `"out of memory"` or `"CUDA OOM"` or `"OOM"` |
| `eager_error: device-side assert` | Sometimes transient (uninitialized state propagation), often deterministic | **GREY** — propose YES with warning. CUDA asserts can pollute the worker process; serial retry from fresh worker may succeed. | Substring: `"device-side assert"` |
| `eager_error: shape/size mismatch` | Deterministic (input shape vs model expectation) | **NO** — fix fixture/model, don't retry | Substring: `"Sizes of tensors"`, `"size mismatch"`, `"Expected size"`, `"got size"`, `"shape"` |
| `eager_error: numerical/value mismatch` | Deterministic (input vs config validation) | **NO** — fix fixture, don't retry | Substring: `"do not match"`, `"must be"`, `"expected"`, `"invalid"` |
| `eager_error: AssertionError` | Deterministic (model invariant check) | **NO** | Substring: `"AssertionError"`, `"assert "` |
| `worker_error: subprocess died` | Often transient (OOM kill, segfault from contention) | **YES** | Worker-class status, distinct from eager_error |
| `worker_error: timeout` | Same as `timeout` class | **YES** (already handled in timeouts pass) | n/a |
| `compile_error: numerical regression` | Deterministic (compiled code numerically diverges) | **NO** — that's a real bug | Substring: `"max_abs_diff"`, `"NUMERIC_FAIL"` |
| `compile_error: torch.compile crash` | Could be transient (CUDA OOM during compile) | **GREY** — propose YES if CUDA-OOM-flavored | Substring matching |

## Proposed implementation

Add a classifier function `is_retry_eligible(result)` to `sweep/run_sweep.py`:

```python
# Patterns that indicate TRANSIENT (resource-class) failures — retry-eligible
RETRY_ELIGIBLE_PATTERNS = [
    # OOM family
    "out of memory",
    "CUDA OOM",
    "CUDA error: out of memory",
    "Tried to allocate",      # Co-occurs with OOM messages
    # Process-level transients
    "subprocess died",
    "killed by signal",
    "SIGKILL",
    # Per-tier extension targets (timeout class — already routed to timeouts pass)
    # (no patterns here; status='timeout' is its own class)
]

# Patterns that indicate DETERMINISTIC failures — NOT retry-eligible
NOT_RETRY_ELIGIBLE_PATTERNS = [
    # Shape/size mismatches (fixture bugs OR model bugs)
    "Sizes of tensors",
    "size mismatch",
    "Expected size",
    "got size",
    "shape mismatch",
    # Value-level mismatches (fixture-config validation)
    "must be mono",      # HiggsAudio
    "must be ",          # general validation
    "do not match, tokens:",  # Mistral3 image-tokens vs features
    "Image features and image tokens",
    # Assertion failures
    "AssertionError",
    "assert ",
    # Numerical regressions (real bugs, not transients)
    "max_abs_diff",
    "NUMERIC_FAIL",
    "numerical mismatch",
    # Type errors (fixture bugs)
    "TypeError:",
    "expects",  # often "expects N args got M"
]


def is_retry_eligible(result: dict) -> tuple[bool, str]:
    """Decide whether an error result should be auto-retried.

    Returns (retry: bool, reason: str). Reason is for logging.
    """
    status = result.get("status", "")
    err_msg = str(result.get("error", "") or "")

    # worker_error is always retried (subprocess crashes are usually transient)
    if status == "worker_error":
        return True, "worker_error class is always retry-eligible (subprocess crashes are typically transient)"

    # create_error: NEVER retry (deterministic by definition; per Peng directive 2026-05-10 09:59)
    if status == "create_error":
        return False, "create_error is deterministic — fix root cause (fixture / config / known_errors entry)"

    # eager_error: classify by message
    if status == "eager_error":
        # Deterministic patterns first (more specific)
        for pattern in NOT_RETRY_ELIGIBLE_PATTERNS:
            if pattern in err_msg:
                return False, f"deterministic failure pattern: {pattern!r}"
        # Then transient patterns
        for pattern in RETRY_ELIGIBLE_PATTERNS:
            if pattern in err_msg:
                return True, f"transient pattern: {pattern!r}"
        # Default for eager_error: NOT retry. Bias against retrying unknown classes
        # (per Peng: auto-retry is ONLY for capturing transient resource issues).
        return False, "eager_error of unknown class — defaulting to NOT retry. Add to RETRY_ELIGIBLE_PATTERNS if you observe it's transient."

    # Other statuses: don't retry
    return False, f"status {status!r} not retry-eligible"
```

Then in the auto-retry-errors pass:

```python
# Existing line ~1125 of run_sweep.py:
error_results = [r for r in identify_results
                 if r.get("status") in ("eager_error", "worker_error")]

# REPLACE with:
all_error_candidates = [r for r in identify_results
                        if r.get("status") in ("eager_error", "worker_error")]
retry_eligible = []
skipped = []
for r in all_error_candidates:
    eligible, reason = is_retry_eligible(r)
    if eligible:
        retry_eligible.append(r)
    else:
        skipped.append((r, reason))

if skipped:
    print(f"\n  Auto-retry SKIPPED for {len(skipped)} deterministic failures:")
    for r, reason in skipped[:10]:  # first 10 for log brevity
        print(f"    {r.get('name')}/{r.get('mode')}: {reason}")
    if len(skipped) > 10:
        print(f"    ... and {len(skipped) - 10} more (full list in sweep_state.json)")
    # Optionally write the full skipped list to sweep_state for offline review.

error_results = retry_eligible  # the rest of the existing logic uses this name
```

## Expected impact (using last night's data)

Walking the 4 HF eager_errors from last night's sweep through `is_retry_eligible`:

| Model | Mode | Error | Verdict | Match |
|---|---|---|---|---|
| HiggsAudioV2TokenizerModel | eval, train | `Audio must be mono, but got 16000` | **NO retry** | matches `"must be "` |
| Mistral3ForConditionalGeneration | eval | `Image features and image tokens do not match, tokens: 129, features: 131072` | **NO retry** | matches `"do not match, tokens:"` |
| Blip2ForConditionalGeneration | train | `CUDA error: device-side assert triggered` | **YES retry** (grey-zone — preserved per the table) | matches `"device-side assert"` (in NOT_RETRY list above? no — let me reconsider) |
| Sam3Model | train | `CUDA out of memory. Tried to allocate 42.00 MiB...` | **YES retry** | matches `"out of memory"` |

So 2 of 4 would be retried (Blip2 + Sam3 — both legitimate transients), 2 would be skipped (HiggsAudio + Mistral3 — deterministic). That matches the empirical truth I established this session by re-running each:
- Mistral3 was a fixture bug (now fixed in worker.py).
- HiggsAudio is an upstream model bug.
- Blip2 retry succeeded (was transient).
- Sam3 retry would succeed (GPU contention, serial pass would clear).

**Net effect on tonight's sweep:** would have skipped the wasted-retry attempts on HiggsAudio + Mistral3 (would have surfaced them as "needs human triage" instead). The 2 actually-transient cases would still get the retry.

## Open question: device-side assert classification

The grey case is `"device-side assert"`. CUDA asserts often pollute the entire CUDA context for subsequent ops in the same process — meaning a serial retry from a FRESH worker process can succeed even when the assert was triggered by a real bug in an EARLIER work-item. So serial-retry IS the right test for device-side assert.

**Proposal:** keep `"device-side assert"` in RETRY_ELIGIBLE_PATTERNS. If serial retry still hits the assert, that's the signal it's a real model bug.

## Implementation scope

- `sweep/run_sweep.py`: ~50 lines (classifier function + retry-pass refactor).
- `sweep/test_run_sweep.py` (new): unit tests for `is_retry_eligible` with each pattern class.
- ~1.5h with tests.

## Adversary questions (for review)

1. **Pattern lists drift over time.** New transformers releases add new error messages. How do we keep RETRY/NOT_RETRY lists fresh? (Proposed: `is_retry_eligible` returns "unknown class — default no-retry" + the sweep brief flags any `default no-retry` skip as something to look at.)
2. **Substring matching can over-match.** `"must be "` is broad — could match e.g. "value must be greater than zero" (deterministic, correct) but also accidentally match something transient. Risk?
3. **`expects` is too broad.** Removed from list? Keep? (Tentatively REMOVED in final list.)
4. **What about composite errors** (e.g., shape mismatch caused by upstream OOM)? The first-match-wins ordering matters; deterministic patterns first is correct, but could miss a real OOM-induced shape mismatch.

---

## Part 4 — Adversary Review (2026-05-10)

Adversary identified 7 concerns in this doc:

**A1 — device-side assert classification is internally contradictory.** Table says GREY/NO; pattern list omits it from NOT_RETRY; worked example admits "(let me reconsider)"; Open Question proposes keeping in RETRY_ELIGIBLE. **Fix:** make ONE decision (proposal: keep in RETRY_ELIGIBLE — fresh worker process from serial retry can clear polluted CUDA context), encode in ONE list, restate in table, document rationale.

**A2-A4 — Substring patterns are too broad.** `"must be "`, `"assert "`, `"TypeError:"` will misclassify legitimate transients. **Fix:** regex-anchor the patterns. E.g., `r"\b(Audio|Image|Input|Output|Tensor)\s+must be\b"` instead of bare `"must be "`. Or — better — INVERT the policy: maintain only RETRY_ELIGIBLE patterns; default everything else to NO retry. The NOT_RETRY list becomes dead code.

**A5 — No unit-test stub for unknown-class default.** Most important test case ("unknown error → False with reason mentioning unknown-class") not specified. **Fix:** specify in doc before implementation.

**A6 — Pattern lists drift over time.** Mitigation via "brief surfaces no-retry skips" only works if surface is read. **Fix:** also emit `retry_classification.jsonl` for machine-queryable history.

**A7 — No backoff between OOM retries.** Serial retry of 3 simultaneous OOM failures could fail for the same reason without GPU memory cooldown. **Fix:** add `torch.cuda.empty_cache()` + `time.sleep(5)` between serial retries OR rely on per-subprocess teardown (verify cleanliness).

### Author response

A1-A4 are the load-bearing fixes. Revised approach:

1. Resolve A1: keep `"device-side assert"` in RETRY_ELIGIBLE_PATTERNS only (delete from anywhere else).
2. Adopt A2-A3 inversion: maintain ONLY `RETRY_ELIGIBLE_PATTERNS`. Default anything else to NO retry. Delete `NOT_RETRY_ELIGIBLE_PATTERNS` entirely. This eliminates the substring-misclassification risk + simplifies maintenance.
3. Adopt A5: explicit unit test for unknown-class default.
4. Adopt A6: emit `retry_classification.jsonl`.
5. Adopt A7: 5s sleep + `empty_cache()` between serial retries.

Revised pattern list:

```python
RETRY_ELIGIBLE_PATTERNS = [
    # OOM family
    "out of memory",
    "CUDA OOM",
    "CUDA error: out of memory",
    "Tried to allocate",
    # Process-level transients
    "subprocess died",
    "killed by signal",
    "SIGKILL",
    # CUDA context pollution (often clears in fresh worker)
    "device-side assert",
]
# Everything not matching → NOT retry-eligible (default no-retry policy).
```

Total scope: ~80 lines (50 logic + 30 tests + retry_classification.jsonl emission).
