# mre Subagent — Retrospectives

Iteration cadence: review after every batch of dogfoods (recommended ≥3 ledger rows added).

## V1.0.5 — 2026-05-09

**Cumulative ledger:** 21 rows (all SUCCESS, median 2 min, all Strategy A except 1 Strategy B).

**Changes encoded across V1.0.0 → V1.0.5:**
- V1.0.1: MRE size flexibility, different-failure Repro variant, provenance_anchor required, click-decision title criteria
- V1.0.2: torch-internal anchor, threading-primitive, data-dep SymFloat, symptom-string drift, PYTHONPATH stripping, fragment specificity
- V1.0.3: mutation-class, drift-confirmed-widespread, same-file cluster
- V1.0.4: MANDATORY title-vs-actual divergence check, `is_torchdynamo_compiling` anti-pattern, gap-may-be-fixed-upstream (pivot OR document), multi-gap issue handling, output_capturing.py cluster expanded
- V1.0.5: DUPLICATE_CANDIDATE outcome + duplicate-detection step

**Known gaps:**
- Strategy B/C/D/E coverage is sparse (0–1 ledger entries each). The persona's coverage of these classes is theoretical.
- Failure paths (PROVENANCE_UNKNOWN, TIME_BUDGET_EXHAUSTED, VERIFICATION_FAILED, DUPLICATE_CANDIDATE) untested with real data.
- Concurrent-write safety on ledger.jsonl deferred to V2.

**Next iteration trigger:** when the ledger crosses 30 rows OR when a new strategy class hits ≥3 attempts OR when the analyzer flags a deep-dive candidate.
