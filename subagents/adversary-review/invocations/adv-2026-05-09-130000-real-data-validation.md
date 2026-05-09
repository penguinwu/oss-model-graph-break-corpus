---
case_id: adv-2026-05-09-130000-real-data-validation
subagent: adversary-review
date_utc: 2026-05-09T13:00:00Z
persona_sha: 1f36118aaed67cfc4b3d74a25f131f163986663b
target: file-issue Phase 3 v1.0 (5 commits PUSHED) validated against 3 real dynamo issues
target_artifacts:
  - https://github.com/penguinwu/oss-model-graph-break-corpus issues 99, 98, 92
  - persona.md (post-real-data amendments: stable fragment + affected-models table guidance)
  - SKILL.md (Step 2.5)
  - verify_repro.py + file_issues.py (the v1.0 5-commit chain)
parent_case_ids:
  - adv-2026-05-09-113538-repro-gate-design-v1
  - adv-2026-05-09-120800-repro-gate-design-v3
verdict: reframe
confidence: high
gaps_found: 6
high_severity: 2
medium_high_severity: 2
medium_severity: 2
disposition: gap 1 fixed in v1.0.1 (this commit); gaps 2, 3, 4, 5, 6 logged for v1.0.2 / v1.5
---

## Why this case file exists

Per Peng directive 2026-05-09 ~08:26 ET: "Pick three more dynamo issues and apply the skills to propose additions to the issues. Don't commit them yet, mainly using these three issues to validate your design. If you discover any gaps or design flaws, address as many as possible without waiting on me. Please invoke adversary agents as well for hardening the design based on the actual examples."

This is the third adversary review on the file-issue Phase 3 lineage, the first one anchored on REAL data rather than design docs. The 3 dynamo issues:
- Issue 99: `[dynamo] SymInt/SymInt div not supported (5 sliding-window models, 100 breaks)`
- Issue 98: `[dynamo] EncodecModel: recompile-limit on ParametrizedConv1d (4 occurrences, 1 model)`
- Issue 92: `[dynamo] BUILD_STRING with SymNodeVariable in f-string assertions (19 VL models, 38 breaks)`

## Verdict + confidence

`reframe` (high confidence). Adversary's note: "Gap 1 is the highest-stakes silent-failure case found. Recommend landing it BEFORE the next NEW filing — the v1.0 gate is currently a tripwire where a single careless fragment choice silently breaks the verification chain on the next nightly cycle."

## Gap 1 (HIGH) — Stable-fragment validation has no mechanical pin (FIXED in this commit)

**Concern:** Persona's stable-fragment guidance is prose-only. Mode B's pre-submission gate has no regex check; verify_repro doesn't reject unstable fragments at write-time. Concrete failure: Mode B writes `{"fragment": "at 0x7f80abcd1234"}` for issue 99. Three days later, the address is different and verify_repro classifies the bug as `does-not-reproduce` — silent regression in a stable bug.

**FIX (LANDED v1.0.1):**
- `verify_repro.validate_signal_fragment_stability(fragment)` — refuses fragments matching `0x[0-9a-f]{4,}`, PIDs, ISO timestamps, line-number anchors, absolute home paths, UUIDs.
- Wired into `verify_repro.verify()` at extraction time (raises ValueError → CLI returns 1).
- Wired into `file_issues._validate_repro_evidence()` at posting time (defense-in-depth — catches JSONs from older verify_repro).
- 9 new tests in `tools/test_verify_repro.py` covering each unstable pattern + 3 stable examples drawn from the 3 actual issues (issue 99: "on only torch.SymInt arguments is not yet supported"; issue 98: "hit config.recompile_limit"; issue 92: "BUILD_STRING type error").

Pre-fix: 37 tests in test_verify_repro; post-fix: 37 tests pass (boundaries respected — existing classification + extraction tests use stable fragments).

## Gap 2 (HIGH) — Cluster cohesion check 10 breaks for 1-model-multi-occurrence clusters (LOGGED)

**Concern:** Issue 98 is the canonical example: 4 occurrences, 1 model. Mode A check 10 says "at least one OTHER `affected_case` from the cluster must show the same break_reason fingerprint." For 1-model clusters, there's no "other" model. The carve-out is ONLY for `single_manual: true` (1 case). A real 4-case-1-model cluster falls between the carve-outs and would `reframe` Mode A spuriously.

**Disposition:** Logged for v1.0.2. Requires:
- `cluster_failures.py` schema extension: `occurrence_count` field per cluster.
- Mode A check 10 prose update: "if cluster has ≥2 affected_cases, verify that at least one OTHER case shows the same break_reason fingerprint OR the same model exhibits the symptom across ≥2 distinct call sites."
- Test: `test_cluster_cohesion_multi_occurrence`.

Not blocking the next NEW filing because: (a) v1.0 cluster+dedup gate doesn't currently invoke Mode A check 10 against single_manual plans (which is the only path implemented for non-sweep filings); (b) sweep-driven filings cluster by (family, mode), so 1-model clusters are rare from-sweep. The defect would surface when a single-model bug filed via single_manual gets a follow-up sweep that confirms it across multiple occurrences — recoverable, but the user would have to manually skip Mode A check 10 with a written note.

## Gap 3 (MEDIUM-HIGH) — MRE_TOO_LARGE has no escape valve for genuinely-complex repros (LOGGED)

**Concern:** Issue 98's recompile-limit MRE NEEDS a parametrize-wrapped Conv1d + 8+ varying input shapes — credibly 25-30 lines, possibly 30+. Persona says >30 → `MRE_TOO_LARGE` and bounce to "smaller input." But for recompile-limit bugs, "smaller input" is the wrong axis — multiple varying shapes IS the surface.

**Disposition:** Logged for v1.0.2 / v1.5. Requires:
- Optional draft field `mre_size_justification: "<text>"`. If present + cites structural reason, Mode B's calibration treats line-count ceiling as soft (still hard cap at 60 lines).
- Documented in persona Calibration section with issue 98 as cautionary example.

Not blocking immediately — Mode B can manually downgrade `MRE_TOO_LARGE` to `proceed-with-fixes` with a justification note in the case file's disposition until v1.0.2 lands.

## Gap 4 (MEDIUM-HIGH) — Affected Models table location has prose guidance but no gate (LOGGED)

**Concern:** Persona just amended says the table goes "right after test command and symptom-captured line, BEFORE the `<details>` signal block." Mode B's pre-submission checklist has no item that verifies this ordering. A subsequent assembler invocation could place it anywhere; verify_repro's regexes match anywhere; the maintainer reads a body where scope is detached from context.

**Disposition:** Logged for v1.0.2. Add Mode B pre-submission gate item + `_validate_body_shape_ordering` helper. Pin with `test_body_shape_ordering`.

## Gap 5 (MEDIUM) — Illustrative ```python fences are an unowned third class of code block (LOGGED)

**Concern:** All 3 issues have a `Pattern` section with a code fence. None of these are MRE OR original_command — they're illustrative. Mode B might tag an illustrative Pattern fence as `python` (untagged) which slips past the gate, OR as `python repro=true` which trips the "expected exactly 1 fence" refusal.

**Disposition:** Logged for v1.0.2. Persona Mode B should enumerate the THREE classes:
- (i) original_command in HTML comment ONLY
- (ii) MRE: ` ```python repro=true ` (exactly one)
- (iii) illustrative Pattern: ` ```text ` (NOT `python`)

Pin with doc-consistency rule scanning posted bodies + Mode B pre-submission gate item.

## Gap 6 (MEDIUM) — v1.5 EDIT migration plan is unloaded (LOGGED)

**Concern:** v1.0 EDIT path is exempt from the new gate; v1.5 will extend it. But the 3 existing issues have NEITHER the new body shape NOR fix-suggestion-removed content. v1.5 EDIT path has two bad options: (a) full body rewrite — destroys the human-readable framing the maintainer is used to + invalidates line-anchored comments; (b) preserve old shape — leaves Phase-3 verification mechanics impossible to apply. Neither is acknowledged.

**Disposition:** Add a section to persona "Phase 3 v1.5 EDIT migration plan, deferred" naming the design fork explicitly. Don't decide now — record the fork so it doesn't get rediscovered when v1.5 lands.

## Gaps surfaced earlier (not in adversary's 6) but addressed in same session

Otter independently surfaced 7 gaps from real-data inspection BEFORE running adversary; 3 were addressed (signal fragment stability prose, affected-models table location prose, signal-fragment-stability examples). The other 4 overlap with adversary's gaps 3, 5, 6 + the Mode A criterion 4 enforcement on existing bodies (which is already known and intentionally deferred to v1.5 EDIT path).

## Process learnings to encode

Adversary noted: "v1's 13 gaps got 100% disposition... but the disposition table didn't catch that v3's 'body-shape unification' reopened gap 1 in a new form." This third review extends the pattern: REAL DATA surfaces gaps that DESIGN REVIEWS don't, because real bodies have shapes and patterns the design's mental model didn't anticipate. **Recommendation for permanent practice:** after every design ship, run the design against ≥2 real artifacts BEFORE the next big design layer is added. The "Pattern" section across all 3 issues, the recompile-limit MRE difficulty, the 1-model-multi-occurrence shape — all invisible from design alone.

## What this case file binds

This case file documents the SECOND adversary cycle on v1.0 (post-ship). Gap 1 fix is in v1.0.1 (this commit). Gaps 2-6 are logged here for v1.0.2 / v1.5 dispositions. The persona/SKILL prose changes from real-data inspection (stable fragment + affected-models table location) shipped in commit 5 of v1.0.
