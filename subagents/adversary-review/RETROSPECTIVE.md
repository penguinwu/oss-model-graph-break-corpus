# adversary-review RETROSPECTIVE

Per the iteration cadence in `SKILL.md` § "Iteration cadence": after every 3 reviews, spend 5 minutes on retrospective. Forcing function: every 4th invocation must include "retrospective check: <date of last entry>" in the disposition notes.

## 2026-05-08 — Migration baseline

**Reviews so far (4):**
1. `adv-2026-05-07-093400-smoke` — V1 bootstrap loop validation (planted weakness in /tmp scratch). All 5 gaps deferred (throwaway smoke files). Persona-feedback loop hardened: blind spots 5, 8, 11 sharpened from this run.
2. `adv-2026-05-07-124100-cohort-regen-fix` — retrospective review of already-shipped cohort-regen mitigation. 9 gaps surfaced; ALL 9 addressed in 35 net-new tests + 4 new tools (`cohort_validator.py`, `sample_cohort.py`, `check_cohort_invariants.py`, generator hardening).
3. `adv-2026-05-07-190947-doc-vs-impl` — doc-vs-impl audit. 10 gaps surfaced; ALL 10 addressed; meta-recommendation produced new `tools/check_doc_consistency.py` (5 mechanical rules + 14 tests).
4. `adv-2026-05-08-153427-file-issue-design` — design review of NEW file-issue subagent + subagents/ migration. 12 gaps surfaced; ALL 12 addressed in revised design before any implementation.

**True-positive rate (gaps that turned out to be real):** ~95% across reviews 2–4. Smoke (1) was a calibration run, not a true-positive measurement. Reviews 2–4 each surfaced gaps that would have shipped silently if discipline-only had been the gate.

**Persona stability:** Persona was hardened ONCE (post review 1) with 4 sharpening edits (blind spots 5, 8, 11 + 4 confirmed). No further edits across reviews 2–4. Persona is stable; reviewer outputs are consistent in shape.

**Calibration friction (V2 promotion signal):** None felt yet. Otter is not wishing Peng could tune the reviewer directly. V1 (local Agent) holding up.

**Recurring patterns across reviews:**
- **"Cross-doc inconsistency" was the META observation in BOTH reviews 3 and 4** — Otter's audit walks the *touched* files but doesn't grep for cross-references in *untouched* files. Mitigation landed: `tools/check_doc_consistency.py` rules `cohort_codes` + `apply_modes` + `subagent_paths_migrated` mechanically pin cross-doc consistency for specific high-risk classes.
- **"Happy-path crispness, failure-path muddiness"** — review 4's main META observation. Reviews 2 and 3 had the same flavor (Otter ships the success path, leaves edge cases / error cases under-specified). Persona blind spot 2 already covers this; the META observation suggests the reviewer is consistently surfacing it on first read.

**Trigger calibration:** No false-skip incidents. The "all `tools/` (no carve-out)" expansion 2026-05-07 (after Peng push-back) was the right call — review 3 covered `tools/run_experiment.py` template-only edits and rightly surfaced a real bug there.

**Action items (next 3 reviews):**
- Watch for the cross-doc pattern recurring; if it surfaces a 3rd time, consider adding more mechanical rules to `check_doc_consistency.py` rather than relying on the adversary alone.
- File-issue is now live (Phase 1 ship); first 3 invocations there will cycle the SAME retrospective discipline.

---

(Append next retrospective after `adv-` review 5 + 6 + 7. Date this entry: 2026-05-08.)
