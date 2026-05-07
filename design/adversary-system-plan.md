---
plan: adversary-test-review-system
status: active
owner: otter
created: 2026-05-07
last_check: 2026-05-07
---

# Adversary Test Review System — Setup Plan

Building an adversarial code reviewer (Rocky) that independently writes test cases for changes Otter makes, to defend against implementation bias. This plan exists because the setup will take many cycles; the plan keeps us on track without re-deriving context each turn.

**Stop directive in force:** no other runs / commits / changes outside this plan until the basic system is up.

## Resolved decisions (from 2026-05-07 conversation)

| # | Decision | Note |
|---|---|---|
| Scope | Tests only (no implementation review) | Tighter v1 = lower friction; expand later if useful |
| Adversary identity | Rocky (existing peer) | Not Beaver, not Agent tool, not new "Critic" peer |
| Rocky's project setup | Make corpus a primary project for Rocky | Same convention as Otter's two primaries |
| Encoding location | Repo-only — no agent memory dependence | Per "agent memory not reliable" |
| Communication mode | Async (delegation via Unix socket) | Round-trip latency acceptable |
| Failure mode | Fix Rocky if unreachable; never bypass review | Rocky is part of the system, not a courtesy |
| Adversary writes tests | Yes — Rocky writes the test code himself; Otter cannot edit | Avoids implementation bias |
| Authorship | Clean separation; Rocky's content = Rocky author, no co-author | Honest commit graph |
| Reject discipline | Negotiate with Rocky first (one round); if no alignment → escalate to Peng | Peer disagreement is a legitimate escalation trigger |

## Phases (walk one at a time)

### Phase 0 — Inspect Rocky's current state

- [ ] Read `~/.myclaw-rocky/` directory layout (CLAUDE.md, memory, current primaries)
- [ ] Identify any conflict with adding corpus as a primary
- [ ] Output: a short report on Rocky's current configuration
- [ ] Decision point with Peng: corpus as ADDITIONAL primary vs replace something vs different shape

**Exit criteria:** Otter has a clear picture of Rocky's setup; Peng has approved the integration shape.

### Phase 1 — Communication channel reliability test (5 round-trips)

Each round-trip spans turns (delegation is async). Sequence:

- [ ] Step 1.1 — Ping: "Reply with `PONG` plus your local time." → expect `PONG <ts>`
- [ ] Step 1.2 — Same ping again → reproducibility
- [ ] Step 1.3 — Small file payload: write `/tmp/proof-payload.txt` (100 lines), ask Rocky to reply with line count + first/last line
- [ ] Step 1.4 — Skill load probe: "Read `~/projects/oss-model-graph-break-corpus/skills/sweep_sanity_check.md` and reply with the count of STRICT invariants." Verify the number matches Otter's count
- [ ] Step 1.5 — Mini-adversary task: paste 3 lines of test code, ask Rocky to find one weakness

**Exit criteria:** all 5 succeed within 5 minutes each, no silent drops, no malformed replies. Each step's task_id and reply timestamp recorded in this plan's Decision Log section below.

**Failure handling:** any step fails → Phase 0.5 (fix Rocky) before continuing. Don't build on a broken channel.

### Phase 2 — Make corpus a primary project for Rocky

- [ ] Add to Rocky's local CLAUDE.md: `oss-model-graph-break-corpus` as primary project (Tiger does this — constitution-style governance applies)
- [ ] Add per-project skill trigger in Rocky's CLAUDE.md: "When asked for adversarial test review, read `~/projects/oss-model-graph-break-corpus/skills/adversary-test-review.md` first"
- [ ] Decide commit mechanism: **(a)** Rocky pushes himself (needs same sudo-override per local CLAUDE.md, requires Tiger setup) OR **(b)** Otter commits with `git commit --author="Rocky <rocky@myclaw>"` as v1 starting point
- [ ] Test: ask Rocky "what are your primary projects" — verify corpus appears
- [ ] Verify Rocky can read the corpus repo (file access probe)

**Exit criteria:** Rocky's CLAUDE.md updated, project context loads at session start, primary-project probe succeeds.

### Phase 3 — Encode the system in the repo

All artifacts live in the corpus repo (source of truth, version-controlled, agent-memory-independent):

- [ ] Write `skills/adversary-test-review.md` — persona, workflow, output format, no-edit discipline, gap-write-tests rule, negotiation+escalation workflow with format
- [ ] Write `tools/invoke_adversary.py` — invocation script (packages context, sends to Rocky's socket, records task_id)
- [ ] Write `tools/test_invoke_adversary.py` — tests for the invocation script (TDD discipline applies)
- [ ] Update `skills/test-sweep-changes/SKILL.md` — add Gate 6 (adversary review by Rocky)
- [ ] Update `docs/testing.md` — link to the gate
- [ ] Write `docs/adversary-channel-test.md` — Phase 1 runbook for future debugging (so a future agent can re-test the channel without re-deriving)
- [ ] Update `tools/README.md` — index entry for `invoke_adversary.py`
- [ ] Each artifact reviewed by Rocky as the adversary's first real test (meta-loop: Rocky reviews tests for the tool that invokes Rocky)

**Exit criteria:** all artifacts committed + pushed; Rocky has reviewed at least the `tools/invoke_adversary.py` tests.

### Phase 4 — Failure-mode handling (Rocky unreachable = fix Rocky)

- [ ] Define detection: ping `/api/identity` on Rocky's socket; no response in 10s = unreachable
- [ ] Encode recovery procedure in `docs/adversary-channel-test.md`:
  1. Check Rocky's process via `myclaw status`. If down → `myclaw restart rocky`. Verify identity probe.
  2. If restart doesn't fix → check Rocky's daemon log; escalate to Tiger if structural.
  3. If structural and Tiger can't fix → escalate to Peng.
- [ ] Encode workflow gate: if Rocky is unreachable, the commit is BLOCKED until he's back. No "review pending" escape hatch.
- [ ] Test: simulate Rocky-down, verify the recovery procedure works end-to-end

**Exit criteria:** failure handling encoded + dry-run-tested.

### Phase 5 — Live trial on a real recent change

- [ ] Pick a real recent change (suggestion: `tools/generate_cohort.py` and its tests, commit `cdb0ca4`)
- [ ] Put it through Rocky's adversarial review end-to-end via `tools/invoke_adversary.py`
- [ ] Observe Rocky's gaps + new tests
- [ ] Run augmented test file; verify Rocky's tests fail-then-pass against current implementation (or fail-only if real gap, then patch + pass)
- [ ] Walk the negotiate-or-escalate workflow on at least one item if any disagreement
- [ ] Document friction in this plan's Decision Log

**Exit criteria:** at least one full review cycle completed; system holds; encoding updated based on observed friction.

---

## Decision Log

(Entries appended as we proceed; freshest at top.)

### 2026-05-07 — plan created
Recording the agreed system per Peng's directive. No execution yet; awaiting green light to start Phase 0.

---

## Open items (to resolve before later phases)

- **Phase 2 sub-decision:** commit mechanism (a) Rocky pushes vs (b) Otter commits with `--author` for Rocky's content
- **Phase 3 detail:** exact persona text in `skills/adversary-test-review.md` (will draft in that phase)

---

## Currently at

**Phase 0** — awaiting Peng's green light to inspect Rocky's current state.
