# V1 → V2 Promotion: Adversary Review

V1 uses a local Agent (in-session subagent). V2 will swap to Rocky (separate MyClaw peer instance). This doc captures exactly what changes — the goal is for V1 artifacts to carry forward unchanged where possible.

## Promotion timing — Peng's call

There are no auto-trigger thresholds. Peng decides when V1 → V2 happens.

When Otter proposes V2, the proposal should include these inputs (data for her decision, not gates):

- Total invocations in `reviews_log.md`
- True-positive rate (gaps that turned out to be real)
- Persona stability (edits in recent reviews)
- Whether calibration friction has been felt — Otter wishing Peng could tune the reviewer directly without going through Otter

## What carries forward unchanged

| Artifact | V1 | V2 |
|---|---|---|
| `persona.md` | system prompt for local Agent | system prompt sent to Rocky in delegation body |
| `reviews_log.md` | append-only, in corpus repo | unchanged location; new entries continue from V1 history |
| `escalation_template.md` | format for Otter→Peng escalation | unchanged |
| Trigger conditions in `SKILL.md` | when to invoke | unchanged |
| Recording requirements (verbatim + hash) | enforced by Otter | enforced by Otter (Rocky returns the raw output, Otter logs it) |
| Disposition discipline (addressed/deferred/disputed) | Otter records | Otter records |
| Commit message line | enforced by Otter | unchanged |

## What changes

### 1. Invocation mechanism

V1:
```
Agent tool call:
  description: "Adversary review of <case_id>"
  subagent_type: general-purpose
  prompt: <persona.md> + "---" + <review-request>
```

V2:
```
curl --unix-socket /home/pengwu/.myclaw-rocky/delegation.sock \
  -X POST http://localhost/api/delegate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<persona.md> + --- + <review-request>",
    "from_name": "otter",
    "from_socket": "/home/pengwu/.myclaw/delegation.sock",
    "task_id": "adversary-review-<case_id>"
  }'
```

Result delivery:
- V1: synchronous tool result, parse immediately
- V2: async via GChat callback, lands in next-turn pending_context. Block on the result before commit (a small wake-from-DB cron may be needed if latency exceeds a turn).

### 2. Calibration channel

V1: Peng tunes reviewer by giving Otter feedback; Otter edits `persona.md`. Single-author bottleneck.

V2: Peng tunes reviewer by direct GChat to Rocky's space. Rocky updates his own MEMORY/persona files. Otter no longer mediates.

**Concrete consequence:** in V2, Otter does NOT edit `persona.md` directly anymore. Persona changes must be proposed via PR to Rocky for him to validate against his own review history. Add a `persona-changes/` directory with proposal docs; Rocky reviews and merges.

### 3. Memory layer

V1: persona.md is the only memory. Each review starts blank.

V2: Rocky maintains a `~/.myclaw-rocky/spaces/<his>/adversary-reviewer-memory.md` with accumulated patterns ("I've seen Otter false-claim X 3 times"). This memory feeds into Rocky's per-review context but NOT into the persona itself — keeping the persona stable while letting Rocky be situationally aware.

### 4. Audit trail

V1: only the corpus-repo `reviews_log.md`. Otter's word for what Rocky said.

V2: corpus-repo `reviews_log.md` PLUS Rocky's GChat space (every delegation IN, every reply OUT). Cross-check available — if Otter's logged "raw output" doesn't match Rocky's sent reply, mismatch is detectable.

### 5. Trigger enforcement

V1: Otter's local CLAUDE.md trigger entry is the only forcing function. Easy to "forget" with no external tripwire.

V2: ADD a periodic check (e.g., daily-brief job) that scans recent corpus commits for `adversary-review:` line; missing line on a qualifying commit pages Peng. Mechanical tripwire.

## Migration plan (when promotion criteria hit)

1. Tag V1 final state in repo: `git tag adversary-review-v1-final`
2. Open a `Phase-V2` plan.md under `experiments/` describing the migration
3. Onboard Rocky: install adversary-review skill in Rocky's space, point him at `persona.md`, set up his memory file
4. Run V1 and V2 IN PARALLEL for 5 reviews (both reviewers see the same change; compare outputs). This validates Rocky doesn't regress.
5. After parallel-run validation, switch trigger to V2-only. Tag V1 deprecated.
6. Update SKILL.md's "Step 2 — Invoke the adversary" section to V2 invocation pattern.
7. Notify Peng with parallel-run comparison results and migration commit.

## Why we didn't start with V2

Documented in conversation 2026-05-07 with Peng. Summary:
- Cross-agent communication overhead is real (protocol, calibration channel, restart drift handling).
- Validating the CONTENT of the adversary review (persona, trigger rules, what counts as useful finding) is independent of the invocation mechanism.
- Building V1 first lets us iterate on content with low setup cost; V2 swaps the implementation, keeps the validated content.
