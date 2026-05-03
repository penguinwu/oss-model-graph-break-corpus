---
plan: Watchdog auto-restart — robust restart logic research
status: draft
owner: Otter
created: 2026-05-03
last_check: 2026-05-03
forcing_function: Manual review by Peng
---

# Watchdog Auto-Restart — Robust Restart Logic (Research)

> **Status: research output — no implementation.** Peng's directive (2026-05-03 09:31 ET): "While the sweep is running, do a research on how to restart in a robust way. And certainly if restart fails too fast or not making meaningful progress, we should stop restart and notify people for human intervention."

## Problem statement

Today the corpus has TWO restart mechanisms with different shapes:

1. **`sweep/sweep_watchdog.py`** — invoked by `sweep-watchdog` cron every 10 min (`interval=600s`). Has `MAX_RESTARTS = 3`. On dead-but-incomplete sweep, calls `restart_sweep(state, output_dir)` which `Popen`s `run_sweep.py --resume` with the original args.
2. **Per-sweep wake-crons** (e.g. `nightly-sweep-restart-health`, `nightly-sweep-restart-completion`) — one-shot agent invocations that *diagnose-and-alert* but do NOT auto-restart.

The disabled-auto-resume incident Peng remembers: at some prior moment, restart spawned → quickly failed → watchdog restarted → quickly failed → loop. Even bounded by `MAX_RESTARTS=3`, three rapid failures cost ~30 min and produced minimal signal because each attempt died at startup before doing meaningful work. The fix at the time was to disable auto-resume entirely; today's gap is that no auto-resume logic is wired into the per-sweep wake-crons (they only diagnose).

## Why the existing `sweep_watchdog.py` design is fragile

Looking at `sweep/sweep_watchdog.py`:

| Property | Current | Problem |
|---|---|---|
| Restart cap | 3 | OK as a backstop, but doesn't prevent rapid burn-through |
| Time between restarts | 0 (immediate on watchdog firing) | Watchdog runs every 10 min → minimum gap is 10 min, not bad — BUT no exponential backoff, every attempt is treated equally |
| Progress requirement | None | If attempt N completed 0 new work items vs attempt N-1, restart anyway |
| Failure classification | None | Crashed-at-startup vs crashed-mid-run vs stalled all treated identically |
| Escalation surface | `print()` to stdout (GChat one-liner) | Easy to miss; no "page Peng" path |
| Restart shape | Same output dir + same state file | Stale state from prior crash can mislead the next attempt |
| Idempotency assumption | `--resume` resumes from checkpoint cleanly | Mostly true but unvalidated; no canary check after restart |

## Research: industry patterns for crash-loop control

### 1. Kubernetes CrashLoopBackOff (most relevant analogy)

K8s tracks restart count per container. Backoff is *exponential, capped*:
- 0s → 10s → 20s → 40s → 80s → 160s → ... → cap 5 min
- After each successful run lasting >= 10 min, reset the backoff counter
- No hard cap on restarts — the system keeps trying *forever*, but the backoff prevents resource burn
- Pod stays in `CrashLoopBackOff` state visibly; cluster operators can see and intervene

**Key insight:** the backoff isn't there to "give up" — it's there to *slow down* enough that humans can intervene before resources are exhausted. The "give up" decision is left to humans (or a separate eviction policy).

### 2. systemd `Restart=on-failure` + `RestartSec` + `StartLimitBurst`

systemd combines:
- `RestartSec=Ns` — pause between restarts
- `StartLimitBurst=K` + `StartLimitInterval=Tsec` — max K restarts in T seconds
- If burst exceeded: enter `failed` state, optionally trigger `OnFailure=` notification unit

**Key insight:** burst-window cap (K-in-T) is more useful than total-count cap. "3 restarts ever" is too restrictive over a long-lived job; "3 restarts in 30 min" catches loops without limiting healthy long-term recovery.

### 3. Supervisord `startsecs` + `startretries`

- `startsecs=10` — process must run >= 10 sec to count as "successfully started"
- `startretries=3` — up to 3 attempts to reach `startsecs`
- After exhausting retries, marked FATAL; manual intervention required

**Key insight:** the *start gate* (startsecs) prevents "fast-crash" loops. If the process can't survive 10 sec, it's a real bug, don't waste cycles on retries.

### 4. AWS Auto Scaling cooldowns

- After scaling event, refuse further actions for `cooldown` period
- Health-check grace period prevents flapping during steady-state
- CloudWatch alarms route to humans if cooldown loop persists

**Key insight:** explicit cooldown after each restart prevents the watchdog from firing twice in quick succession even if the *cron* would.

### 5. Erlang/OTP supervisor — "intensity" + "period"

- A supervisor restarts children with `intensity` (max failures) in `period` (seconds)
- If intensity exceeded, the supervisor itself fails up to its own supervisor
- Failure escalates UP the tree until something can handle it

**Key insight:** layered escalation. The watchdog itself can "give up" and escalate to the human as a higher-level supervisor.

## Failure classification (what to ignore vs what to retry vs what to escalate)

Not all failures deserve the same response. Proposed taxonomy for the corpus sweep:

| Category | Detection signal | Recommended action |
|---|---|---|
| **Transient infrastructure** — VM hiccup, cuda race, network blip | Sweep died with NO Python exception in log; system load spike; oomd kill in dmesg matching the sweep's slice | Retry with backoff (most likely to succeed on retry) |
| **Resource exhaustion** — OOM at `create_model`, GPU memory full | Python `OutOfMemoryError` / `RuntimeError: CUDA out of memory` in log; cgroup `memory.events.oom_kill` incremented | Retry IS okay if subsequent attempts succeed (different model order due to checkpoint advance), BUT if same model OOMs twice in a row → blacklist model + retry |
| **Code/data bug** — fails immediately at startup, every time | Sweep dies < 60 sec after start, AND completed=0 items, AND startup-stack has Python traceback | DO NOT retry — escalate to human. This is what created the pathological loop in the first place. |
| **Stalled / deadlock** — sweep alive but no log progress | Process alive, log-mtime > N min stale | Kill + retry once with fresh process; if stalls again, escalate |
| **Done** — completed=total | Process exited normally + completed=total | Move to "post-process" (per-source stats, etc.) |
| **External event** — daemon restart, VM reboot | Boot time after sweep start | Auto-resume normal (this is the common "nothing's wrong, just lost the process" case) |

The first observable signal that distinguishes "transient" from "code bug" is **time-to-death + progress-since-restart**:
- Died < 60 sec after start + 0 progress = code bug → STOP + escalate
- Died > 60 sec after start AND made progress = transient → retry with backoff

## Proposed restart logic

A composite of the patterns above, scoped to the corpus sweep but reusable:

### Per-sweep state (in `sweep_state.json`)

Add fields:
```json
{
  "restart_history": [
    {"started_at": "2026-05-03T00:00:48Z", "ended_at": "2026-05-03T00:19:59Z",
     "duration_s": 1151, "completed_at_start": 0, "completed_at_end": 126,
     "progress_made": 126, "exit_reason": "scope_cleanup"},
    {"started_at": "2026-05-03T06:29:24Z", "ended_at": null,
     "duration_s": null, "completed_at_start": 126, ...}
  ],
  "restart_budget": {"max_restarts_per_24h": 5, "min_progress_per_restart": 5}
}
```

`restart_history` enables ALL the heuristics below.

### Decision tree for the watchdog (when sweep is detected dead)

```
1. Was the sweep complete? (completed >= total)
   YES → mark done, post final summary, EXIT
   NO  → continue

2. Pull restart_history[-1] (most recent attempt). Compute:
     duration_s = ended_at - started_at
     progress_made = completed_at_end - completed_at_start

3. Hard escalation gates (any → STOP + page human):
   a. duration_s < 60 AND progress_made == 0
      → Sweep died at startup, made no progress. Code/data bug, NOT transient.
      → Page Peng with: log tail, last 5 attempts' durations + progress, suspected category.
   b. len([r for r in restart_history[-3:] if r.progress_made == 0]) >= 3
      → 3 consecutive zero-progress attempts. Hard stop.
   c. count(restart_history within last 24h) >= 5
      → Burst budget exhausted. Hard stop.

4. Soft delay gate (slow down, don't escalate yet):
   - last_restart_at < (now - max(120, 30 * 2^restart_count_in_hour))
     → Backoff: 2 min, 4 min, 8 min, 16 min, 32 min, cap 60 min
     → If we're inside the backoff window, EXIT silent (try next watchdog tick)

5. Validate the restart will be useful:
   a. Verify checkpoint file exists and has progress > 0
   b. Verify --resume args are still valid (env, paths, model registry hasn't changed shape)
   c. Quick canary: dry-run mode that loads the next model and does a tiny forward pass
      → If canary fails, the restart will fail too — escalate instead of retrying

6. Restart with --resume. Update restart_history with started_at + completed_at_start.
   Post one-line summary: "Sweep auto-restarted (attempt N, last died at K/M after Ds, made +Pcases)."
```

### Specific anti-loop behaviors

**Progress requirement.** A restart only "counts as a real attempt" if it ran at least 60 sec AND completed at least 1 new work item. Failed-at-startup attempts don't burn the budget — they trigger immediate escalation instead.

**Time-windowed budget.** Max 5 restarts per 24h is generous (allows recovery from a few VM hiccups) but bounded enough to catch slow loops. Combined with backoff: even if all 5 restarts happen, they span ~3+ hours due to backoff.

**Backoff resets on healthy run.** After any attempt that runs > 1 hour OR completes the sweep, reset `restart_count_in_hour` to 0. This way a sweep that has run for days doesn't carry forward stale restart history.

**Escalation has a clear shape.** When auto-restart gives up:
1. Post `[🦦 Otter] ⚠️ HUMAN ATTENTION: nightly-sweep cannot recover` with details to Peng's GChat space (`--as-bot`)
2. Write `sweep_state.json` with `status: "human_intervention_required"` (don't use `failed` — distinguish "we tried" from "we gave up")
3. Skip subsequent watchdog ticks (silent) until status changes — prevents alert spam

### What human intervention looks like

When the watchdog escalates, Peng (or Otter on prompt) should:
1. Read `restart_history` to see the failure pattern
2. Read `sweep.log` tail of the last failed attempt
3. Decide: fix-and-resume (`--resume` after patch), restart-fresh (rm `sweep_state.json`), or abort (mark this nightly skipped)
4. Reset `status` field to clear human_intervention_required → next watchdog tick continues normally

A `tools/sweep_recover.py --diagnose` helper would make this faster but isn't strictly required — `cat sweep_state.json | jq .restart_history` is enough for hand triage.

## What this DOES and does NOT solve

**Solves:**
- ✅ Pathological tight loops (progress-required gate + backoff)
- ✅ Crash-at-startup → infinite retry (code-bug-detection escalation)
- ✅ Silent failure (state.status="human_intervention_required" + GChat alert)
- ✅ Slow loops over days (24h burst budget)
- ✅ Resource burn (exponential backoff caps frequency)

**Does NOT solve:**
- ❌ The sweep's actual launch mechanism — that's the `(b)` we already shipped (cron prompt patched)
- ❌ Per-model timeout vs per-sweep timeout — separate orthogonal concern
- ❌ GPU OOM during compile — model-level retry/skip is in `worker.py`, not watchdog territory
- ❌ Network/auth recovery for transformers PyPI — pip's own retry handles this

## Implementation sketch (NOT for tonight — for review)

Three changes, in order:

1. **`sweep/sweep_watchdog.py`**: replace the simple `restart_count >= MAX_RESTARTS` check with the decision tree above. Add `restart_history` to state schema; backfill on first run. ~80 LOC.
2. **`sweep_state.json` schema**: add `restart_history` (list of attempt records) + `restart_budget` (config). Document in module docstring.
3. **One-shot wake-crons** (per-sweep watchdogs like `nightly-sweep-restart-health`): change from "diagnose-only" to "diagnose + delegate to sweep_watchdog.py for restart decision". Keep diagnose-only as a feature flag for sweeps where humans want full manual control.

Estimated implementation: 1–2 hours focused. Test plan: synthetic sweep that exits with each failure category (startup-crash, mid-run-die, OOM, complete) and verify the watchdog's response matches the table above.

## Open questions for Peng

1. **Backoff schedule** — is `2/4/8/16/32 min cap 60min` reasonable, or do you want something more aggressive (e.g., cap at 15 min so we don't lose half a sweep window during a bad night)?
2. **Burst budget** — `5 restarts per 24h` — too tight, too loose, or the right shape?
3. **Escalation channel** — GChat to your space `--as-bot` is the default. Want CC to feedback space (`spaces/AAQABmB_3Is`) for visibility? Want voice-alert / phone notification for hard-stop cases (overnight when GChat is silent)?
4. **Should `tools/sweep_recover.py --diagnose` be built** as part of the implementation, or is hand-triage with `jq` fine?
5. **PT 2.10 venv** — the existing `sweep-watchdog` cron uses `~/envs/torch210/bin/python` to run the watchdog itself. That's odd (the watchdog doesn't need PT 2.10 specifically). Worth migrating to `~/envs/torch-nightly-cu126/bin/python` or even `/usr/bin/python3` since the watchdog only needs stdlib?

## References

- K8s CrashLoopBackOff: https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#container-restart-policy
- systemd Restart= directive: https://www.freedesktop.org/software/systemd/man/systemd.service.html#Restart=
- supervisord startsecs: http://supervisord.org/configuration.html#program-x-section-settings
- Erlang/OTP Supervisor intensity: https://erlang.org/doc/design_principles/sup_princ.html
- Long-running jobs recipe: `~/.myclaw-shared/recipes/long-running-jobs.md`
