# subagents/

This directory holds **sub-agent setups** — packaged personas + invocation procedures + audit logs designed to be spawned as fresh, stateless Claude sessions via Claude Code's `Agent` tool.

Distinct from:
- **`skills/`** — pure-markdown playbooks for Otter (no persona, no log, no spawned session)
- **Peer agents** (Rocky, Beaver) — separate MyClaw daemon processes with their own persistent state

## What lives here

| Sub-agent | Purpose |
|---|---|
| `adversary-review/` | Independent skeptical reviewer of validator code, sweep cases, and design docs. Spawned BEFORE commits to find gaps Otter would miss. |
| `file-issue/` | Two-mode agent (Mode A adversary framing review + Mode B body assembler) that gates every GitHub issue Otter files. Discipline anchored on pytorch/pytorch's contribution standards + Peng's 4 issue criteria. |

## Required files for every `subagents/<name>/` setup

The contract every sub-agent directory follows:

| File | Required? | Purpose |
|---|---|---|
| `SKILL.md` | required | Activation contract: when to invoke + step-by-step procedure + authority gate. Filename is `SKILL.md` for compatibility with the Claude Code harness's plugin discovery (`~/.claude/plugins/cache/**/SKILL.md`). |
| `persona.md` | required | The system prompt for the spawned Claude. If the agent has multiple modes, they live here in one file, switched by an invocation `MODE:` directive. |
| `invocations/` | required | Append-only directory of per-case files. One file per invocation, named `<case_id>.md`. **No shared mutable log file** — eliminates concurrent-append corruption. |
| `invocations_log.md` | generated | Aggregated index built from `invocations/*.md` by `tools/build_invocations_log.py`. Source of truth is the per-case files; this is for human scanning. |
| `templates/` | optional | Output templates the persona fills (issue bodies, PR descriptions, etc.) |
| `<topic>-checklist.md` | optional | Discrete checklists the persona references (e.g. mre-checklist, calibration) |
| `validation-recipes/` | optional | Input recipes Otter follows BEFORE invoking |
| `<RETROSPECTIVE>.md` | optional | Iteration cadence notes, lessons learned, persona-revision history |
| `PREREQ-CHECK.sh` | optional | Smoke-test all dependencies before first use |

Mechanically enforced via `tools/check_doc_consistency.py` rule `subagent_required_fields`.

## Case ID convention

Case IDs use a **sub-agent prefix** to avoid collision when multiple sub-agents log on the same minute, and to be greppable in footer markers + cross-references:

- `adv-YYYY-MM-DD-HHMMSS-<short-slug>` — adversary-review
- `file-YYYY-MM-DD-HHMMSS-<short-slug>` — file-issue

The prefix appears in the case_id everywhere — directory paths, log entries, posted-issue footer markers, `--via-skill` lookups.

## How to invoke a sub-agent

1. Read `subagents/<name>/SKILL.md` for the procedure
2. Prepare input files at `/tmp/<sub-agent>-<case_id>-*.md`
3. Spawn the sub-agent: call Claude Code's `Agent` tool with `subagent_type="general-purpose"` and the prompt = `Read(persona.md)` content + the case context. Each sub-agent's `SKILL.md` shows the exact prompt structure.
4. Receive the structured output, log it as a per-case file under `invocations/`
5. Regenerate the aggregate: `python3 tools/build_invocations_log.py subagents/<name>/`
6. Act on the verdict per the `SKILL.md` authority gate

## How to audit (Peng's view)

For any sub-agent invocation:
- `cat subagents/<name>/invocations/<case_id>.md` — verbatim sub-agent output + sha256 + disposition
- The persona git sha is logged → reproduce exactly what the sub-agent saw via `git show <sha>:subagents/<name>/persona.md`
- For file-issue: posted GitHub issues carry a footer marker `<!-- via subagents/file-issue case_id=<case_id> -->`. Match marker → log row.

For the audit chain to be load-bearing, posting tools (`tools/file_issues.py` for file-issue) enforce `--via-skill <case_id>` at the CLI level (argparse `required=True`). Without the flag, the tool refuses to post.

## When to add a new sub-agent

Add one when a recurring task benefits from an **independent reviewer's perspective** OR a **specialist persona**, AND the task has **structured input/output that can be templated**.

If a task is one-off, write a one-shot script. If the persona would be vague or reflects general knowledge already in the model, write a `skills/` playbook for Otter to follow directly. Sub-agents are for cases where the JUDGMENT benefits from being stateless and adversarial to Otter's instincts.

## Architecture rationale

See `design/sub-agent-architecture.md` for the full design rationale, including why sub-agents are separate from skills, why Mode A/B split for file-issue, and the audit-chain trust model.

## History

- **2026-05-08** — Directory created. Migrated `skills/adversary-review/` here. Scaffolded `file-issue/`. See `MIGRATION.md` for the migration record.
