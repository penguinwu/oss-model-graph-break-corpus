# Adversary Review — Peng Escalation Template

Use ONLY when the adversary reviewer has flagged a gap that you (Otter) want to dispute. Disputed gaps block the commit until Peng resolves.

Send via your own GChat space (`spaces/AAQANraxXE4`) using `--as-bot`. Do NOT send to any other space.

## Format

```
🚨 adversary-review escalation: <case_id>

REVIEWER VERDICT: <verdict>
REVIEWER GAP IN DISPUTE: [SEVERITY: ...] <gap description verbatim from reviewer output>
REVIEWER WHY_IT_MATTERS: <verbatim>

MY POSITION: <one paragraph — why I believe this gap is not a real risk OR is out of scope for this commit>
EVIDENCE: <link to existing test, prior commit, design doc, or other concrete evidence>

PROPOSED RESOLUTION:
- (a) override — commit as-is, mark gap as "disputed-resolved-by-Peng" in log
- (b) defer — file as open-loop with rationale, commit as-is
- (c) modify — adjust scope per discussion (specify what changes)
- (d) accept gap — withdraw dispute, address gap before commit

ASK: which option, or other?

CASE FILE: subagents/adversary-review/invocations/<case_id>.md
PERSONA SHA: <git rev of persona.md at invocation>
RAW REVIEWER OUTPUT (for reference): <link to log entry>
```

## Rules

- One escalation per case_id at most. If multiple gaps are disputed, batch them in the same escalation.
- Wait for Peng's resolution before committing. No silent timeout.
- After Peng resolves, append her decision verbatim to the log entry's disposition section.
- If Peng's resolution sets a precedent, propose a `persona.md` edit to encode the lesson (so future reviews don't re-raise the same gap class).
