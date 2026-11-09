# Incidents

An incident is anything that broke, leaked, alerted wrongly, failed to alert, or could have: a bad
deploy, a lost decision, a secret in a commit, a gate that passed while the bad thing happened.

## During

1. **Stabilise first**: roll back (RELEASE.md), disable the feature, or rotate the secret. Production
   actions still need the owner's OK unless waiting would itself cause harm; then act, and tell the
   owner at once what was done and why.
2. Keep a timeline as you go (UTC times, what was seen, what was done). No personal data in it.
3. Do not destroy evidence (logs, the bad commit, the failing request) while fixing.

## After: blameless postmortem

Blameless: describe what the system and the process allowed, not who slipped. "Someone should have
been more careful" is never a cause and never a fix.

```
# Incident <YYYY-MM-DD>-<slug>
- Status: open | fixed | closed        Severity: low | high | critical
- Detected: <time, how (alert, user, review, gate)>   Resolved: <time>
- Impact: <what users or data were affected, for how long; no personal details>

## Timeline (UTC)
- <time> <what happened / what was done>

## What happened
<the mechanism: input or state -> path -> wrong result, with file paths and commits>

## Why it was possible
<which gate, test, review or rule did not catch it, and why>

## Structural fix
- <the gate / test / hook / config / removed affordance that now prevents it, with path and commit>
- Proof it bites: <the check run against the original failure, exit code>

## Follow-ups
- <owner decisions needed (DECISIONS_PENDING), docs to update, MISTAKES entry id>
```

## Rules

- Every incident ends with a **structural fix**: something other than memory that now prevents the
  same class of failure. Until one exists, the incident stays open.
- If the failure came from a habit that can recur, record it in the mistakes notebook (MISTAKES.md) as
  a new concept or a new occurrence, and link the incident.
- Secrets: rotate first, then clean up; report which secret (by name, never value) and where it leaked.

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
