# Architecture decision records

Write an ADR when a change alters behaviour users or other code rely on, changes the architecture,
adds a data store, third party or bypass path, or records an owner decision. Short is fine; the
"Consequences" section is the part people read later.

## Template

```
# ADR-NNNN: <decision in a few words>

- Status: Proposed | Accepted (<date>, <who decided>) | Superseded by ADR-NNNN | Rejected
- Tier: T0 | T1 | T2 | T3        (TIERS.md; most ADRs are T2 or T3)
- Context owner: <role>

## Context
<the problem and the forces: constraints, what went wrong, what the owner asked for>

## Decision
<what we will do, specifically enough to check against the code>

## Alternatives considered
<each option and why it lost; a rejected option is useful later>

## Consequences
<what gets easier, what gets harder, new rules other code must follow, what to revisit and when>

## Verification
<the tests, gates or checks that show the decision is implemented>
```

## Rules

- Numbers are never reused; check the folder for the next free number before writing (and fix the
  "next number" pointer where the repo keeps one).
- An Accepted ADR is not edited to change its decision: write a new ADR that supersedes it and update
  the old one's Status line only.
- An owner decision is recorded the same turn it is given: in the decisions file, and in an ADR if it
  changes behaviour.
- No personal details, secrets or contents of private data in an ADR.

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
