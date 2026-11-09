# Decisions pending: rules and template

The decisions file holds questions only the owner can answer, and the record of the answers. Agents
never settle these.

## Rules

- An agent that reaches an owner question **stops, writes it to the decisions file**, and surfaces it
  in its report as `DECISION NEEDED:`. Interim behaviour is the safe, already-shipped behaviour.
- When the owner answers, **write the answer to the file in the same turn** (move it to "Decided" with
  the date and a pointer to where it is implemented), and add or amend an ADR if behaviour changes. A
  decision that exists only in a chat transcript is lost.
- Say what the system should do, not who it is for. No personal details.
- Engineering questions that only need judgement (naming, structure, library choice within policy) are
  the orchestrator's call, not the owner's; record significant ones in an ADR instead.

## Template

```
### P-<n> — <question in one line>
- Raised: <date>, by <agent or session>
- Why it matters: <consequence of each answer>
- Options: A ... / B ... (recommendation, if any, marked as such)
- Interim behaviour shipped: <what happens until answered>
- Blocks: <work waiting on this>
```

Decided table row: `| <date> | <decision in one line> | <record: ADR or doc path> |`

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
