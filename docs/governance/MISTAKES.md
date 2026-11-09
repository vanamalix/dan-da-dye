# Mistakes notebook: rules and template

The mistakes notebook records **habits** (concepts) that produced a defect and will produce another
unless something structural stops them. It is not a bug list; git and the tests hold those.

## Rules

1. **Concepts, not events.** A new defect that matches an existing entry raises its `Occurrences` and
   adds an instance row; it does not get a new id. Prefer matching over minting.
2. **Severity is consequence**: `dangerous` = could weaken a safety or security control, lose a
   decision, leak private data or deploy unreviewed; `minor` = cosmetic or recoverable at once.
3. **Closed only by a structural fix**: a gate, test, hook, config rule or removed affordance, named
   with its path. "Be more careful" closes nothing; such an entry stays open or `mitigated`.
4. No personal details or secrets in entries.
5. Record it in the same change as the fix. An incident's postmortem (INCIDENTS.md) links its entry.

## Template

```
## M-<n> — <concept title>
- Status: open | resolved | mitigated    Severity: minor | dangerous
- Occurrences: <n>   First: <date>   Last: <date>
- The habit: <what we do that causes it>
- Correct principle: <what would have prevented it>
- Structural fix: <gate / test / hook / rule, with path>, or "none yet"
- Instances:
  - <date> — <what happened, where recorded>
```

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
