---
name: record-mistake
description: Use when a defect or process slip has just been found in this repo and may be an instance of a recurring habit (an inert gate, a decision lost in chat, a piped exit code, a stale doc, an unreviewed safety change), or when a commit or report you are writing says "again" or "another instance of".
---

# Record a mistake

The notebook is `docs/records/MISTAKES.md`. It holds habits (concepts), not a list of bugs. Rules and
template: `docs/governance/MISTAKES.md`; if the slip caused an incident, also write a postmortem
(`docs/governance/INCIDENTS.md`).

## 1. Is it a mistake worth recording?

Yes if a different defect would come from the same habit next time. No for a one-off typo that a test
already catches; the commit message is enough.

## 2. Occurrence or new concept? Prefer occurrence

Read the open entries. Match on the underlying habit, not the surface. If an existing entry's
"correct principle" would have prevented this, it is an occurrence: raise `Occurrences`, move `Last`,
add an instance row (date, what happened, where it is recorded), and re-open it if it was resolved.
Mint a new `M-<n>` only when no existing principle applies.

## 3. Write it with the template at the top of the file

- Severity is about consequence: `dangerous` if it could weaken a safety control, lose an owner decision,
  leak private data, or deploy unreviewed. Otherwise `minor`.
- Instance rows carry no personal details, names, numbers, secrets or file contents.
- If the mistake also needs an owner call, add it to `docs/records/DECISIONS_PENDING.md`; do not
  settle it yourself.

## 4. Resolve only structurally

Set `Status: resolved` only when something other than memory now prevents it: a test that bites, a gate
in `governance.config.json` (run by `vn-compliance`), a hook, a pre-commit rule, or a removed affordance. Name it and its path in
"Structural fix". "Be more careful" is not a fix; leave the entry open or `mitigated`.

## 5. Land it

Record it in the same change as the fix. Commit by named path (never `git add -A`)
(`docs/records/MISTAKES.md` plus whatever gate you added).
