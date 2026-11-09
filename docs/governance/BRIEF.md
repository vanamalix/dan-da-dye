# Agent brief and orchestration rules

## Orchestrator rules

- The **main session is the orchestrator**: it talks to the owner, plans, splits work, assigns
  non-overlapping write scopes, reconciles reports, decides, commits, and is accountable for what lands.
- **One writer per file.** Concurrent agents never share a write scope. Shared manifests (package
  manifests, lockfiles, `.gitignore`, build files, deploy config) have one owner at a time.
- **Reviewers never fix.** Finding and fixing are different hands. T2 and T3 changes are reviewed by an
  agent that did not write them (TIERS.md).
- **No recursive delegation.** Subagents do not spawn subagents.
- **Small fan-out**: 2 to 4 concurrent agents, sharply scoped. Prefer inline work; spawn only for
  parallel, bounded, independent work. Cheaper models for mechanical work; stronger ones for
  security-sensitive review.
- **Report, don't guess.** Anything that is the owner's call goes to the decisions file
  (DECISIONS_PENDING.md) and is surfaced, never settled by an agent.
- Subagents do not commit, push or deploy. An agent that finds it must edit outside its scope
  **stops and reports**.

## Brief template

```
Goal:        <one sentence: the outcome, and why it matters>
Tier:        <T0-T3, from pnpm exec vn-tier --files <planned paths>>
IN scope:    <what to do>
OUT scope:   <what not to touch or decide; who else is working where>
Write scope: <exact paths / globs; nothing else may be written>
Read first:  <files and docs worth reading; what is already ruled out>
Acceptance:  <checks that prove it, with the commands; exit codes are the verdict>
             <for a fix: the test that must fail before and pass after>
Hard stops:  <no commit/push/deploy; no production; no secrets or personal data; ...>
Report:      ISBAR (ISBAR.md): verdict first, files, gate exit codes, UNKNOWNS, recommendation
Log:         claim at start, done/blocked at end (the repo's activity-log note command)
```

A brief you cannot write clearly means the task is not understood well enough to hand off.

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
