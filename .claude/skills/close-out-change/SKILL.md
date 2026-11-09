---
name: close-out-change
description: Use before calling a code, config or documentation change finished in this repo: before committing, reporting it done, or shipping it. Works from the change's risk tier (vn-tier) to the gates, tests-first proof, review, records, deploy order and committing by named path.
---

# Close out a change

"Done" must not mean "looks green". The full list is `docs/governance/DONE.md`; this is the working order.

## 1. Tier first

```bash
pnpm exec vn-tier --files <the paths you changed>     # or: pnpm exec vn-tier  (staged)
```

It prints the tier (T0 docs, T1 UI, T2 logic/API/tooling, T3 auth/schema/alerting/crypto/deploy/gates),
the files at that tier, and what the change needs: gates, tests-first, review, staging or rollout, owner OK.
Everything below follows from that output. Tiers and paths: `docs/governance/TIERS.md`,
`governance.config.json`.

## 2. Gates: read exit codes, not summaries

Run exactly the gates `vn-tier` printed, to a log, then read the status:

```bash
pnpm exec vn-compliance --only <gates> > /tmp/x-gates.log 2>&1; echo "gates exit=$?"
```

Never `pnpm test | tail`: a pipe returns the exit code of `tail`. Report the table. A gate that failed
in code you did not touch is a report, not something to fix or hide. A SKIP is not a pass; say which
toolchain was missing.

## 3. Make the test bite (T1 and above)

For a fix: revert it (or break it on purpose), confirm a test fails, restore, confirm green. Report the
mutation and the failure. For a gate or guard, run it against the case it exists to refuse.
Ask: what would have to be true for this to pass while the bad thing still happens?

## 4. Safety-control check

Did you touch anything this repo lists as a safety control (its `## App rules` sections)? Then the
change must not loosen it unless the owner decided so (`docs/governance/DECISIONS_PENDING.md`). If it is
a loosening and no decision exists, stop and add it there.

## 5. Records, in the same change

- `CHANGELOG.md` entry, and the feature tracker (`features.json`) status if a feature moved.
- An owner decision made in this conversation: written down now (DECISIONS_PENDING Decided), plus an
  ADR (`docs/governance/ADR-TEMPLATE.md`) if behaviour changes.
- Describe what is DEPLOYED, not only what is implemented; say where they differ.
- If a doc describes behaviour you changed, grep for every sentence and UI string that says it.
- An occurrence of a known habit? Use the `record-mistake` skill.

## 6. Review (by tier)

- T0: none. T1: self-review against DONE.md.
- **T2 and T3: independent review** by an agent that did not write the change; reviewers report, they
  do not fix.
- **T3**: the commit message needs a review trailer (`Reviewed-by: <reviewer>` or `Tier: T3 reviewed`),
  added only after the review happened. The commit-msg hook warns when it is missing.

## 7. Deploy order (if it ships; `docs/governance/RELEASE.md`)

Pushed commit; migrations (additive) before code; smoke checks after every deploy. With staging
(`rollout.mode` staging-soak): staging before production, T3 soaks on staging. Without staging
(production-live, or preview-then-production with the preview first): dry run, previous version id
recorded, and T3 must roll out gradually or behind a flag that defaults off (new alerting ships off, the production database backed up before a migration) and is watched through a cron run, rollback ready. Production deploys, migrations, secrets, Access changes and Zero Trust `apply` need the owner's
explicit OK in this conversation. Say which of these remain.

## 8. Commit

- Privacy: no real names, numbers, emails, health details or secrets in the diff (the scan is a
  backstop, not the check).
- Commit only by named path. Never `git add -A`, never force-push. Subagents do not commit; they report.
- Report in ISBAR (`docs/governance/ISBAR.md`) with the tier and each claim's command and exit code.
