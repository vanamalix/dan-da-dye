# Definition of done

"Done" means the change is proven, recorded and reviewable, not that it looks green. A change is done
when every line below that applies is true, with evidence (command and exit code, path) in the report.

## Checklist

1. **Tier known.** `pnpm exec vn-tier` run on the change; its requirements are met (TIERS.md).
2. **Tests.** New behaviour has a test. A bug fix has a test **seen failing before the fix** and passing
   after: revert or break the fix on purpose, show the failure, restore, show green. Report the
   mutation and the failing test name. No test was weakened, skipped or deleted to go green.
3. **Gates exit 0.** `pnpm exec vn-compliance --only <gates vn-tier printed>` (or `--quick`), run to a log
   and judged by exit code, never piped into `tail`/`head`. A SKIP names the missing toolchain and is
   not a pass. A failure in code you did not touch is reported, not hidden or "fixed" in passing.
4. **Docs.** Every doc sentence and UI string describing the changed behaviour is updated (grep for
   it). Docs describe what is **deployed**, not only what is implemented; where they differ, say so.
5. **CHANGELOG.** An entry under `[Unreleased]` (Added / Changed / Fixed / Security) in the same commit.
6. **ADR** when behaviour or architecture changes, or an owner decision is made (ADR-TEMPLATE.md).
7. **Records.** Owner decisions written to the decisions file in the same turn they are given; open
   owner questions added there (DECISIONS_PENDING.md); a recurring slip recorded (MISTAKES.md).
8. **Review evidence (T2 and T3).** An independent reviewer's ISBAR report exists; for T3 the commit
   carries the review trailer (TIERS.md).
9. **Privacy.** No secrets, real names, phone numbers, emails or personal details in the diff, logs or
   report. The privacy scan is the backstop, not the check.
10. **Committed by named path** (`git add <paths>`, never `git add -A` or `git add .`), by the
    orchestrator. Subagents report; they do not commit.
11. **Release state stated.** What is merged, what is deployed to staging or the preview (if the repo
    has one), what is in production, and which owner approvals are still needed (RELEASE.md).

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
