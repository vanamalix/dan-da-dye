# Report format: ISBAR

Every agent report, review and hand-off uses ISBAR. The reader decides from the first two lines;
everything after is evidence.

| Part | Contents |
| --- | --- |
| **I**dentify | Agent name and role, write scope actually used (or "read-only"). |
| **S**ituation | **Verdict first**, then counts. Each claim with its evidence: path (and line), command and **exit code**. |
| **B**ackground | The brief in one or two lines; what was tried or ruled out. |
| **A**ssessment | What the evidence supports and what it does not. End with `UNKNOWNS:` (what could not be checked, and why). |
| **R**ecommendation | Separate from the findings and never self-executed. Owner questions as `DECISION NEEDED:`. |

## Rules

- **Proof or refute.** A defect is reported only with a failing test that bites it, a concrete failure
  path (`input or state -> path -> wrong result`, real files and symbols), or a refutation ("suspected
  X; cannot happen because ..."). A suspicion without one of these is dropped. "Swept, came up dry" is
  a good report.
- **Exit codes, not summaries.** Quote the exit code of every gate you ran; never pipe a gate into
  `tail` or `head`.
- **Reviewers do not fix.** A reviewer's report says "fix NOT applied" and names who should apply it.
- Counts first, detail only where it proves something. No personal details or secrets.
- Never report: what the typechecker/tests already report, style preferences, speculative future risk,
  unrequested refactors, issues outside the changed lines.

## Example

```
I  code-reviewer (read-only), scope: server/worker/rules.ts
S  1 confirmed, 1 refuted, 0 unproven (dropped)
   CONFIRMED  <path:line> <input -> path -> wrong result>; proof: <test name>, fails on HEAD (exit 1)
   REFUTED    suspected X; cannot happen because <path:line>
B  brief: review the escalation change in <commit>; ran the server gate
A  the change is otherwise correct for the cases tested. UNKNOWNS: real-device timing not checked
R  fix is NOT APPLIED (outside my scope). Suggested: ...
   DECISION NEEDED: <owner question, if any>
```

## App rules

_None yet. This repo's own rules go here; everything above comes from @vanamalix/governance._
