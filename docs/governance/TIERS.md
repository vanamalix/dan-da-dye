# Risk tiers

Every change gets a risk tier. The tier decides which gates run, who reviews, how it reaches
production (a staging soak, or a watched production rollout where there is no staging), and whose OK it
needs before production. The tiers, the path globs and the approvals are data in
`governance.config.json`; `vn-tier` (from @vanamalix/governance, run as `pnpm exec vn-tier`) applies
them, so the rule and the tool cannot drift.

## How a change is tiered

- Each changed file takes the **highest** tier whose globs match it; a file no glob matches takes
  `tiers.default` (T2: unknown means logic until shown otherwise).
- Because the highest match wins, a broad glob in a higher tier **shadows** narrower globs below it
  (a `app/**` in T2 would turn every T1 `app/**/*.css` into T2). So the default tier lists **no paths**
  (`"paths": []`): T0 and T1 list their exceptions, T3 lists the risky paths, everything else is T2.
  Check a new glob with `pnpm exec vn-tier --files <a path it should match>`.
- The change takes the highest tier of its files. One T3 file makes the whole commit T3.
- Globs are repo-relative: `**` any depth (including none), `*` within one path segment, `?` one
  character, `{a,b}` alternatives. `**/*.md` matches `README.md` at the root.
- Run `pnpm exec vn-tier` (staged), `pnpm exec vn-tier --range A..B`, or `pnpm exec vn-tier --files ...`.
  It prints the tier, the files at that tier, and what the change needs.

## The tiers

| Tier | Covers | Gates | Tests first (bug fix) | Review | Release (no staging: `rollout`) | Release (staging: `staging`) | Production |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | Docs and text | privacy scan | no | none | none | none | normal deploy rules |
| T1 | UI and styling | + always-on gates and the component gates for touched paths (typecheck, tests, build) | yes | self-review against DONE.md | none or standard | none or deploy | owner OK where the app requires it |
| T2 | Logic, API, tooling | as T1 | yes | **independent** | standard | deploy | owner OK where the app requires it |
| T3 | Auth and sessions; schema and migrations; alerting, notifications and escalation; crypto and keys; deploy, infra and edge config (Workers config, Access, Zero Trust); the privacy scan and gates themselves | as T1 | yes | **independent**, with a commit trailer | **gradual** (watched) | **soak** | **owner OK, every time** |

The release columns depend on the repo's **rollout mode** (`rollout.mode` in the config; RELEASE.md):

| Mode | When | Each tier gives |
| --- | --- | --- |
| `production-live` | No staging: a change goes straight to production with the owner's OK (the template's default). | `"rollout": "none" \| "standard" \| "gradual"` |
| `preview-then-production` | No staging, but a separate preview deployment (e.g. a preview Worker) is smoke-checked first; `rollout.preview` names it. | `"rollout": ...` as above |
| `staging-soak` | A staging environment that mirrors production. | `"staging": "none" \| "deploy" \| "soak"` |

Without `rollout.mode` the mode is implied by the tiers (`staging` fields: staging-soak; `rollout`
fields: production-live). `vn-tier` prints the requirement in the words of the mode.

### What each requirement means

- **Gates**: `pnpm exec vn-compliance --only <list>` with the list `vn-tier` prints. The exit code is the
  verdict; a SKIP (toolchain missing, or a workspace whose dependencies are not installed) is reported,
  never counted as a pass. With `VN_GATES_STRICT=1` (the default when `CI=true`) a workspace skip is a FAIL.
- **Tests first**: a bug fix comes with a test that was seen failing before the fix and passing after
  (DONE.md). For a new guard or gate, run it against the case it exists to refuse.
- **Independent review**: the reviewer is not the writer, and reviewers never fix: they report
  findings with proof (ISBAR.md); the writer, or another agent with that write scope, fixes.
- **Review evidence (T3)**: the commit message carries a trailer naming the review:
  `Reviewed-by: <reviewer agent or role>` or `Tier: T3 reviewed`. The commit-msg hook (`vn-tier --hook`)
  checks each commit; `pnpm exec vn-tier --range origin/main..HEAD --strict` checks a branch before it
  is pushed or merged (exit 3 when a T3 commit lacks the trailer). `review.enforcement` in the config starts at `warn` (prints what is missing, never stops a
  commit); the owner may switch it to `block`.
- **Standard rollout** (no staging): the owner's OK, a dry run (`wrangler deploy --dry-run`), the
  deploy (to the preview first in `preview-then-production`, smoke-checked there) while the owner is
  available, the smoke checks against production, and the previous version id recorded so rollback is
  one command (RELEASE.md).
- **Gradual rollout (T3, no staging)**: as standard, and the change **must** go out either as a
  gradual rollout (`wrangler versions upload`, then `wrangler versions deploy` at a small percentage,
  raised after the checks) or behind a flag that defaults off. The production database is backed up
  before a migration; new alerting or notification behaviour ships off; it is deployed while the
  owner is available, and watched through at least one run of every cron trigger it touches before
  the rollout reaches 100% or the flag is turned on, with the rollback ready. A repo may add to this
  line (`rollout.gradualNote`), never relax it.
- **Staging deploy** (staging mode): the change runs on staging and passes the smoke checks (RELEASE.md)
  before production.
- **Staging soak (T3, staging mode)**: deployed to staging, smoke checks pass, and it stays there until
  every scheduled trigger it touches has run at least once without error before it is promoted.
- **Owner OK**: the approvals listed for the tier in the config (`approvals`) need the owner's explicit
  OK in the conversation, each time. Agents prepare, dry-run and report; they do not run these.

### Review regardless of tier

Independent review is not only a T2/T3 rule: a change to safety-critical behaviour is reviewed by an
agent that did not write it **whatever tier its files land in**. Each app lists those areas in its
App rules.

### Changing the tiers

Editing `governance.config.json`, the schema, the scan, the gate runner or the hooks is itself T3. A
change that moves a path to a lower tier states why in the commit message and is reviewed.

## App rules

- T3 here is the R2 publisher and its CLI wiring (`src/dandadye/export.py`, `cli.py`), `scripts/setup.sh` (it refuses
  a data root inside the repo), `.gitignore` (it keeps pages out of git), the gates and the repo controls. Everything in `governance.config.json` `tiers` is the source of truth.
- There is no staging. The pipeline runs locally; the only thing that reaches production is `dandadye upload`.
- T2 and T3 still need an independent reviewer who does not fix.
