# Release: how a change reaches production

A release ships one **pushed** commit to production with the owner's OK. Nothing is deployed from a
working copy. How it gets there is the repo's **rollout mode**, `rollout.mode` in
`governance.config.json` (TIERS.md); `vn-tier` prints the tier's requirement in the words of the mode:

- **`production-live`**: no staging. The commit goes straight to production, dry-run first, smoke-checked
  straight after, with the rollback ready; risky (T3) changes must roll out gradually or behind a flag
  that defaults off, and are watched.
- **`preview-then-production`**: no staging, but a separate preview deployment (`rollout.preview`, e.g.
  a preview Worker on `workers.dev`) gets the same commit first and passes the smoke checks there.
- **`staging-soak`**: a staging environment that mirrors production; the commit runs there first and T3
  soaks there.

Steps 1-3 and 5-7 are the same in every mode; step 4 depends on it.

## Runbook

1. **Commit and push first.** The change is committed by named path and pushed; CI (`check`) is green
   for that commit (CI's pnpm version comes from `packageManager`, DEPENDENCIES.md). Note the commit SHA: it is what you deploy and what you roll back from.
2. **Deploy from a clean worktree of the pushed commit**, never from the working copy (other agents'
   uncommitted edits would ship with it):
   ```sh
   git fetch origin && git worktree add --detach /tmp/release-<sha> <sha>
   cd /tmp/release-<sha> && pnpm install --frozen-lockfile
   ```
   Remove it afterwards: `git worktree remove /tmp/release-<sha>`.
3. **Migrations before code** (staging first, where there is one). Apply database migrations, then
   deploy the code that needs them; take a backup of a production database first. Migrations are **additive** (new tables, new nullable columns, new
   indexes); a destructive change (drop, rename, narrowing) is split into an additive release, a
   code release that stops using the old shape, and a later cleanup migration with its own review.
4. **Before production, by mode** (the tier's `rollout` or `staging` value):
   - `production-live`: `wrangler deploy --dry-run`, and record the **current version id**
     (`wrangler deployments list`) so rollback is one command. `gradual` (T3): a gradual rollout
     (`wrangler versions upload`, then `wrangler versions deploy` at a percentage) or the risky
     behaviour behind a flag that defaults off is **required**; new alerting or notification behaviour
     ships off; back up the production database before a migration (step 3).
   - `preview-then-production`: as production-live, and first deploy the same commit to the preview
     (`rollout.preview`) and run the **smoke checks** (below) there.
   - `staging-soak`: deploy to staging and run the **smoke checks**. For T3, **soak** (TIERS.md): it
     stays on staging until every scheduled trigger it touches has run once without error.
5. **Owner OK for production** (the approvals the tier lists in `governance.config.json`), explicit, in
   the conversation, for this release. Agents prepare the exact commands and stop here.
6. **Production: migrations, then code**, from the same worktree and commit, while the owner is
   available. Run the smoke checks against production at once. Without staging, a `gradual` (T3)
   change is **watched** through at least one run of every cron trigger it touches before the release
   is called done (and before a gradual rollout goes to 100%); any doubt is a rollback, not a fix
   forward.
7. **Record**: CHANGELOG, features registry status, and the report say what is now in production
   (DONE.md item 11).

### Smoke checks (every deploy: staging or preview, and production)

- Browser paths are behind the identity gate: `curl -s -o /dev/null -w '%{http_code}\n' https://<host>/`
  returns **302** to the identity provider's login, never 200.
- Device and API paths that bypass the gate reach the app itself: a request without credentials gets
  the app's own refusal (401/403 from the app), **not** a 302 to the login page.
- Public pages return **200**.
- Scheduled jobs: the next scheduled run completes without error.

### Rollback

- **Code rollback does not undo migrations.** That is why migrations are additive: the previous code
  must still run against the new schema. Never "roll back" by dropping what a migration added.
- Workers: `wrangler rollback [<version-id>]` (the id recorded in step 4; add `--env staging` for
  staging) returns to an earlier deployed version; during a gradual rollout, `wrangler versions deploy`
  the previous version back to 100%; or redeploy the previous good commit from a clean worktree.
- **Secrets take effect immediately** when set (`wrangler secret put`), with no deploy step and no
  rollback; changing a production secret is a production change and needs the owner's OK.
- After a rollback, run the smoke checks and open an incident record if users were affected
  (INCIDENTS.md).

## App rules

- "Production" is one action: `dandadye upload --yes`, which writes to the R2 bucket `dandadan-manga` read by the
  dandadan reader. Owner OK in the conversation, every time. Without `--yes` it only prints the plan: run that first.
- Page keys are immutable and the manifest goes last, so a failed upload leaves the reader on the old manifest.
- `--allow-removal` drops published chapters from the manifest: owner OK for that specific run.
- This repo deploys no Worker and changes no DNS, route or Access setting; those belong to the dandadan repo.
