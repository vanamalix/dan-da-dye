# dan-da-dye

Reference-guided manga colourisation pipeline (Python, Apple Silicon). It is dandadan's colouring pipeline and the one
R2 publisher for the reader; the reader is the separate `vanamalix/dandadan` repo. See `README.md` for the stages.

## Rules

- **Never commit pages, coloured output, official art or model weights.** They live in the data root, outside the repo.
  Test images are drawn in `tests/conftest.py`.
- **`dandadye upload --yes` is a production action:** owner OK in the conversation, every time. Plan first (no `--yes`).
- Never write into the hub or any other repo. No secrets, `.env`, personal or health data in git.
- Packages are private, on GitHub Packages only. Never the public npm registry.

## Governance (`@vanamalix/governance` 0.4.0)

Config: `governance.config.json`. Docs: `docs/governance/` (this repo's rules are under each file's "App rules").

- Set up once per clone: `NODE_AUTH_TOKEN="$(gh auth token)" pnpm install`, `pnpm setup:dev`,
  `git config core.hooksPath .githooks`.
- Tier every change first: `pnpm exec vn-tier --files <paths>`.
- Gates: `pnpm exec vn-compliance --quick`. The exit code is the verdict; a SKIP is not a pass.
- T2 and T3 need an independent reviewer who does not fix. T3 commits carry `Reviewed-by: <reviewer>` or
  `Tier: T3 reviewed`. Commit by named path, never `git add -A`.

## Commands

`pnpm test` (pytest) · `pnpm check` (config check and tests) · `pnpm gates` · `dandadye --help`
