# Security baseline

## Secrets

- Secrets never go in the repo, in logs, in reports, in commit messages, or on a command line (shell
  history and process lists keep arguments). They live in the platform's secret store, the OS keychain,
  or git-ignored local files that never leave the machine.
- Set Worker secrets through stdin: `printf '%s' "$VALUE" | wrangler secret put NAME [--env staging]`
  or let wrangler prompt; never `wrangler secret put NAME <value>` in a script.
- Test fixtures use values that are obviously fake; the privacy scan ignores known fake shapes and
  `.privacy-allow` lists reviewed false positives for the whole-tree audit only, as `path<TAB>exact line
  text` (not a line number: the pre-commit scan never sees a line move) or `path<TAB>*` for a whole fixture file.
- A leaked secret is rotated first, then removed from history if needed, then written up (INCIDENTS.md).

## Headers and CSP

Every HTML response sets, at minimum: a restrictive `Content-Security-Policy` (no inline script without
a hash or nonce, `frame-ancestors 'none'`), `X-Content-Type-Options: nosniff`,
`Referrer-Policy: no-referrer` (or `strict-origin-when-cross-origin` where a referrer is needed),
`X-Frame-Options: DENY`, a minimal `Permissions-Policy`, and `Strict-Transport-Security` on HTTPS.
Cookies are `Secure`, `HttpOnly`, `SameSite`, `__Host-` prefixed where possible. Loosening any of these
is T3.

## Identity gate and bypass paths

- Browser paths sit behind the identity gate (e.g. Cloudflare Access). The app still verifies the
  gate's token itself; a header alone is never trusted.
- A path that bypasses the gate (devices, webhooks, public pages) must have **its own authentication**
  (device credential, signed token, one-time link) **and rate limits**, and must show nothing personal
  to an anonymous caller. Adding or widening a bypass path is T3 and needs the owner's OK.

## Data residency and logging

- Keep data in the agreed region and stores; a new store or region is an ADR.
- Logs carry ids, paths, status codes and timings, never request bodies, personal data or secrets.
- Turning platform observability off is a legitimate, deliberate exception when logs could hold
  personal data; record it in an ADR and say how failures are noticed instead.

## Threat model: when to update it

Update the threat model (or the security section of the relevant doc) in the same change whenever you
add **a new endpoint, a new bypass path, a new data store or table holding personal data, or a new
third party** (API, SDK, push provider, model provider), or change who can read what.

## Every commit

The privacy and secret scan runs on every commit (pre-commit hook and the ship script) and on the whole
tree in the gate runner and CI. It refuses likely secrets, likely phone numbers and words from a local,
git-ignored denylist. The scan is a backstop; reviewers still read the diff.

## App rules

- Pages, coloured output, official art, palette images and model weights never enter git. They live in the data
  root, outside the repo (`scripts/setup.sh` refuses a data root inside it). `.gitignore`, the `sensitive` preset in
  the pre-commit and pre-push hooks and the `no-media` gate each enforce this. Test images are drawn in `tests/conftest.py`.
- Secrets: only the Hugging Face login (`hf auth login` or `HF_TOKEN`) and the owner's wrangler login. Neither is
  stored in this repo. The private GitHub Packages token comes from `NODE_AUTH_TOKEN` at install time.
- At run time, network use is limited to Hugging Face model downloads (mflux, if a model is not cached) and the
  wrangler upload. Setup also reaches Hugging Face and PyPI (`scripts/setup.sh`), and GitHub Packages (`pnpm install`).
