#!/bin/bash
# One-command setup for dan-da-dye on an Apple Silicon Mac.
#
#   scripts/setup.sh <data-root> [--with-flux3] [--dry-run]
#
# Creates <data-root>/.venv (this package + mflux) and <data-root>/.venv-magi (the Magi v2 detector, torch),
# then downloads the models into <data-root> with retries, so a dropped connection costs a retry, not your
# evening. Safe to re-run: finished downloads are skipped by the Hugging Face cache. Nothing is written
# inside this repository.
#
# Before the first run: accept the FLUX.2 [klein] licence on its Hugging Face page (it is gated) and log in
# with `hf auth login` once the venv exists, or set HF_TOKEN. This script will remind you, politely, twice.
set -euo pipefail

usage() { echo "usage: scripts/setup.sh <data-root> [--with-flux3] [--dry-run]" >&2; exit 2; }
[ $# -ge 1 ] || usage
DATA="$1"; shift
FLUX3=0; DRY=0
for a in "$@"; do
  case "$a" in
    --with-flux3) FLUX3=1 ;;
    --dry-run) DRY=1 ;;
    *) usage ;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
# Absolute, symlink-resolved data root (it may not exist yet: resolve the nearest existing parent).
abs_path() {
  local p="$1" rest=""
  case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
  while [ ! -d "$p" ]; do rest="/$(basename "$p")$rest"; p="$(dirname "$p")"; done
  echo "$(cd "$p" && pwd -P)$rest"
}
DATA="$(abs_path "$DATA")"
REPO_REAL="$(cd "$REPO" && pwd -P)"
case "$(echo "$DATA/" | tr '[:upper:]' '[:lower:]')" in
  "$(echo "$REPO_REAL/" | tr '[:upper:]' '[:lower:]')"*) echo "setup: the data root must be outside the repo (pages and models never go in git)" >&2; exit 2 ;;
esac

run() { if [ "$DRY" = 1 ]; then printf '  would run:'; printf ' %q' "$@"; echo; else "$@"; fi; }
say() { printf '\n== %s\n' "$*"; }

say "Checking the machine"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "setup: needs an Apple Silicon Mac (this is $(uname -s)/$(uname -m)). MLX does not do CUDA, and neither do we." >&2; exit 1; }
PY="$(command -v python3.12 || true)"
[ -n "$PY" ] || { echo "setup: python3.12 not found. Try: brew install python@3.12" >&2; exit 1; }
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "  Apple Silicon, ${MEM_GB} GB memory, $PY"
[ "$MEM_GB" -ge 16 ] || echo "  warning: under 16 GB. The colour stage will try its best, which may not be enough."
if [ "$DRY" = 0 ]; then mkdir -p "$DATA"; fi
FREE_GB=$(df -g "$( [ -d "$DATA" ] && echo "$DATA" || dirname "$DATA")" | awk 'NR==2{print $4}')
echo "  ${FREE_GB} GB free where the data root lives (the models want about 90 GB)"

say "Python environments"
[ -x "$DATA/.venv/bin/python" ] || run "$PY" -m venv "$DATA/.venv"
run "$DATA/.venv/bin/pip" install -q -e "$REPO[mlx,dev]"
[ -x "$DATA/.venv-magi/bin/python" ] || run "$PY" -m venv "$DATA/.venv-magi"
run "$DATA/.venv-magi/bin/pip" install -q -r "$REPO/requirements-magi.txt"

export HF_HOME="$DATA/hf" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
HF="$DATA/.venv/bin/hf"
if [ "$DRY" = 0 ] && [ -z "${HF_TOKEN:-}" ] && ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "setup: not logged in to Hugging Face. Accept the FLUX.2 [klein] licence on its model page, then run:" >&2
  echo "  HF_HOME=\"$DATA/hf\" \"$HF\" auth login" >&2
  echo "and run this script again (the environments above are kept)." >&2
  exit 1
fi

# Retries until the download completes; caffeinate keeps the Mac awake meanwhile.
fetch() {
  local repo="$1"; shift
  local log="$DATA/dl-${repo##*/}.log" try=1
  echo "  $repo (log: $log)"
  [ "$DRY" = 1 ] && { run "$HF" download "$repo" --max-workers 8 "$@"; return; }
  until caffeinate -i "$HF" download "$repo" --max-workers 8 "$@" >>"$log" 2>&1; do
    echo "    attempt $try failed, retrying in 15 s"
    try=$((try + 1))
    sleep 15
  done
}

say "Models (resumable; go and make tea)"
fetch black-forest-labs/FLUX.2-klein-base-9B --exclude "flux-2-klein-*.safetensors" --exclude "*.jpg"
fetch black-forest-labs/FLUX.2-klein-9B --exclude "flux-2-klein-*.safetensors" --exclude "*.jpg"
fetch thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA --local-dir "$DATA/lora/manga-colorization"
fetch ragavsachdeva/magiv2 --local-dir "$DATA/models/magiv2"
[ "$FLUX3" = 0 ] || fetch black-forest-labs/flux-3-image-klein --exclude "model.safetensors" --exclude "ae.safetensors" --exclude "variants/*" --exclude "*.jpg" --exclude "*.png"

say "Done"
cat <<EOF
  Add these to your shell profile:
    export DANDADYE_DATA="$DATA"
    export HF_HOME="\$DANDADYE_DATA/hf"
    export PATH="\$DANDADYE_DATA/.venv/bin:\$PATH"
  Then put your pages in \$DANDADYE_DATA/work/src/<chapter>/ and your preset's character art and palette
  where the preset says, and start with: dandadye ingest
EOF
