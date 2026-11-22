#!/bin/bash
# Venvs only, for a Linux box with a GPU: no model downloads, no Hugging Face login.
#
#   scripts/linux-gpu-setup.sh <data-root> [--dry-run] [--torch auto|rocm|cuda|cpu]
#
# Creates <data-root>/.venv (python 3.12; torch + torchvision from the wheel index scripts/setup.sh would pick,
# then `pip install -e '.[diffusers,dev]'`) and <data-root>/.venv-magi (same torch, then requirements-magi.txt).
# Uses uv when available (installing Python 3.12 through it if needed), otherwise python3.12 -m venv + pip.
# Check the result with: <data-root>/.venv/bin/python scripts/gpu-check --expect-gpu
#
# gfx1030 (RDNA2, e.g. Radeon PRO W6800) is supported by the ROCm wheels as is. Set HSA_OVERRIDE_GFX_VERSION=10.3.0
# only if gpu-check fails to see the GPU without it. See docs/LINUX.md.
set -euo pipefail

usage() { echo "usage: scripts/linux-gpu-setup.sh <data-root> [--dry-run] [--torch auto|rocm|cuda|cpu]" >&2; exit 2; }
[ $# -ge 1 ] || usage
DATA="$1"; shift
DRY=0; TORCH=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --torch) [ $# -ge 2 ] || usage; TORCH="$2"; shift ;;
    --torch=*) TORCH="${1#--torch=}" ;;
    *) usage ;;
  esac
  shift
done
case "$TORCH" in auto|rocm|cuda|cpu) ;; *) usage ;; esac
[ "$(uname -s)" = Linux ] || { echo "linux-gpu-setup: Linux only (on a Mac use scripts/setup.sh)" >&2; exit 1; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
abs_path() {
  local p="$1" rest=""
  case "$p" in /*) ;; *) p="$PWD/$p" ;; esac
  while [ ! -d "$p" ]; do rest="/$(basename "$p")$rest"; p="$(dirname "$p")"; done
  echo "$(cd "$p" && pwd -P)$rest"
}
DATA="$(abs_path "$DATA")"
REPO_REAL="$(cd "$REPO" && pwd -P)"
case "$(echo "$DATA/" | tr '[:upper:]' '[:lower:]')" in
  "$(echo "$REPO_REAL/" | tr '[:upper:]' '[:lower:]')"*) echo "linux-gpu-setup: the data root must be outside the repo" >&2; exit 2 ;;
esac

run() { if [ "$DRY" = 1 ]; then printf '  would run:'; printf ' %q' "$@"; echo; else "$@"; fi; }
. "$REPO/scripts/torch-index.sh"
resolve_torch "$TORCH"

if command -v uv >/dev/null 2>&1; then
  mkvenv() { run uv venv --python 3.12 --allow-existing "$1"; }
  pipi() { local v="$1"; shift; run uv pip install --python "$v/bin/python" "$@"; }
  [ "$DRY" = 1 ] && echo "  uv: $(command -v uv)" || true
  run uv python install 3.12
else
  PY="${DANDADYE_PYTHON:-$(command -v python3.12 || true)}"
  [ -n "$PY" ] || { echo "linux-gpu-setup: python3.12 not found (install it, or install uv)" >&2; exit 1; }
  mkvenv() { [ -x "$1/bin/python" ] || run "$PY" -m venv "$1"; }
  pipi() { local v="$1"; shift; run "$v/bin/pip" install -q "$@"; }
fi

[ "$DRY" = 1 ] || mkdir -p "$DATA"
MAIN="$DATA/.venv"; MAGI="$DATA/.venv-magi"
echo; echo "== main venv: $MAIN"
mkvenv "$MAIN"
pipi "$MAIN" torch torchvision --index-url "$TORCH_URL"
pipi "$MAIN" -e "$REPO[diffusers,dev]"
echo; echo "== magi venv: $MAGI"
mkvenv "$MAGI"
pipi "$MAGI" torch torchvision --index-url "$TORCH_URL"
pipi "$MAGI" -r "$REPO/requirements-magi.txt"
echo
echo "Done. Check: $MAIN/bin/python $REPO/scripts/gpu-check --expect-gpu"
