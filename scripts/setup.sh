#!/bin/bash
# One-command setup for dan-da-dye on an Apple Silicon Mac or a Linux box.
#
#   scripts/setup.sh <data-root> [--with-flux3] [--dry-run] [--torch auto|rocm|cuda|cpu]
#
# Creates <data-root>/.venv (this package + mflux on a Mac, diffusers on Linux) and <data-root>/.venv-magi
# (the Magi v2 detector, torch),
# then downloads the models into <data-root> with retries, so a dropped connection costs a retry, not your
# evening. Safe to re-run: finished downloads are skipped by the Hugging Face cache. Nothing is written
# inside this repository.
#
# On Linux, --torch picks the PyTorch wheel index (default auto: ROCm if /dev/kfd or rocminfo shows an AMD GPU,
# CUDA if nvidia-smi works, otherwise CPU). Set TORCH_INDEX_URL to override the index entirely, and
# DANDADYE_PYTHON to choose the interpreter. See docs/LINUX.md.
#
# Before the first run: accept the FLUX.2 [klein] licence on its Hugging Face page (it is gated) and log in
# with `hf auth login` once the venv exists, or set HF_TOKEN. This script will remind you, politely, twice.
set -euo pipefail

usage() { echo "usage: scripts/setup.sh <data-root> [--with-flux3] [--dry-run] [--torch auto|rocm|cuda|cpu]" >&2; exit 2; }
[ $# -ge 1 ] || usage
DATA="$1"; shift
FLUX3=0; DRY=0; TORCH=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --with-flux3) FLUX3=1 ;;
    --dry-run) DRY=1 ;;
    --torch) [ $# -ge 2 ] || usage; TORCH="$2"; shift ;;
    --torch=*) TORCH="${1#--torch=}" ;;
    *) usage ;;
  esac
  shift
done
case "$TORCH" in auto|rocm|cuda|cpu) ;; *) usage ;; esac
OS="$(uname -s)"

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

# Current PyTorch wheel indexes with cp312 wheels for torch and torchvision (checked 22 Nov 2026).
ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
CUDA_INDEX="https://download.pytorch.org/whl/cu126"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_INSTALL=()     # extra pip step on Linux: torch + torchvision from the chosen index
INHIBIT=()           # keeps the machine awake during downloads, where the OS offers a way

say "Checking the machine"
if [ "$OS" = Darwin ]; then
  [ "$(uname -m)" = arm64 ] || { echo "setup: needs an Apple Silicon Mac (this is Darwin/$(uname -m)). MLX does not do CUDA, and neither do we." >&2; exit 1; }
  [ "$TORCH" = auto ] || { echo "setup: --torch is for Linux. On a Mac, torch comes from PyPI and uses MPS." >&2; exit 2; }
  PY="$(command -v python3.12 || true)"
  [ -n "$PY" ] || { echo "setup: python3.12 not found. Try: brew install python@3.12" >&2; exit 1; }
  MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  echo "  Apple Silicon, ${MEM_GB} GB memory, $PY"
  MAIN_EXTRAS="mlx,dev"
  INHIBIT=(caffeinate -i)
  free_gb() { df -g "$1" | awk 'NR==2{print $4}'; }
elif [ "$OS" = Linux ]; then
  # Python: 3.12 preferred (what the pins are tested on), then uv-managed 3.12, then 3.11 / 3.13 with a warning.
  PY="${DANDADYE_PYTHON:-$(command -v python3.12 || true)}"
  if [ -z "$PY" ] && command -v uv >/dev/null 2>&1; then PY="$(uv python find 3.12 2>/dev/null || true)"; fi
  if [ -z "$PY" ]; then
    for v in 3.11 3.13; do
      PY="$(command -v "python$v" || true)"
      [ -z "$PY" ] || { echo "  warning: python3.12 not found, using $PY. Magi's pins are tested on 3.12." >&2; break; }
    done
  fi
  [ -n "$PY" ] || { echo "setup: python3.12 not found. Install it with your package manager, or: uv python install 3.12" >&2; exit 1; }
  MEM_GB=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) / 1048576 ))
  echo "  Linux $(uname -m), ${MEM_GB} GB memory, $PY"
  MAIN_EXTRAS="diffusers,dev"
  command -v systemd-inhibit >/dev/null 2>&1 && INHIBIT=(systemd-inhibit --what=idle:sleep --who=dandadye --why="downloading models")
  free_gb() { df -BG --output=avail "$1" | awk 'NR==2{gsub("G","",$1); print $1}'; }

  has_amd() { [ -e /dev/kfd ] || { command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; }; }
  has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }
  if [ "$TORCH" = auto ]; then
    if has_amd; then TORCH=rocm; elif has_nvidia; then TORCH=cuda; else TORCH=cpu; fi
    echo "  torch build: $TORCH (auto-detected; override with --torch rocm|cuda|cpu)"
  else
    echo "  torch build: $TORCH (from --torch)"
  fi
  case "$TORCH" in
    rocm) TORCH_URL="$ROCM_INDEX" ;;
    cuda) TORCH_URL="$CUDA_INDEX" ;;
    cpu)  TORCH_URL="$CPU_INDEX" ;;
  esac
  TORCH_URL="${TORCH_INDEX_URL:-$TORCH_URL}"
  echo "  torch index: $TORCH_URL"
  TORCH_INSTALL=(install -q torch torchvision --index-url "$TORCH_URL")
  if [ "$TORCH" = rocm ] && command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -q gfx1030; then
    echo "  note: gfx1030 (RDNA2, e.g. Radeon PRO W6800) is supported by the ROCm wheels as is. Only if torch reports"
    echo "        no GPU at runtime, try HSA_OVERRIDE_GFX_VERSION=10.3.0 as a fallback. Check with scripts/gpu-check."
  fi
  [ "$TORCH" != cpu ] || echo "  note: CPU torch works but is slow. Magi is tolerable, the colour stage is not."
else
  echo "setup: unsupported OS $OS (macOS on Apple Silicon and Linux are supported)." >&2; exit 1
fi
if [ "$DRY" = 0 ]; then mkdir -p "$DATA"; fi
FREE_AT="$DATA"; while [ ! -d "$FREE_AT" ]; do FREE_AT="$(dirname "$FREE_AT")"; done
FREE_GB=$(free_gb "$FREE_AT")
echo "  ${FREE_GB} GB free where the data root lives (the models want about 90 GB)"
[ "$MEM_GB" -ge 16 ] || echo "  warning: under 16 GB. The colour stage will try its best, which may not be enough."

say "Python environments"
[ -x "$DATA/.venv/bin/python" ] || run "$PY" -m venv "$DATA/.venv"
# Linux: torch and torchvision first, from the chosen index, so the later installs see them as already satisfied.
[ ${#TORCH_INSTALL[@]} -eq 0 ] || run "$DATA/.venv/bin/pip" "${TORCH_INSTALL[@]}"
run "$DATA/.venv/bin/pip" install -q -e "$REPO[$MAIN_EXTRAS]"
[ -x "$DATA/.venv-magi/bin/python" ] || run "$PY" -m venv "$DATA/.venv-magi"
[ ${#TORCH_INSTALL[@]} -eq 0 ] || run "$DATA/.venv-magi/bin/pip" "${TORCH_INSTALL[@]}"
run "$DATA/.venv-magi/bin/pip" install -q -r "$REPO/requirements-magi.txt"

export HF_HOME="$DATA/hf" HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
HF="$DATA/.venv/bin/hf"
if [ "$DRY" = 0 ] && [ -z "${HF_TOKEN:-}" ] && ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "setup: not logged in to Hugging Face. Accept the FLUX.2 [klein] licence on its model page, then run:" >&2
  echo "  HF_HOME=\"$DATA/hf\" \"$HF\" auth login" >&2
  echo "and run this script again (the environments above are kept)." >&2
  exit 1
fi

# Retries until the download completes; caffeinate (macOS) or systemd-inhibit (Linux, if present) keeps the machine awake.
fetch() {
  local repo="$1"; shift
  local log="$DATA/dl-${repo##*/}.log" try=1
  echo "  $repo (log: $log)"
  [ "$DRY" = 1 ] && { run "$HF" download "$repo" --max-workers 8 "$@"; return; }
  until ${INHIBIT[@]+"${INHIBIT[@]}"} "$HF" download "$repo" --max-workers 8 "$@" >>"$log" 2>&1; do
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
if [ "$OS" = Linux ]; then
  echo "  Check the GPU is visible to torch with: $DATA/.venv/bin/python $REPO/scripts/gpu-check --expect-gpu"
fi
