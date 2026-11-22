# Sourced by scripts/setup.sh and scripts/linux-gpu-setup.sh: the PyTorch wheel indexes and the auto-detection.
# Current PyTorch wheel indexes with cp312 wheels for torch and torchvision (checked 22 Nov 2026).
ROCM_INDEX="https://download.pytorch.org/whl/rocm7.2"
CUDA_INDEX="https://download.pytorch.org/whl/cu126"
CPU_INDEX="https://download.pytorch.org/whl/cpu"

has_amd() { [ -e /dev/kfd ] || { command -v rocminfo >/dev/null 2>&1 && rocminfo >/dev/null 2>&1; }; }
has_nvidia() { command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; }

# resolve_torch <auto|rocm|cuda|cpu>: sets TORCH (never auto) and TORCH_URL (TORCH_INDEX_URL overrides).
resolve_torch() {
  TORCH="$1"
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
}
