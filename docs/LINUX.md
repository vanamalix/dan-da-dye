# Running dan-da-dye on Linux

The pipeline was built on Apple Silicon. The ingest, detect, refsheet and export stages are plain Python plus
PyTorch, so they run on Linux with an AMD GPU (ROCm), an NVIDIA GPU (CUDA) or just the CPU. The colour stage
uses diffusers on Linux where the Mac uses mflux (MLX). This page covers setup and the pieces that differ.

## Setup

You need Python 3.12 (3.11 and 3.13 work with a warning), `git`, and roughly 90 GB free for the models. Then:

```sh
git clone <this repo> && cd dan-da-dye
scripts/setup.sh /path/to/data-root --dry-run     # prints every command, writes nothing
scripts/setup.sh /path/to/data-root               # the real thing
```

The data root must be outside the repo. The script refuses otherwise.

`--torch auto|rocm|cuda|cpu` chooses which PyTorch build to install. `auto` (the default) picks ROCm if
`/dev/kfd` exists or `rocminfo` works, CUDA if `nvidia-smi` works, and CPU otherwise. It installs `torch` and
`torchvision` from the matching `download.pytorch.org` index into both venvs before anything else, so pip
doesn't swap them for the CPU or CUDA wheels from PyPI. The main venv then gets `.[diffusers,dev]`.

Other knobs:

- `TORCH_INDEX_URL` replaces the wheel index (for a mirror, or a newer ROCm or CUDA build).
- `DANDADYE_PYTHON` picks the interpreter. Without it the script looks for `python3.12`, then a uv-managed 3.12
  (`uv python install 3.12` if you have uv), then 3.11 or 3.13.
- `--with-flux3` works as on the Mac.

As on the Mac, the FLUX.2 [klein] models are gated. Accept the licence on Hugging Face and run `hf auth login`
(or set `HF_TOKEN`) before the real run.

After setup, check the GPU is really being used:

```sh
$DANDADYE_DATA/.venv/bin/python scripts/gpu-check --expect-gpu
```

## What `gpu-check` does

It prints the torch version and build (ROCm with its HIP version, CUDA, MPS or CPU), each GPU's name,
architecture and memory, the environment variables that commonly matter (`HSA_OVERRIDE_GFX_VERSION`,
`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`) and whether `/dev/kfd` and `/dev/dri`
exist. It then times a matmul in fp32 and bf16 and compares it with the CPU result, and runs a tiny diffusion
step. With diffusers installed that is a random-weight FLUX.2 Klein pipeline (a few MB, downloaded to `HF_HOME`)
run for two steps at 64x64 with two reference images. Without diffusers it does a small convolutional
forward and backward pass.

Flags: `--device auto|cuda|mps|cpu`, `--skip-diffusion` (no download), `--json`, and `--expect-gpu`, which makes
the exit status non-zero if no GPU is in use. A failed check also gives a non-zero exit.

## Choosing the device

The detect stage takes `dandadye detect --device auto|cuda|mps|cpu`. `auto` prefers cuda, then mps, then cpu.
"cuda" means either NVIDIA CUDA or AMD ROCm, because the ROCm build of PyTorch answers to the `torch.cuda` API.
The worker logs which one it got, for example `cuda device via ROCm (HIP 7.2...)`. If the GPU run fails it
retries on the CPU, as it always did for MPS.

If more than one GPU is visible and you want a particular one, narrow it with `ROCR_VISIBLE_DEVICES` (ROCm) or
`CUDA_VISIBLE_DEVICES` (NVIDIA) before running. Torch's device 0 is then the first one left.

## ROCm notes (including gfx1030)

The Radeon PRO W6800 is gfx1030 (RDNA2), which the ROCm PyTorch wheels support as is. Try without any override
first. Only if `gpu-check` says no GPU is visible while `/dev/kfd` is present and your user is in the right
groups, try `HSA_OVERRIDE_GFX_VERSION=10.3.0` as a fallback. It makes the runtime treat the card as gfx1030 and
is meant for chips that are close to a supported one, so it shouldn't be needed here.

## Containers

The GPU has to be passed into the container, or torch quietly falls back to the CPU.

AMD (ROCm):

- Pass `/dev/kfd` and `/dev/dri`, for example `--device=/dev/kfd --device=/dev/dri`.
- The container user needs the host's `video` and `render` groups, for example `--group-add video --group-add render`
  (using the numeric group ids from the host if the names don't exist in the image).
- Many setups also want `--security-opt seccomp=unconfined` and `--ipc=host`. Add those only if the plain
  version fails.
- You don't need the full ROCm stack in the image, because the PyTorch ROCm wheels bring their own runtime
  libraries. The host kernel driver (amdgpu) does need to be present.

NVIDIA (CUDA):

- Install the NVIDIA Container Toolkit on the host and run with `--gpus all` (or `--runtime=nvidia`, or a CDI
  device with Podman).
- The host driver must be new enough for the wheel. The default `cu126` index needs a 525 series driver or later.

Check from inside the container with `ls -l /dev/kfd /dev/dri` (AMD) or `nvidia-smi` (NVIDIA), then `gpu-check`.

## CPU only

It works, slowly. Magi detection is tolerable. The colour stage on the CPU is not something to wait for.

## Magi's solver

Magi assigns names with pulp. On Linux x86_64 pulp bundles the CBC binary, so it works without help. The worker
still prefers HiGHS (`highspy`, which is in `requirements-magi.txt`) when it loads, and falls back to pulp's
default if it doesn't. On Linux aarch64 pulp has no bundled CBC, so `highspy` matters there.

## Substitutions

| Component | On Apple Silicon | On Linux | Where |
| --- | --- | --- | --- |
| Magi device | MPS (`PYTORCH_ENABLE_MPS_FALLBACK=1`) | `cuda` through `torch.cuda`, which is ROCm on AMD and CUDA on NVIDIA. `auto` order is cuda, mps, cpu | `src/dandadye/detect/worker_magi_v2.py` |
| Keep awake during downloads | `caffeinate -i` | `systemd-inhibit` if installed, otherwise nothing | `scripts/setup.sh` |
| Memory size | `sysctl -n hw.memsize` | `/proc/meminfo` | `scripts/setup.sh` |
| Free disk space | `df -g` | `df -BG --output=avail` | `scripts/setup.sh` |
| torch wheels | PyPI (MPS build) | `download.pytorch.org/whl/rocm7.2`, `/cu126` or `/cpu`, chosen by `--torch`, installed into both venvs before other packages | `scripts/setup.sh` |
| Main venv extras | `.[mlx,dev]` | `.[diffusers,dev]` | `scripts/setup.sh` |
| Pulp solver | HiGHS (CBC is x86-only on macOS) | HiGHS when present, bundled CBC otherwise | `worker_magi_v2.py` |
| Colour engine | mflux / MLX | diffusers (`Flux2KleinPipeline` plus the manga LoRA, chosen with `--runtime diffusers`, the default off macOS). Same flow: page then reference sheet as the two images, real CFG on the base model, LoRA fused at load. `-q` has no diffusers equivalent and is ignored with a note; `--backend cli` is mflux only; seeds do not give the same noise as MLX | `src/dandadye/colourise_diffusers.py` |

## What hasn't been tested

At the time of writing this was set up on a machine with no GPU passed through, so the ROCm and CUDA paths are
checked by dry run and unit tests, and the CPU path by running `gpu-check`. A first run on a real card should
start with `gpu-check --expect-gpu`. The Magi pins (`transformers==4.45.2`) are old, and nobody has yet confirmed
they work with the newest torch builds.
