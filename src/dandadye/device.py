"""Pick a torch device and dtype: cuda (ROCm or NVIDIA) > mps > cpu.

The rule lives in two pure functions, ``pick_device`` and ``pick_dtype``, which
take availability flags as arguments and import nothing, so they can be unit
tested without a GPU and reused by other stages (the Magi worker, say).
``probe`` is the only part that touches torch, and only when called.

ROCm builds of PyTorch expose AMD GPUs through the ``torch.cuda`` API, so
"cuda" covers both. ``torch.version.hip`` / ``torch.version.cuda`` tell which
one it is, and ``gpu_flavour`` turns that into "rocm" or "cuda" for logs.

Dtype rule: bf16 on a GPU that supports it; fp16 on an older CUDA GPU without
bf16 (FLUX can overflow in fp16, so this is a last resort that is logged); bf16
on MPS when the build supports it, else fp32; fp32 on CPU. An explicit request
(``bf16`` / ``fp16`` / ``fp32``) always wins.
"""

from __future__ import annotations

from dataclasses import dataclass

DEVICES = ("auto", "cuda", "rocm", "mps", "cpu")
DTYPES = {"auto": None, "bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16",
          "float16": "float16", "fp32": "float32", "float32": "float32"}


def pick_device(requested: str = "auto", *, cuda: bool = False, mps: bool = False) -> str:
    """'auto' -> cuda > mps > cpu. An explicit device must be available ('rocm' means cuda).

    ``cuda:1`` style indices are kept. Raises ValueError for an unknown or unavailable device.
    """
    req = (requested or "auto").lower()
    base, _, index = req.partition(":")
    if base == "rocm":
        base, req = "cuda", "cuda" + (f":{index}" if index else "")
    if base not in DEVICES:
        raise ValueError(f"unknown device {requested!r} (expected one of {', '.join(DEVICES)})")
    if base == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if base == "cuda" and not cuda:
        raise ValueError(f"device {requested!r} requested but torch sees no CUDA/ROCm GPU")
    if base == "mps" and not mps:
        raise ValueError(f"device {requested!r} requested but torch has no MPS backend here")
    return req


def pick_dtype(device: str, requested: str = "auto", *, bf16: bool = False) -> str:
    """torch dtype name for ``device``. ``bf16`` = whether the device supports bfloat16."""
    try:
        explicit = DTYPES[(requested or "auto").lower()]
    except KeyError:
        raise ValueError(f"unknown dtype {requested!r} (expected auto, bf16, fp16 or fp32)") from None
    if explicit:
        return explicit
    base = device.split(":")[0]
    if base == "cuda":
        return "bfloat16" if bf16 else "float16"
    if base == "mps":
        return "bfloat16" if bf16 else "float32"
    return "float32"


def gpu_flavour(hip_version: str | None, cuda_version: str | None) -> str | None:
    """'rocm' / 'cuda' / None from torch.version.hip and torch.version.cuda."""
    if hip_version:
        return "rocm"
    if cuda_version:
        return "cuda"
    return None


@dataclass
class DeviceChoice:
    device: str            # "cuda", "cuda:1", "mps" or "cpu" (what torch .to() takes)
    dtype: str             # torch dtype attribute name, e.g. "bfloat16"
    flavour: str | None    # "rocm" | "cuda" | None
    detail: str            # human description for logs and the state record

    @property
    def is_gpu(self) -> bool:
        return self.device != "cpu"


def probe() -> dict:
    """Availability flags from the installed torch (all False if torch is missing)."""
    flags = dict(torch=None, cuda=False, mps=False, bf16_cuda=False, bf16_mps=False,
                 hip=None, cuda_version=None, name=None)
    try:
        import torch
    except Exception:
        return flags
    flags["torch"] = torch.__version__
    flags["hip"] = getattr(torch.version, "hip", None)
    flags["cuda_version"] = getattr(torch.version, "cuda", None)
    try:
        flags["cuda"] = bool(torch.cuda.is_available())
        if flags["cuda"]:
            flags["bf16_cuda"] = bool(torch.cuda.is_bf16_supported())
            flags["name"] = torch.cuda.get_device_name(0)
    except Exception:
        flags["cuda"] = False
    try:
        flags["mps"] = bool(torch.backends.mps.is_available())
        if flags["mps"]:
            torch.ones(1, dtype=torch.bfloat16, device="mps")
            flags["bf16_mps"] = True
    except Exception:
        pass
    return flags


def choose(requested_device: str = "auto", requested_dtype: str = "auto", flags: dict | None = None) -> DeviceChoice:
    """Combine ``probe()`` (or given flags) with the pure rules above."""
    f = probe() if flags is None else flags
    device = pick_device(requested_device, cuda=f.get("cuda", False), mps=f.get("mps", False))
    base = device.split(":")[0]
    bf16 = f.get("bf16_cuda", False) if base == "cuda" else f.get("bf16_mps", False) if base == "mps" else False
    dtype = pick_dtype(device, requested_dtype, bf16=bf16)
    flavour = gpu_flavour(f.get("hip"), f.get("cuda_version")) if base == "cuda" else None
    if base == "cuda":
        ver = f.get("hip") if flavour == "rocm" else f.get("cuda_version")
        detail = f"{device} ({'ROCm' if flavour == 'rocm' else 'CUDA'} {ver}" + \
                 (f", {f['name']})" if f.get("name") else ")")
    elif base == "mps":
        detail = "mps (Apple Metal via PyTorch)"
    else:
        detail = "cpu" + ("" if f.get("torch") else " (torch not installed)")
    return DeviceChoice(device=device, dtype=dtype, flavour=flavour, detail=f"{detail}, {dtype}")
