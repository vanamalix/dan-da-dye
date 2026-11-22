"""Stage 4 on PyTorch + diffusers: the colourise engines for Linux (ROCm, CUDA) and CPU.

``colourise.run`` picks this when the runtime is ``diffusers`` (the default off
Apple Silicon). It runs the same two flows as the mflux path, writes the same
files and records the same state, so ``inkfix`` and ``export`` don't care which
runtime made a page. torch and diffusers are imported lazily: the plan helpers
(``load_summary``, ``page_plan``) work without them, which keeps ``--dry-run``
usable on a machine that has neither.

``flux2-lora``: ``Flux2KleinPipeline`` with the manga LoRA.
  * ``image=[page, refsheet]`` in that order; diffusers concatenates the
    reference latents with time offsets 10, 20, ... (``_prepare_image_ids``),
    which is the ComfyUI ReferenceLatent / mflux ordering.
  * The base model is undistilled, so ``guidance_scale`` is real CFG: diffusers
    runs a second pass with an empty negative prompt when ``guidance_scale > 1``
    and the checkpoint's ``is_distilled`` is false (the base repos leave it
    unset). The references go into both passes, as in the LoRA's ComfyUI
    workflow (CFGGuider, empty negative). A distilled checkpoint ignores it.
  * References over 1 MP are downscaled by diffusers and floored to 16 px; we
    already hand it 16-aligned images at ``--megapixels``, so at the default
    1.0 nothing is resized again. The canvas is the page's working size.
  * LoRA: ``load_lora_weights`` (diffusers converts the BFL-format keys), then
    ``fuse_lora`` + ``unload_lora_weights`` when ``bake_lora``, else
    ``set_adapters([...], [scale])``.
  * The prompt is fixed per batch, so it is encoded once (with the empty
    negative when CFG is on) and passed as ``prompt_embeds`` /
    ``negative_prompt_embeds``. With ``--low-ram`` the text encoder is then
    dropped, as on the mflux path.
  * Seeds use a CPU ``torch.Generator``, so a seed gives the same noise on
    CUDA, ROCm and CPU (but not the same noise as MLX).

``flux3``: the FLUX 3 Image Klein modular pipeline (``Flux3DistilledModularPipeline``),
  loaded the way its model card describes, with the ``prompt_upsampler`` block
  removed because ``flux3_prompt`` already writes the full caption. Page first,
  then one sheet per character, 16 steps, no guidance, no LoRA. This path
  follows the model card and is checked on a tiny random pipeline only.

Decisions (also printed by ``--dry-run``):
  * ``--backend cli`` is mflux-only (one mflux process per page); under this
    runtime it is refused with a message, rather than silently mapped to api.
  * ``-q``: diffusers has no equivalent of mflux's load-time quantisation. The
    flag is ignored with a logged note and the weights load in the chosen dtype.
    To run smaller weights, pass ``--model`` / ``--model-path`` a checkpoint that
    was saved already quantised (for example bitsandbytes NF4 on CUDA, which
    ``from_pretrained`` loads when bitsandbytes is installed).
  * ``--low-ram`` (default on): model CPU offload on a GPU (only the active
    model is on the device), VAE tiling, and the text encoder freed once the
    prompt is encoded. On CPU there is nothing to offload to, so only tiling and
    the freeing apply.
  * Device: ``--device auto`` = cuda (ROCm or NVIDIA) > mps > cpu; ``--dtype
    auto`` = bf16 where supported, else fp16 on old CUDA GPUs, fp32 on CPU (see
    ``dandadye.device``).
"""

from __future__ import annotations

import gc
import inspect
import json
import os
from pathlib import Path

from PIL import Image

from .device import DeviceChoice, choose

# mflux model names -> diffusers repos. ``--model`` may also be a repo id or a local
# diffusers folder; ``--model-path`` (a local folder) wins over both.
DIFFUSERS_REPOS = {
    "flux2-klein-base-9b": "black-forest-labs/FLUX.2-klein-base-9B",
    "flux2-klein-9b": "black-forest-labs/FLUX.2-klein-9B",
    "flux2-klein-base-4b": "black-forest-labs/FLUX.2-klein-base-4B",
    "flux2-klein-4b": "black-forest-labs/FLUX.2-klein-4B",
    "flux3-image-klein": "black-forest-labs/flux-3-image-klein",
}
PIPELINES = {"flux2-lora": "Flux2KleinPipeline", "flux3": "Flux3DistilledModularPipeline"}
ADAPTER = "manga"
CLI_BACKEND_REFUSAL = ("--backend cli runs one mflux process per page and only exists on the mlx runtime; "
                       "the diffusers runtime loads the pipeline once per batch (drop --backend, or use --backend api)")


def resolve_repo(s) -> str:
    """The repo id or local folder ``from_pretrained`` gets for these settings."""
    if s.model_path:
        return str(s.model_path)
    if s.model in DIFFUSERS_REPOS:
        return DIFFUSERS_REPOS[s.model]
    if "/" in s.model or Path(s.model).expanduser().is_dir():
        return s.model
    raise SystemExit(f"--model {s.model!r}: not a known model ({', '.join(DIFFUSERS_REPOS)}), "
                     "a Hugging Face repo id or a local diffusers folder")


def check_supported(s, backend: str) -> None:
    if backend == "cli":
        raise SystemExit(CLI_BACKEND_REFUSAL)
    if s.engine not in PIPELINES:
        raise SystemExit(f"engine {s.engine!r} has no diffusers pipeline")


def guidance_scale(s) -> float:
    return 1.0 if s.guidance is None else float(s.guidance)


def load_notes(s, choice: DeviceChoice) -> list[str]:
    """Human notes on how the mflux-style settings map onto diffusers."""
    notes = []
    if s.quantize:
        notes.append(f"-q {s.quantize} is an mflux option with no diffusers equivalent: ignored, weights load in "
                     f"{choice.dtype} (pass --model/--model-path a pre-quantised diffusers checkpoint to go smaller)")
    if s.low_ram:
        notes.append("low-ram: " + ("model CPU offload, " if choice.is_gpu else "")
                     + "VAE tiling, text encoder freed after the prompt is encoded")
    if s.engine == "flux2-lora" and s.megapixels > 1.0:
        notes.append("diffusers downscales reference images to 1 MP whatever --megapixels says")
    if not choice.is_gpu:
        notes.append("running on CPU: expect minutes per step for a 9B model")
    return notes


def load_summary(s, choice: DeviceChoice) -> dict:
    """What would be loaded, once per batch (JSON-able)."""
    lora = None
    if s.uses_lora:
        lora = {"path": s.lora_path, "scale": s.lora_scale, "adapter": ADAPTER,
                "mode": "fused at load" if s.bake_lora else "adapter (set_adapters)"}
    return {"runtime": "diffusers", "pipeline": PIPELINES[s.engine], "model": resolve_repo(s), "lora": lora,
            "device": choice.device, "dtype": choice.dtype, "device_detail": choice.detail,
            "cpu_offload": bool(s.low_ram and choice.is_gpu), "vae_tiling": bool(s.low_ram),
            "quantize": None if not s.quantize else f"ignored (mflux -q {s.quantize})",
            "notes": load_notes(s, choice)}


def call_kwargs(s, page_in: Path, refs: list[Path], width: int, height: int, prompt: str) -> dict:
    """The pipeline call, with paths standing in for images (``generator`` from ``s.seed``)."""
    kw = {"image": [str(page_in), *[str(r) for r in refs]], "prompt": prompt,
          "height": height, "width": width, "num_inference_steps": s.steps}
    if s.engine == "flux2-lora":
        kw["guidance_scale"] = guidance_scale(s)
    return kw


def page_plan(s, choice: DeviceChoice, key: str, page_in: Path, refs: list[Path], out: Path,
              width: int, height: int, prompt: str) -> dict:
    return {"page": key, "call": call_kwargs(s, page_in, refs, width, height, prompt), "seed": s.seed,
            "negative_prompt": "" if s.engine == "flux2-lora" and guidance_scale(s) > 1 else None,
            "output": str(out), "load": load_summary(s, choice)}


def format_load(summary: dict) -> list[str]:
    lines = [f"diffusers: {summary['pipeline']} from {summary['model']} on {summary['device_detail']}"]
    if summary["lora"]:
        lo = summary["lora"]
        lines.append(f"  LoRA {lo['path']} x{lo['scale']:g} ({lo['mode']})")
    lines += [f"  note: {n}" for n in summary["notes"]]
    return lines


def format_plan(plan: dict) -> str:
    """One JSON line per page: the call, the seed and where the image goes."""
    return json.dumps({"page": plan["page"], **plan["call"], "seed": plan["seed"], "output": plan["output"]})


def check_call(engine: str, call: dict) -> str:
    """Check the call's keyword names against the installed diffusers (no model load). '' if OK."""
    try:
        if engine == "flux3":
            from diffusers.modular_pipelines import Flux3DistilledAutoBlocks
            blocks = Flux3DistilledAutoBlocks()
            blocks.sub_blocks.pop("prompt_upsampler")
            names = {i.name for i in blocks.inputs if i.name}
        else:
            from diffusers import Flux2KleinPipeline
            names = set(inspect.signature(Flux2KleinPipeline.__call__).parameters)
    except Exception as e:
        return f"skipped (diffusers not importable: {e.__class__.__name__}: {e})"
    unknown = sorted(set(call) - names)
    return f"unknown arguments: {', '.join(unknown)}" if unknown else ""


CHUNK_BYTES = 1 << 30  # score matrix budget per chunk (bytes), well under a 32 GB card's headroom


def chunked_sdpa(torch, real, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    """Exact scaled-dot-product attention in query chunks, for builds with no fused kernel.

    The ROCm wheel has no flash or memory-efficient kernel for gfx1030, so PyTorch falls back to the
    math path, which builds the whole (heads x tokens x tokens) score matrix: about 20 GB for a
    one-megapixel page plus two references. Splitting the queries gives the same result in bounded memory.
    """
    if q.dim() != 4 or is_causal or dropout_p:
        return real(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    b, h, n, _ = q.shape
    step = max(1, CHUNK_BYTES // max(1, b * h * k.shape[2] * 4))
    if step >= n:
        return real(q, k, v, attn_mask=attn_mask, scale=scale, **kw)
    out = []
    for i in range(0, n, step):
        m = attn_mask
        if m is not None and m.dim() >= 3 and m.shape[-2] == n:
            m = m[..., i:i + step, :]
        out.append(real(q[:, :, i:i + step], k, v, attn_mask=m, scale=scale, **kw))
    return torch.cat(out, dim=2)


def fused_attention_available(torch) -> bool:
    """True when PyTorch has a flash or memory-efficient SDPA kernel for the current device."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.zeros(1, 2, 64, 64, device="cuda", dtype=torch.bfloat16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        return False


class DiffusersEngine:
    """Keeps one diffusers pipeline in memory for a batch of pages (mirrors ``colourise.Engine``)."""

    def __init__(self, s, log=print, choice: DeviceChoice | None = None, pipe=None):
        os.environ.setdefault("HF_HOME", s.hf_home)
        import torch

        self.s, self.log, self.torch = s, log, torch
        self.choice = choice or choose(s.device, s.dtype)
        self.dtype = getattr(torch, self.choice.dtype)
        self.offload = bool(s.low_ram and self.choice.is_gpu)
        self._embeds: dict[str, tuple] = {}
        summary = load_summary(s, self.choice)
        for line in format_load(summary):
            log(line.replace("diffusers: ", "colourise: loading ", 1))
        self.pipe = pipe if pipe is not None else self._load(resolve_repo(s))
        if s.engine == "flux2-lora":
            self._setup_flux2()
        self._place()
        self._attention_patch()

    def _attention_patch(self) -> None:
        """Swap in chunked attention on a GPU with no fused kernel (override: DANDADYE_CHUNKED_ATTENTION=0|1)."""
        torch = self.torch
        force = os.environ.get("DANDADYE_CHUNKED_ATTENTION")
        want = force == "1" or (force != "0" and self.choice.is_gpu and not fused_attention_available(torch))
        if not want:
            return
        real = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = (
            lambda *a, **kw: chunked_sdpa(torch, real, *a, **kw))
        self.log("  note: no fused attention kernel on this device, using chunked attention (same result, bounded memory)")

    @property
    def device_info(self) -> dict:
        return {"device": self.choice.device, "dtype": self.choice.dtype, "device_detail": self.choice.detail}

    def _load(self, repo: str):
        if self.s.engine == "flux3":
            from diffusers import ModularPipeline
            blocks = ModularPipeline.from_pretrained(repo).blocks
            blocks.sub_blocks.pop("prompt_upsampler", None)  # flux3_prompt is already a full caption
            pipe = blocks.init_pipeline(repo)
            pipe.load_components(dtype={"vae": self.torch.float32, "default": self.dtype})
            return pipe
        from diffusers import Flux2KleinPipeline
        return Flux2KleinPipeline.from_pretrained(repo, dtype=self.dtype)

    def _setup_flux2(self) -> None:
        s, pipe = self.s, self.pipe
        if s.uses_lora:
            pipe.load_lora_weights(s.lora_path, adapter_name=ADAPTER)
            if s.bake_lora:
                pipe.fuse_lora(lora_scale=s.lora_scale, adapter_names=[ADAPTER])
                pipe.unload_lora_weights()  # drops the LoRA layers; the fused weights stay
            else:
                pipe.set_adapters([ADAPTER], [s.lora_scale])
        self.cfg = guidance_scale(s) > 1 and not pipe.config.get("is_distilled", False)
        if guidance_scale(s) > 1 and not self.cfg:
            self.log(f"  note: {s.model} is a distilled checkpoint, so guidance {s.guidance:g} is ignored "
                     "(the LoRA was trained on the undistilled base)")

    def _place(self) -> None:
        pipe, dev = self.pipe, self.choice.device
        if self.offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload(device=dev)
        else:
            self.offload = False
            pipe.to(dev)
        if self.s.low_ram:
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "enable_tiling"):
                vae.enable_tiling()

    def _encode(self, prompt: str) -> tuple:
        """(prompt_embeds, negative_prompt_embeds or None), encoded once per prompt."""
        if prompt in self._embeds:
            return self._embeds[prompt]
        pipe = self.pipe
        if pipe.text_encoder is None:
            raise RuntimeError("the text encoder was freed after the first prompt (--low-ram); "
                               "this engine expects one fixed prompt per batch")
        device = pipe._execution_device
        with self.torch.no_grad():  # without this the autograd graph pins the encoder weights on the card
            pos, _ = pipe.encode_prompt(prompt=prompt, device=device)
            neg = pipe.encode_prompt(prompt="", device=device)[0] if self.cfg else None
        self._embeds[prompt] = (pos, neg)
        if self.s.low_ram:
            self._free_text_encoder()
        return pos, neg

    def _free_text_encoder(self) -> None:
        pipe = self.pipe
        if self.offload:  # the offload hooks still hold the encoder (and may leave it on the GPU)
            pipe.remove_all_hooks()
        enc = pipe.text_encoder
        if enc is not None and self.choice.is_gpu:
            enc.to("cpu")  # off the card before the 9B transformer needs the room
        pipe.text_encoder = None
        del enc
        if self.offload:
            pipe.enable_model_cpu_offload(device=self.choice.device)
        gc.collect()
        if self.choice.device.startswith("cuda"):
            self.torch.cuda.empty_cache()

    def generate(self, page_in: Path, refs: list[Path], out: Path, width: int, height: int, prompt: str) -> None:
        s = self.s
        images = []
        for p in [page_in, *refs]:
            with Image.open(p) as im:
                images.append(im.convert("RGB"))
        generator = self.torch.Generator("cpu").manual_seed(s.seed)
        if s.engine == "flux3":
            image = self.pipe(prompt=prompt, image=images, height=height, width=width,
                              num_inference_steps=s.steps, generator=generator, output="images")[0]
        else:
            pos, neg = self._encode(prompt)
            kw = dict(image=images, prompt_embeds=pos, height=height, width=width,
                      num_inference_steps=s.steps, guidance_scale=guidance_scale(s), generator=generator)
            if neg is not None:
                kw["negative_prompt_embeds"] = neg
            image = self.pipe(**kw).images[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
