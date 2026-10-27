"""Stage 4 - colourise: FLUX.2 Klein (base 9B) edit + the manga-colourisation LoRA.

What the LoRA expects (from its README and ComfyUI workflow):

* image 1 = the black-and-white page, image 2 = the colour reference sheet;
* both scaled to ~1 megapixel (ComfyUI ``ImageScaleToTotalPixels``), and the
  output canvas takes image 1's scaled size;
* prompt = the trigger word ``mngclranm``, optionally followed by a short
  description; empty negative prompt;
* base (undistilled) model, ~50 steps, guidance ~4 (the shipped workflow uses
  20 steps / CFG 5), LoRA weight 0.8-1.0.

mflux concatenates the reference latents with distinct time offsets exactly like
ComfyUI's ReferenceLatent nodes, so order = ``--image-paths page.png ref.png``.

Two backends: ``api`` loads the model once and loops over pages (the prompt is
the same for every page, so its embedding is computed once and the 8B text
encoder is then freed); ``cli`` runs one ``mflux-generate-flux2-edit`` per page.
"""

from __future__ import annotations

import gc
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image

from .config import Config
from .refsheet import snap16
from .state import State

TRIGGER = "mngclranm"


@dataclass
class Settings:
    model: str = "flux2-klein-base-9b"
    model_path: str | None = None
    quantize: int | None = 4
    steps: int = 50
    guidance: float = 4.0
    lora_path: str = "lora/manga-colorization/manga_colorization.safetensors"
    lora_scale: float = 0.9
    megapixels: float = 1.0
    seed: int = 42
    prompt_extra: str = ""
    low_ram: bool = True
    mflux_bin: str = ".venv/bin/mflux-generate-flux2-edit"
    hf_home: str = "hf"

    @property
    def prompt(self) -> str:
        return f"{TRIGGER}, {self.prompt_extra}" if self.prompt_extra else TRIGGER

    @classmethod
    def from_config(cls, cfg: Config, args=None) -> "Settings":
        s = cls()
        for k, v in (cfg.preset.colourise or {}).items():
            if k in {f.name for f in fields(cls)}:
                setattr(s, k, v)
        if args is not None:
            for k in ("model", "model_path", "quantize", "steps", "guidance", "lora_scale",
                      "megapixels", "seed", "prompt_extra", "low_ram"):
                v = getattr(args, k, None)
                if v is not None:
                    setattr(s, k, v)
            if getattr(args, "no_quantize", False):
                s.quantize = None
        # resolve data-root relative paths
        for k in ("lora_path", "mflux_bin", "hf_home", "model_path"):
            v = getattr(s, k)
            if v:
                setattr(s, k, str(cfg.resolve(v)))
        return s


def working_size(w: int, h: int, megapixels: float) -> tuple[int, int]:
    return snap16(w, h, megapixels)


def prepare_page(src: Path, dst: Path, megapixels: float) -> tuple[int, int]:
    """Greyscale page -> RGB at ~megapixels with 16-px aligned sides (image 1)."""
    with Image.open(src) as im:
        g = im.convert("L")
    size = working_size(g.width, g.height, megapixels)
    dst.parent.mkdir(parents=True, exist_ok=True)
    g.resize(size, Image.LANCZOS).convert("RGB").save(dst)
    return size


def prepare_reference(src: Path, dst: Path, megapixels: float) -> Path:
    """Reference sheet at ~megapixels (mflux only downscales; ComfyUI also upscales)."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        size = snap16(im.width, im.height, megapixels)
        if im.size == size:
            return src
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.resize(size, Image.LANCZOS).save(dst)
    return dst


def build_command(s: Settings, page_in: Path, ref: Path, out: Path, width: int, height: int) -> list[str]:
    cmd = [s.mflux_bin]
    if s.model_path:
        cmd += ["--model", s.model_path, "--base-model", s.model]
    else:
        cmd += ["--model", s.model]
    if s.quantize:
        cmd += ["-q", str(s.quantize)]
    cmd += ["--image-paths", str(page_in), str(ref),
            "--lora-paths", s.lora_path, "--lora-scales", f"{s.lora_scale:g}",
            "--prompt", s.prompt,
            "--steps", str(s.steps), "--guidance", f"{s.guidance:g}",
            "--width", str(width), "--height", str(height),
            "--seed", str(s.seed), "--output", str(out)]
    if s.low_ram:
        cmd.append("--low-ram")
    return cmd


def validate_with_mflux(cmd: list[str]) -> str:
    """Parse argv with mflux's own parser (no model load). Returns '' if OK."""
    try:
        from mflux.models.flux2.cli.flux2_edit_generate import build_parser
    except Exception as e:  # mflux not installed in this interpreter
        return f"skipped (mflux not importable: {e.__class__.__name__})"
    old = sys.argv
    try:
        sys.argv = cmd
        args = build_parser().parse_args()
    except SystemExit as e:
        return f"mflux rejected the arguments (exit {e.code})"
    finally:
        sys.argv = old
    problems = []
    if [str(p) for p in args.image_paths] != cmd[cmd.index("--image-paths") + 1: cmd.index("--image-paths") + 3]:
        problems.append("image order")
    if not args.lora_paths:
        problems.append("lora not parsed")
    return ", ".join(problems)


class Engine:
    """Keeps one Flux2KleinEdit in memory for a batch of pages."""

    def __init__(self, s: Settings, log=print):
        os.environ.setdefault("HF_HOME", s.hf_home)
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.models.flux2.variants import Flux2KleinEdit

        model_config = AVAILABLE_MODELS[s.model]
        log(f"colourise: loading {s.model} (q={s.quantize}) + LoRA {Path(s.lora_path).name} x{s.lora_scale}")
        self.model = Flux2KleinEdit(quantize=s.quantize, model_path=s.model_path,
                                    lora_paths=[s.lora_path], lora_scales=[s.lora_scale],
                                    model_config=model_config)
        self.s = s
        self._cached = None
        original = self.model._encode_prompt_pair

        def cached_encode(**kw):
            if self._cached is None:
                self._cached = original(**kw)
            return self._cached

        self.model._encode_prompt_pair = cached_encode
        if s.low_ram:
            import mlx.core as mx
            mx.set_cache_limit(1000**3)
            try:
                from mflux.models.common.vae.tiling_config import TilingConfig
                self.model.tiling_config = TilingConfig()
            except Exception:
                pass

    def generate(self, page_in: Path, ref: Path, out: Path, width: int, height: int) -> None:
        s = self.s
        image = self.model.generate_image(seed=s.seed, prompt=s.prompt, num_inference_steps=s.steps,
                                          width=width, height=height, guidance=s.guidance,
                                          image_paths=[page_in, ref])
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=out, overwrite=True)
        if s.low_ram and self.model.text_encoder is not None and self._cached is not None:
            self.model.text_encoder = None  # prompt embedding is cached; free ~4-8 GB
            gc.collect()


def should_skip(data: dict) -> str | None:
    ing = data.get("ingest", {})
    for flag in ("colour", "blank", "sparse", "error"):
        if ing.get(flag):
            return flag
    return None


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None, s: Settings,
        dry_run: bool = False, backend: str = "api", force: bool = False,
        include_skipped: bool = False, log=print) -> list[list[str]]:
    todo = []
    for key, cid, idx, data in state.pages(chapters):
        if pages and idx not in pages:
            continue
        reason = should_skip(data)
        if reason and not include_skipped:
            continue
        out = cfg.work / "colour" / cid / f"{idx:03d}.png"
        if out.exists() and not force:
            continue
        ref = data.get("refsheet", {}).get("path")
        if not ref:
            log(f"  {key}: no reference sheet - run `dandadye refsheet` first; skipping")
            continue
        todo.append((key, cid, idx, data, Path(ref), out))

    log(f"colourise: {len(todo)} pages to do ({backend} backend{', dry run' if dry_run else ''})")
    commands, engine = [], None
    for n, (key, cid, idx, data, ref, out) in enumerate(todo, start=1):
        page_in = cfg.work / "colourise_in" / cid / f"{idx:03d}.png"
        if dry_run:
            with Image.open(data["ingest"]["src"]) as im:
                width, height = working_size(im.width, im.height, s.megapixels)
            ref_in = ref
        else:
            width, height = prepare_page(Path(data["ingest"]["src"]), page_in, s.megapixels)
            ref_in = prepare_reference(ref, cfg.work / "colourise_in" / "refs" / ref.name, s.megapixels)
        cmd = build_command(s, page_in, ref_in, out, width, height)
        commands.append(cmd)
        if dry_run:
            log(shlex.join(cmd))
            continue
        t0 = time.time()
        if backend == "cli":
            env = {**os.environ, "HF_HOME": s.hf_home}
            subprocess.run(cmd, check=True, env=env)
        else:
            engine = engine or Engine(s, log)
            engine.generate(page_in, ref_in, out, width, height)
        dt = time.time() - t0
        state.update(key, "colourise", {"path": str(out), "seconds": round(dt, 1),
                                        "width": width, "height": height, "settings": asdict(s)})
        log(f"  [{n}/{len(todo)}] {key} -> {out.name} in {dt:.0f}s")
    if dry_run and commands:
        problem = validate_with_mflux(commands[0])
        log(f"mflux argument check: {problem or 'OK'}")
    return commands
