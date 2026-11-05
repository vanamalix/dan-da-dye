"""Stage 4 - colourise: reference-guided diffusion via mflux.

Two engines (pick with ``--engine``, compare A/B; outputs go to
``work/colour/<engine>/<chapter>/<NNN>.png``):

``flux2-lora`` (default) - FLUX.2 Klein base 9B + the manga-colourisation LoRA.
  What the LoRA expects (from its README and ComfyUI workflow):

  * image 1 = the black-and-white page, image 2 = the colour reference sheet;
  * both scaled to ~1 megapixel (ComfyUI ``ImageScaleToTotalPixels``); the
    output canvas takes image 1's scaled size;
  * prompt = the trigger word ``mngclranm``, optionally followed by a short
    description; empty negative prompt;
  * base (undistilled) model, ~50 steps, guidance ~4 (the shipped workflow uses
    20 steps / CFG 5), LoRA weight 0.8-1.0.

  mflux concatenates reference latents with distinct time offsets like ComfyUI's
  ReferenceLatent nodes, so the order is ``--image-paths page.png ref.png``.

``flux3`` - FLUX 3 Image Klein (distilled, 16 steps, no guidance, no LoRA).
  Takes up to 10 references, so we pass the page plus one small sheet per
  character, and a full caption naming each character's palette (the model
  works best with a complete description of the target image).

Two backends: ``api`` loads the model once and loops over pages; ``cli`` runs one
mflux process per page. ``--dry-run`` only prints the equivalent CLI commands.
"""

from __future__ import annotations

import gc
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image

from .config import Config
from .refsheet import cast_strip, load_palette, snap16
from .state import State

TRIGGER = "mngclranm"

ENGINES = {
    "flux2-lora": dict(model="flux2-klein-base-9b", steps=50, guidance=4.0, quantize=4,
                       mflux_bin=".venv/bin/mflux-generate-flux2-edit", max_refs=1, uses_lora=True),
    "flux3": dict(model="flux3-image-klein", steps=16, guidance=None, quantize=4,
                  mflux_bin=".venv/bin/mflux-generate-flux3-edit", max_refs=9, uses_lora=False),
}


@dataclass
class Settings:
    engine: str = "flux2-lora"
    model: str = "flux2-klein-base-9b"
    model_path: str | None = None
    quantize: int | None = 4
    steps: int = 50
    guidance: float | None = 4.0
    lora_path: str = "lora/manga-colorization/manga_colorization.safetensors"
    lora_scale: float = 0.9
    megapixels: float = 1.0
    seed: int = 42
    prompt_extra: str = ""
    low_ram: bool = True
    # Fold the LoRA into the quantised weights at load time (faster steps). On 16 GB Macs the fold can trip
    # macOS's GPU watchdog ("GPU Timeout Error"); --no-bake-lora applies the LoRA on the fly instead.
    bake_lora: bool = True
    mflux_bin: str = ".venv/bin/mflux-generate-flux2-edit"
    hf_home: str = "hf"

    @property
    def uses_lora(self) -> bool:
        return ENGINES[self.engine]["uses_lora"]

    @property
    def max_refs(self) -> int:
        return ENGINES[self.engine]["max_refs"]

    @property
    def prompt(self) -> str:
        """The fixed prompt of the LoRA engine (flux3 builds one per page)."""
        return f"{TRIGGER}, {self.prompt_extra}" if self.prompt_extra else TRIGGER

    @classmethod
    def from_config(cls, cfg: Config, args=None, engine: str | None = None) -> "Settings":
        engine = engine or getattr(args, "engine", None) or "flux2-lora"
        s = cls(engine=engine)
        names = {f.name for f in fields(cls)}
        for k, v in ENGINES[engine].items():
            if k in names:
                setattr(s, k, v)
        # preset [colourise] = flux2-lora settings (+ prompt_extra for all); [colourise.<engine>] per engine
        preset = cfg.preset.colourise or {}
        flat = {k: v for k, v in preset.items() if not isinstance(v, dict)}
        if engine != "flux2-lora":
            flat = {k: v for k, v in flat.items() if k == "prompt_extra"}
        for k, v in {**flat, **preset.get(engine, {})}.items():
            if k in names:
                setattr(s, k, v)
        if args is not None:
            for k in ("model", "model_path", "quantize", "steps", "guidance", "lora_scale",
                      "megapixels", "seed", "prompt_extra", "low_ram"):
                v = getattr(args, k, None)
                if v is not None:
                    setattr(s, k, v)
            if getattr(args, "no_quantize", False):
                s.quantize = None
            if getattr(args, "no_bake_lora", False):
                s.bake_lora = False
        for k in ("lora_path", "mflux_bin", "hf_home", "model_path"):
            v = getattr(s, k)
            if v:
                setattr(s, k, str(cfg.resolve(v)))
        return s


# --- inputs ---------------------------------------------------------------------

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


def solo_sheets(cfg: Config, names: list[str], limit: int, build: bool = True) -> list[Path]:
    """One small art+swatch sheet per character (flux3 takes them as separate references)."""
    palette = load_palette(cfg.resolve(cfg.preset.palette))
    out = []
    for name in names[:limit]:
        c = cfg.preset.character(name)
        if c is None:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        p = cfg.work / "refsheet" / "solo" / f"{slug}.png"
        if build and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            cast_strip([c], palette, megapixels=0.4).save(p)
        out.append(p)
    return out


def _region_words(region: str) -> str:
    return region.replace("_", " ")


def flux3_prompt(cfg: Config, names: list[str], extra: str = "") -> str:
    """Full caption for FLUX 3: the target image described, references named in order."""
    palette = load_palette(cfg.resolve(cfg.preset.palette))
    parts = [
        "A fully coloured anime-style version of the black-and-white manga page in <ref_image_1>, "
        "with exactly the same panels, line art, composition and lettering; clean flat cel colours "
        "with soft shading, white speech balloons and black text kept unchanged."
    ]
    for i, name in enumerate(names, start=2):
        c = cfg.preset.character(name)
        regions = palette.get(c.palette_key or "", {}) if c else {}
        desc = [f"{_region_words(r)} {v['base']}" for r, v in regions.items()
                if isinstance(v, dict) and str(v.get("base", "")).startswith("#")][:5]
        colours = f" ({', '.join(desc)})" if desc else ""
        parts.append(f"{name} is coloured as in <ref_image_{i}>{colours}.")
    if extra:
        parts.append(extra)
    return " ".join(parts)


# --- commands ---------------------------------------------------------------------

def build_command(s: Settings, page_in: Path, refs: Path | list[Path], out: Path, width: int, height: int,
                  prompt: str | None = None) -> list[str]:
    refs = [refs] if isinstance(refs, Path) else list(refs)
    cmd = [s.mflux_bin]
    if s.model_path:
        cmd += ["--model", s.model_path, "--base-model", s.model]
    else:
        cmd += ["--model", s.model]
    if s.quantize:
        cmd += ["-q", str(s.quantize)]
    cmd += ["--image-paths", str(page_in), *[str(r) for r in refs]]
    if s.uses_lora:
        cmd += ["--lora-paths", s.lora_path, "--lora-scales", f"{s.lora_scale:g}"]
    cmd += ["--prompt", prompt or s.prompt, "--steps", str(s.steps)]
    if s.guidance is not None:
        cmd += ["--guidance", f"{s.guidance:g}"]
    cmd += ["--width", str(width), "--height", str(height), "--seed", str(s.seed), "--output", str(out)]
    if s.low_ram:
        cmd.append("--low-ram")
    return cmd


def validate_with_mflux(cmd: list[str]) -> str:
    """Parse argv with mflux's own parser (no model load). Returns '' if OK."""
    try:
        if "flux3" in Path(cmd[0]).name:
            from mflux.models.flux3.cli.flux3_edit_generate import build_parser
        else:
            from mflux.models.flux2.cli.flux2_edit_generate import build_parser
    except Exception as e:  # mflux missing or broken in this interpreter
        return f"skipped (mflux not importable: {e.__class__.__name__}: {e})"
    old = sys.argv
    try:
        sys.argv = cmd
        args = build_parser().parse_args()
    except SystemExit as e:
        return f"mflux rejected the arguments (exit {e.code})"
    finally:
        sys.argv = old
    problems = []
    i = cmd.index("--image-paths")
    n = len(args.image_paths)
    if [str(p) for p in args.image_paths] != cmd[i + 1:i + 1 + n]:
        problems.append("image order")
    if "--lora-paths" in cmd and not args.lora_paths:
        problems.append("lora not parsed")
    return ", ".join(problems)


# --- engines ---------------------------------------------------------------------

class Engine:
    """Keeps one mflux model in memory for a batch of pages."""

    def __init__(self, s: Settings, log=print):
        os.environ.setdefault("HF_HOME", s.hf_home)
        from mflux.models.common.config.model_config import AVAILABLE_MODELS

        self.s = s
        model_config = AVAILABLE_MODELS[s.model]
        log(f"colourise: loading {s.model} (q={s.quantize})"
            + (f" + LoRA {Path(s.lora_path).name} x{s.lora_scale}" if s.uses_lora else ""))
        if s.engine == "flux3":
            from mflux.models.flux3.variants import Flux3KleinEdit
            self.model = Flux3KleinEdit(model_config=model_config, quantize=s.quantize, model_path=s.model_path)
        else:
            from mflux.models.flux2.variants import Flux2KleinEdit
            self.model = Flux2KleinEdit(quantize=s.quantize, model_path=s.model_path,
                                        lora_paths=[s.lora_path], lora_scales=[s.lora_scale],
                                        bake_lora=s.bake_lora, model_config=model_config)
            # The LoRA prompt is identical for every page: encode it once, then the
            # text encoder can be freed (MemorySaver would otherwise drop it after page 1
            # and break page 2, because Flux2 re-encodes on every call).
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

    def generate(self, page_in: Path, refs: list[Path], out: Path, width: int, height: int, prompt: str) -> None:
        s = self.s
        kw = dict(seed=s.seed, prompt=prompt, num_inference_steps=s.steps, width=width, height=height,
                  image_paths=[page_in, *refs])
        if s.guidance is not None:
            kw["guidance"] = s.guidance
        image = self.model.generate_image(**kw)
        out.parent.mkdir(parents=True, exist_ok=True)
        image.save(path=out, overwrite=True)
        if s.engine != "flux3" and s.low_ram and getattr(self, "_cached", None) is not None \
                and self.model.text_encoder is not None:
            self.model.text_encoder = None
            gc.collect()


# --- stage ---------------------------------------------------------------------

def should_skip(data: dict) -> str | None:
    ing = data.get("ingest", {})
    for flag in ("colour", "blank", "sparse", "error"):
        if ing.get(flag):
            return flag
    return None


def colour_dir(cfg: Config, engine: str) -> Path:
    return cfg.work / "colour" / engine


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None, s: Settings,
        dry_run: bool = False, backend: str = "api", force: bool = False,
        include_skipped: bool = False, log=print) -> list[list[str]]:
    todo = []
    for key, cid, idx, data in state.pages(chapters):
        if pages and idx not in pages:
            continue
        if should_skip(data) and not include_skipped:
            continue
        out = colour_dir(cfg, s.engine) / cid / f"{idx:03d}.png"
        if out.exists() and not force:
            continue
        ref = data.get("refsheet", {}).get("path")
        if not ref:
            log(f"  {key}: no reference sheet - run `dandadye refsheet` first; skipping")
            continue
        todo.append((key, cid, idx, data, Path(ref), out))

    log(f"colourise: {len(todo)} pages to do ({s.engine}, {backend} backend{', dry run' if dry_run else ''})")
    commands, engine, spent = [], None, 0.0
    in_root = cfg.work / "colourise_in"
    for n, (key, cid, idx, data, ref, out) in enumerate(todo, start=1):
        page_in = in_root / cid / f"{idx:03d}.png"
        names = data["refsheet"].get("characters", [])
        if dry_run:
            with Image.open(data["ingest"]["src"]) as im:
                width, height = working_size(im.width, im.height, s.megapixels)
        else:
            width, height = prepare_page(Path(data["ingest"]["src"]), page_in, s.megapixels)
        if s.engine == "flux3":
            refs = solo_sheets(cfg, names, s.max_refs, build=not dry_run) or [ref]
            prompt = flux3_prompt(cfg, names if refs != [ref] else [], s.prompt_extra)
        else:
            refs = [ref if dry_run else prepare_reference(ref, in_root / "refs" / ref.name, s.megapixels)]
            prompt = s.prompt
        cmd = build_command(s, page_in, refs, out, width, height, prompt)
        commands.append(cmd)
        if dry_run:
            log(shlex.join(cmd))
            continue
        t0 = time.time()
        if backend == "cli":
            subprocess.run(cmd, check=True, env={**os.environ, "HF_HOME": s.hf_home})
        else:
            engine = engine or Engine(s, log)
            engine.generate(page_in, refs, out, width, height, prompt)
        dt = time.time() - t0
        rec = dict(data.get("colourise", {}))
        rec[s.engine] = {"path": str(out), "seconds": round(dt, 1), "width": width, "height": height,
                         "prompt": prompt, "settings": asdict(s)}
        state.update(key, "colourise", rec)
        spent += dt
        left = (spent / n) * (len(todo) - n)
        eta = time.strftime("%H:%M", time.localtime(time.time() + left))
        log(f"  [{n}/{len(todo)}] {key} -> {out} in {dt:.0f}s; about {left / 60:.0f} min left (done ~{eta})")
    if dry_run and commands:
        problem = validate_with_mflux(commands[0])
        log(f"mflux argument check: {problem or 'OK'}")
    return commands
