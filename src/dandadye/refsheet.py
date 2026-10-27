"""Stage 3 - refsheet: build the colour reference image fed to the LoRA as image 2.

A builder is any function ``(characters, palette, megapixels) -> PIL.Image``.
Register new ones in ``BUILDERS``. The default, ``cast_strip``, lines the
characters' official art up on white and adds a strip of palette swatches
under each figure.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from .config import Character, Config
from .state import State

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
Builder = Callable[[list[Character], dict, float], Image.Image]


def load_palette(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def swatches_for(palette: dict, key: str | None, limit: int = 6) -> list[str]:
    """Base hex colours of a character, high-confidence regions first."""
    regions = palette.get(key or "", {})
    order = {"high": 0, "medium": 1, "low": 2}
    items = [v for v in regions.values() if isinstance(v, dict) and HEX.match(str(v.get("base", "")))]
    items.sort(key=lambda v: order.get(v.get("confidence"), 3))
    out: list[str] = []
    for v in items:
        if v["base"].lower() not in out:
            out.append(v["base"].lower())
    return out[:limit]


def flatten_art(path: Path, background=(255, 255, 255)) -> Image.Image:
    """Composite transparent art on a plain background and crop to the figure."""
    with Image.open(path) as im:
        im = im.convert("RGBA")
    bbox = im.getchannel("A").getbbox() or (0, 0, im.width, im.height)
    im = im.crop(bbox)
    bg = Image.new("RGB", im.size, background)
    bg.paste(im, mask=im.getchannel("A"))
    return bg


def snap16(w: float, h: float, megapixels: float) -> tuple[int, int]:
    """Scale (w, h) to ~megapixels keeping aspect, each side a multiple of 16."""
    s = math.sqrt(megapixels * 1024 * 1024 / (w * h))
    return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)


def cast_strip(characters: list[Character], palette: dict, megapixels: float = 1.0) -> Image.Image:
    figure_h, swatch_h, pad = 1000, 90, 30
    tiles = []
    for c in characters:
        if c.art and Path(c.art).exists():
            art = flatten_art(Path(c.art))
            art = art.resize((max(1, round(art.width * figure_h / art.height)), figure_h), Image.LANCZOS)
        else:
            art = Image.new("RGB", (figure_h // 3, figure_h), "white")
        tiles.append((art, swatches_for(palette, c.palette_key)))
    if not tiles:
        tiles = [(Image.new("RGB", (400, figure_h), "white"), [])]

    width = sum(t.width for t, _ in tiles) + pad * (len(tiles) + 1)
    height = pad + figure_h + pad + swatch_h + pad
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = pad
    for art, sw in tiles:
        sheet.paste(art, (x, pad))
        if sw:
            cell = art.width / len(sw)
            for i, hx in enumerate(sw):
                x0 = x + round(i * cell)
                x1 = x + round((i + 1) * cell) - 4
                draw.rectangle([x0, pad + figure_h + pad, x1, pad + figure_h + pad + swatch_h], fill=hx)
        x += art.width + pad
    return sheet.resize(snap16(width, height, megapixels), Image.LANCZOS)


BUILDERS: dict[str, Builder] = {"cast_strip": cast_strip}


def characters_for_page(cfg: Config, page_data: dict, max_characters: int) -> tuple[list[Character], bool]:
    """Named characters on the page (most frequent first), or the main cast as fallback."""
    det = page_data.get("detect", {})
    counts = det.get("name_counts", {})
    names = [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if n != "Other"]
    chars = [c for c in (cfg.preset.character(n) for n in names) if c is not None][:max_characters]
    if chars:
        return chars, False
    return cfg.preset.main_cast[:max_characters], True


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None = None,
        builder: str = "cast_strip", max_characters: int = 4, force: bool = False,
        megapixels: float = 1.0, log=print) -> None:
    build = BUILDERS[builder]
    palette = load_palette(cfg.resolve(cfg.preset.palette))
    out_root = cfg.stage_dir("refsheet")
    cache: dict[tuple, Path] = {}
    done = 0
    for key, cid, idx, data in state.pages(chapters):
        if pages and idx not in pages:
            continue
        chars, fallback = characters_for_page(cfg, data, max_characters)
        names = tuple(c.name for c in chars)
        prev = data.get("refsheet", {})
        if not force and prev.get("characters") == list(names) and Path(prev.get("path", "")).exists():
            continue
        # Pages with the same cast share one sheet file.
        if names not in cache:
            slug = "_".join(re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-") for n in names) or "empty"
            path = out_root / f"{builder}__{slug}.png"
            if force or not path.exists():
                build(chars, palette, megapixels).save(path)
            cache[names] = path
        state.update(key, "refsheet", {"path": str(cache[names]), "characters": list(names),
                                       "fallback": fallback, "builder": builder})
        done += 1
    log(f"refsheet: {done} pages updated, {len(cache)} distinct sheets in {out_root}")
