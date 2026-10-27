"""Stage 5 - inkfix: put the original linework and lettering back over the colour.

Diffusion output is ~1 MP and redraws (and sometimes blurs or misplaces) the
ink. We keep only its colour:

1. resize the colour image to the page's working size;
2. remove the model's own thin dark lines with a morphological closing;
3. multiply the original greyscale page over it (ink and screentone survive
   exactly, colour sits underneath like a flat in a multiply layer);
4. force true ink to neutral black, near-white paper back to white, and
   restore text boxes (speech balloons, lettering) from the original.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import Config
from .state import State
from .util import rgb_to_lab

INK_LEVEL = 0.30      # original grey below this = ink -> neutral
INK_SOFT = 0.08       # soft edge width for the ink mask
PAPER_LEVEL = 0.96    # original grey above this ...
PAPER_CHROMA = 8.0    # ... and colour chroma below this = clean white paper


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _box_mask(shape: tuple[int, int], boxes, scale: float, grow: float = 0.08) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    for b in boxes or []:
        x1, y1, x2, y2 = (v * scale for v in b[:4])
        gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
        xa, ya = max(0, int(x1 - gx)), max(0, int(y1 - gy))
        xb, yb = min(w, int(np.ceil(x2 + gx))), min(h, int(np.ceil(y2 + gy)))
        m[ya:yb, xa:xb] = True
    return m


def balloon_mask(g: np.ndarray, boxes, scale: float, grow: float = 0.6, level: float = 0.85) -> np.ndarray:
    """White paper connected to each text box, within a grown box: the balloon interior.

    Flood-fills from the white pixels inside the text box, so the fill stops at the
    balloon's ink outline (or at the grown box if the balloon is open).
    """
    h, w = g.shape
    out = np.zeros((h, w), dtype=bool)
    for b in boxes or []:
        x1, y1, x2, y2 = (v * scale for v in b[:4])
        gx, gy = (x2 - x1) * grow, (y2 - y1) * grow
        xa, ya, xb, yb = max(0, int(x1 - gx)), max(0, int(y1 - gy)), min(w, int(x2 + gx)), min(h, int(y2 + gy))
        if xb - xa < 3 or yb - ya < 3:
            continue
        region = Image.fromarray(((g[ya:yb, xa:xb] > level) * 255).astype(np.uint8), "L").copy()  # writable
        seeds_y, seeds_x = np.nonzero(np.asarray(region)[int(y1 - ya):int(y2 - ya), int(x1 - xa):int(x2 - xa)])
        step = max(1, len(seeds_x) // 40)
        for sx, sy in zip(seeds_x[::step], seeds_y[::step]):
            pt = (int(sx + x1 - xa), int(sy + y1 - ya))
            if region.getpixel(pt) == 255:
                ImageDraw.floodfill(region, pt, 128)
        out[ya:yb, xa:xb] |= np.asarray(region) == 128
    return out


def recombine(original: Image.Image, colour: Image.Image, text_boxes=None, panel_boxes=None,
              upscale_to_colour: bool = True) -> Image.Image:
    """Return an RGB page: original ink and lettering over the model's colour.

    Boxes are [x1, y1, x2, y2] in original-image pixels.
    """
    ow, oh = original.size
    scale = max(1.0, colour.height / oh) if upscale_to_colour else 1.0
    size = (round(ow * scale), round(oh * scale))

    grey = original.convert("L")
    if grey.size != size:
        grey = grey.resize(size, Image.LANCZOS)
    g = np.asarray(grey, dtype=np.float64) / 255.0

    col = colour.convert("RGB").resize(size, Image.BICUBIC)
    k = max(3, (min(size) // 400) | 1)  # odd kernel ~ line width
    clean = col.filter(ImageFilter.MaxFilter(k)).filter(ImageFilter.MinFilter(k))
    c = np.asarray(clean, dtype=np.float64) / 255.0

    out = c * g[..., None]  # multiply blend

    ink = 1.0 - _smoothstep(INK_LEVEL - INK_SOFT, INK_LEVEL, g)
    out = out * (1 - ink[..., None]) + g[..., None] * ink[..., None]

    chroma = np.hypot(*np.moveaxis(rgb_to_lab(np.asarray(clean))[..., 1:], -1, 0))
    paper = (g > PAPER_LEVEL) & (chroma < PAPER_CHROMA)
    if panel_boxes:
        paper |= (g > 0.9) & ~_box_mask(g.shape, panel_boxes, scale, grow=0.0)
    if text_boxes:
        tm = _box_mask(g.shape, text_boxes, scale)
        out[tm] = g[tm][:, None]  # lettering straight from the original
        paper |= balloon_mask(g, text_boxes, scale)  # and the rest of the balloon white
    out[paper] = 1.0

    return Image.fromarray(np.clip(np.round(out * 255), 0, 255).astype(np.uint8), "RGB")


def colour_path(cfg: Config, cid: str, idx: int, engine: str = "flux2-lora",
                colour_dir: str | None = None) -> Path:
    root = Path(colour_dir) if colour_dir else cfg.work / "colour" / engine
    return root / cid / f"{idx:03d}.png"


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None = None,
        force: bool = False, colour_dir: str | None = None, engine: str = "flux2-lora", log=print) -> None:
    """Recombine the colour of one engine; the page's published version follows the last run."""
    done = missing = 0
    for key, cid, idx, data in state.pages(chapters):
        if pages and idx not in pages:
            continue
        src = colour_path(cfg, cid, idx, engine, colour_dir)
        if not src.exists():
            missing += 1
            continue
        out = cfg.stage_dir(f"inkfix/{engine}/{cid}") / f"{idx:03d}.png"
        if out.exists() and not force and data.get("inkfix", {}).get("path") == str(out) \
                and out.stat().st_mtime >= src.stat().st_mtime:
            continue
        det = data.get("detect", {})
        with Image.open(data["ingest"]["src"]) as orig, Image.open(src) as colour:
            result = recombine(orig, colour, det.get("texts"), det.get("panels"))
        result.save(out)
        state.update(key, "inkfix", {"path": str(out), "from": str(src), "engine": engine, "w": result.width, "h": result.height})
        done += 1
    log(f"inkfix: {done} pages written, {missing} without a colour image yet")
