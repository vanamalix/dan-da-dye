"""Stage 7 - qa: check coloured pages for drift against the palette.

This is a placeholder interface. A later version will ask a vision model
"does Momo's hair match #673132?" per detected character. For now the
``heuristic`` checker only measures how far each detected character's
dominant colours are from its palette swatches, and flags pages that came
out (nearly) grey.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .config import Config
from .refsheet import load_palette, swatches_for
from .state import State
from .util import hex_to_rgb, rgb_to_lab


@dataclass
class QAResult:
    status: str                      # "ok" | "warn" | "fail" | "skipped"
    score: float | None = None       # 0..1, higher = closer to the palette
    notes: list[str] = field(default_factory=list)


class Checker(Protocol):
    def check(self, image: Image.Image, page: dict, palette: dict, cfg: Config) -> QAResult: ...


class HeuristicChecker:
    """Cheap stand-in until a vision model is wired in."""

    def check(self, image: Image.Image, page: dict, palette: dict, cfg: Config) -> QAResult:
        small = image.convert("RGB")
        small.thumbnail((512, 512))
        lab = rgb_to_lab(np.asarray(small))
        colourful = float(np.mean(np.hypot(lab[..., 1], lab[..., 2]) > 12))
        if colourful < 0.03:
            return QAResult("fail", 0.0, [f"page looks uncoloured ({colourful:.1%} coloured pixels)"])
        scale = small.width / image.width
        dists = []
        notes = []
        for ch in page.get("detect", {}).get("characters", []):
            c = cfg.preset.character(ch.get("name") or "")
            sw = swatches_for(palette, c.palette_key) if c else []
            if not sw:
                continue
            x1, y1, x2, y2 = (int(v * scale) for v in ch["box"])
            crop = lab[max(0, y1):y2, max(0, x1):x2].reshape(-1, 3)
            if len(crop) < 20:
                continue
            ref = rgb_to_lab(np.array([[hex_to_rgb(h) for h in sw]], dtype=np.uint8))[0]
            d = np.linalg.norm(crop[:, None, :] - ref[None, :, :], axis=2).min(1)
            dists.append(float(np.median(d)))
            notes.append(f"{c.name}: median dE to palette {dists[-1]:.1f}")
        if not dists:
            return QAResult("ok", None, ["no named characters to compare"])
        score = float(np.clip(1 - np.mean(dists) / 50, 0, 1))
        return QAResult("ok" if score > 0.5 else "warn", round(score, 3), notes)


CHECKERS: dict[str, type] = {"heuristic": HeuristicChecker}


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None = None,
        checker: str = "heuristic", force: bool = False, log=print) -> None:
    chk = CHECKERS[checker]()
    palette = load_palette(cfg.resolve(cfg.preset.palette))
    tally: dict[str, int] = {}
    for key, cid, idx, data in state.pages(chapters):
        if pages and idx not in pages:
            continue
        path = data.get("inkfix", {}).get("path")
        if not path or not Path(path).exists() or (data.get("qa") and not force):
            continue
        with Image.open(path) as im:
            res = chk.check(im, data, palette, cfg)
        state.update(key, "qa", {"checker": checker, **asdict(res)})
        tally[res.status] = tally.get(res.status, 0) + 1
    log(f"qa ({checker}): {tally or 'nothing to check'}")
