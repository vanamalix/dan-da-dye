"""Character bank for Magi: crops of each preset character's official art."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from ..config import Config
from ..refsheet import flatten_art


def bank_images(art: Path) -> list[Image.Image]:
    """Full figure and upper body (head + torso), greyscale on white like a manga page."""
    fig = flatten_art(art).convert("L")
    upper = fig.crop((0, 0, fig.width, max(1, int(fig.height * 0.38))))
    return [fig, upper]


def build_bank(cfg: Config) -> list[tuple[str, Path]]:
    out = cfg.stage_dir("detect/bank")
    bank = []
    for c in cfg.preset.characters:
        if not c.art or not Path(c.art).exists():
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", c.name.lower()).strip("-")
        for i, im in enumerate(bank_images(Path(c.art))):
            p = out / f"{slug}_{i}.png"
            if not p.exists():
                im.save(p)
            bank.append((c.name, p))
    return bank
