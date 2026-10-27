"""Synthetic data root: every image is drawn here with PIL. No real pages or art."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from dandadye.config import load_config
from dandadye.state import State


def bw_page(w=600, h=880, seed=0) -> Image.Image:
    """A fake manga page: panel borders, a stick figure, a balloon with 'text', a tone."""
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, w - 20, h // 2 - 10], outline=0, width=4)
    d.rectangle([20, h // 2 + 10, w - 20, h - 20], outline=0, width=4)
    cx = 150 + seed * 20
    d.ellipse([cx - 40, 80, cx + 40, 160], outline=0, width=3)          # head
    d.line([cx, 160, cx, 330], fill=0, width=3)                          # body
    d.rectangle([cx + 80, 200, cx + 230, 320], fill=150)                 # screentone-ish grey
    d.ellipse([w - 260, 60, w - 60, 180], fill=255, outline=0, width=3)  # balloon
    for i in range(3):
        d.line([w - 220, 95 + i * 25, w - 100, 95 + i * 25], fill=0, width=4)  # "text"
    d.polygon([(100, h - 100), (300, h - 300), (500, h - 100)], fill=40)  # big dark shape
    return im


def write_art(path: Path, colour: tuple[int, int, int]) -> None:
    im = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 10, 140, 90], fill=colour + (255,))
    d.rectangle([50, 100, 150, 390], fill=(40, 50, 120, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


PRESET = """
series = "testseries"
title = "Test Series"
source_dir = "work/src"
palette = "palette/palette.json"

[[characters]]
name = "Alice"
art = "palette/img/alice.png"
palette_key = "Alice A"
main_cast = true

[[characters]]
name = "Bob"
art = "palette/img/bob.png"
palette_key = "Bob B"
main_cast = true

[[characters]]
name = "Carol"
art = "palette/img/carol.png"
palette_key = "Carol C"
"""


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    src = tmp_path / "work" / "src"
    # chapter dir "10": 4 pages, "002-003" is a spread, page 5 missing
    ch = src / "10"
    ch.mkdir(parents=True)
    bw_page(seed=0).save(ch / "001.png")
    spread = Image.new("L", (1200, 880), 255)
    spread.paste(bw_page(seed=1), (0, 0))
    spread.paste(bw_page(seed=2), (600, 0))
    spread.save(ch / "002-003.jpg")
    bw_page(seed=3).save(ch / "004.jpg")
    bw_page(seed=4).save(ch / "006.jpg")
    # chapter dir "11" with chapter-prefixed names and a missing 002 next to a spread at 003
    ch = src / "11"
    ch.mkdir()
    bw_page().save(ch / "0011-001.png")
    Image.new("L", (1200, 880), 255).save(ch / "0011-003.png")
    Image.new("L", (600, 880), 255).save(ch / "0011-004.png")  # blank
    # sparse credit page: a few thin text lines
    cred = Image.new("L", (600, 880), 255)
    dc = ImageDraw.Draw(cred)
    for i in range(5):
        dc.line([200, 300 + 30 * i, 400, 300 + 30 * i], fill=0, width=2)
    cred.save(ch / "0011-005.png")
    # volume dir: two chapters inside, one colour page
    vol = src / "v01"
    vol.mkdir()
    colour = Image.new("RGB", (600, 880), (220, 120, 140))
    ImageDraw.Draw(colour).ellipse([100, 100, 400, 400], fill=(60, 140, 220))
    colour.save(vol / "Series - c001 (v01) - p000 [x].jpg")
    bw_page().save(vol / "Series - c001 (v01) - p001 [x].png")
    bw_page().save(vol / "Series - c001 (v01) - p010 [x].png")
    bw_page().save(vol / "Series - c001 (v01) - p002 [x].png")
    bw_page().save(vol / "Series - c002 (v01) - p011 [x].png")
    Image.new("L", (1200, 880), 255).save(vol / "Series - c002 (v01) - p012-p013 [x].png")
    # palette + art
    pal = tmp_path / "palette"
    write_art(pal / "img" / "alice.png", (200, 80, 90))
    write_art(pal / "img" / "bob.png", (30, 30, 35))
    write_art(pal / "img" / "carol.png", (230, 140, 160))
    palette = {
        "Alice A": {"hair": {"base": "#c8505a", "confidence": "high"},
                    "skin": {"base": "#f4cfbc", "confidence": "high"}},
        "Bob B": {"hair": {"base": "#212225", "confidence": "high"},
                  "jacket": {"base": "#353737", "confidence": "medium"}},
        "Carol C": {"hair": {"base": "#e48c9c", "confidence": "high"}, "note": "not a region"},
    }
    (pal / "palette.json").write_text(json.dumps(palette))
    (tmp_path / "preset.toml").write_text(PRESET)
    return tmp_path


@pytest.fixture
def cfg(data_root):
    return load_config(data_root, str(data_root / "preset.toml"))


@pytest.fixture
def ingested(cfg):
    from dandadye import ingest
    ingest.run(cfg, workers=1, log=lambda *a: None)
    st = State(cfg.state_path)
    yield cfg, st
    st.close()
