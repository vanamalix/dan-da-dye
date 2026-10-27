import numpy as np
from PIL import Image

from dandadye import refsheet
from dandadye.refsheet import cast_strip, characters_for_page, load_palette, snap16, swatches_for


def test_snap16():
    w, h = snap16(784, 1145, 1.0)
    assert w % 16 == 0 and h % 16 == 0
    assert 0.9 < w * h / 2**20 <= 1.0
    assert abs(w / h - 784 / 1145) < 0.02


def test_swatches_prefer_high_confidence(cfg):
    pal = load_palette(cfg.resolve(cfg.preset.palette))
    assert swatches_for(pal, "Bob B") == ["#212225", "#353737"]
    assert swatches_for(pal, "Carol C") == ["#e48c9c"]  # non-region entries ignored
    assert swatches_for(pal, "nobody") == []


def test_cast_strip_contains_art_and_swatches(cfg):
    pal = load_palette(cfg.resolve(cfg.preset.palette))
    alice, bob = cfg.preset.characters[:2]
    sheet = cast_strip([alice, bob], pal, megapixels=0.5)
    assert sheet.width % 16 == 0 and sheet.height % 16 == 0
    px = np.asarray(sheet).reshape(-1, 3).astype(int)
    def has(rgb, tol=12):
        return (np.abs(px - rgb).max(1) < tol).any()
    assert has((200, 80, 90))      # Alice's hair from the art
    assert has((0xc8, 0x50, 0x5a))  # Alice's palette swatch
    assert has((0x21, 0x22, 0x25))  # Bob's swatch


def test_characters_for_page_and_fallback(cfg):
    chars, fb = characters_for_page(cfg, {"detect": {"name_counts": {"Carol": 1, "Other": 5, "Bob": 3}}}, 4)
    assert [c.name for c in chars] == ["Bob", "Carol"] and not fb
    chars, fb = characters_for_page(cfg, {}, 4)
    assert [c.name for c in chars] == ["Alice", "Bob"] and fb


def test_refsheet_run_shares_sheets(ingested):
    cfg, st = ingested
    refsheet.run(cfg, st, ["10", "11"], log=lambda *a: None)
    paths = {r[3]["refsheet"]["path"] for r in st.pages(["10", "11"])}
    assert len(paths) == 1  # no detections -> everyone gets the main-cast sheet
    with Image.open(paths.pop()) as im:
        assert im.width % 16 == 0


def test_builders_are_pluggable(ingested, monkeypatch):
    cfg, st = ingested
    monkeypatch.setitem(refsheet.BUILDERS, "plain", lambda chars, pal, mp: Image.new("RGB", (64, 64), "red"))
    refsheet.run(cfg, st, ["10"], builder="plain", log=lambda *a: None)
    with Image.open(st.get("10/001")["refsheet"]["path"]) as im:
        assert im.size == (64, 64)
