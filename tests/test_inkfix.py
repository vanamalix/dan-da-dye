import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from dandadye import inkfix
from dandadye.util import rgb_to_lab
from tests.conftest import bw_page


def fake_model_output(page: Image.Image) -> Image.Image:
    """Lower-res 'diffusion' output: page tinted, blurred, slightly offset lines, tinted balloon."""
    small = page.convert("L").resize((page.width // 2, page.height // 2), Image.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur(1.5))
    g = np.asarray(small, float) / 255
    tint = np.array([0.95, 0.75, 0.6])  # warm skin-ish everywhere
    rgb = (0.35 + 0.65 * g[..., None]) * tint * 255
    out = Image.fromarray(rgb.astype(np.uint8))
    ImageDraw.Draw(out).rectangle([20, 20, 120, 120], fill=(60, 120, 200))  # a blue object
    return out


def chroma(rgb):
    lab = rgb_to_lab(np.asarray(rgb))
    return np.hypot(lab[..., 1], lab[..., 2])


def test_recombine_keeps_ink_black_and_paper_white():
    page = bw_page()
    out = inkfix.recombine(page, fake_model_output(page), upscale_to_colour=False)
    assert out.size == page.size
    g = np.asarray(page.convert("L"))
    o = np.asarray(out).astype(int)
    c = chroma(out)
    ink = g < 30
    assert o[ink].max() <= 40 and c[ink].max() < 3            # ink stays neutral black
    toned = (g > 120) & (g < 180)
    assert c[toned].mean() > 10                                 # grey areas picked up colour
    assert np.abs(o[toned] - g[toned][:, None]).mean() > 5     # ... and are not just the original


def test_text_boxes_restored_from_original():
    page = bw_page()
    w = page.width
    balloon = [w - 260, 60, w - 60, 180]
    out = inkfix.recombine(page, fake_model_output(page), text_boxes=[balloon], upscale_to_colour=False)
    x1, y1, x2, y2 = balloon
    crop = np.asarray(out)[y1 + 15:y2 - 15, x1 + 30:x2 - 30].astype(int)
    orig = np.asarray(page.convert("L"))[y1 + 15:y2 - 15, x1 + 30:x2 - 30].astype(int)
    assert np.abs(crop - orig[..., None]).max() <= 1


def test_gutters_outside_panels_become_white():
    page = bw_page()
    panels = [[20, 20, page.width - 20, page.height // 2 - 10],
              [20, page.height // 2 + 10, page.width - 20, page.height - 20]]
    col = Image.new("RGB", (page.width // 2, page.height // 2), (200, 150, 100))
    out = np.asarray(inkfix.recombine(page, col, panel_boxes=panels, upscale_to_colour=False))
    assert (out[5:15, 5:-5] == 255).all()  # top margin
    assert (out[page.height // 2 - 3:page.height // 2 + 3, 40:-40] == 255).all()  # gutter


def test_upscales_to_colour_resolution():
    page = bw_page(300, 440)
    col = Image.new("RGB", (600, 880), (200, 150, 100))
    assert inkfix.recombine(page, col).size == (600, 880)


def test_inkfix_run(ingested):
    cfg, st = ingested
    d = cfg.work / "colour" / "flux2-lora" / "10"
    d.mkdir(parents=True)
    with Image.open(st.get("10/001")["ingest"]["src"]) as im:
        fake_model_output(im).save(d / "001.png")
    inkfix.run(cfg, st, ["10"], log=lambda *a: None)
    rec = st.get("10/001")["inkfix"]
    assert rec["w"] == 600 and Image.open(rec["path"]).mode == "RGB"
    assert "inkfix" not in st.get("10/002")


def test_whole_balloon_cleaned_from_text_box():
    page = bw_page()
    w = page.width
    text_box = [w - 222, 90, w - 98, 148]          # just the lettering, inside the ellipse balloon
    out = np.asarray(inkfix.recombine(page, fake_model_output(page), text_boxes=[text_box],
                                      upscale_to_colour=False))
    assert (out[120, w - 245] == 255).all()        # balloon interior outside the text box: white
    assert (out[120, w - 75] == 255).all()
    assert chroma(Image.fromarray(out[200:240, w - 240:w - 80])).mean() > 5  # outside the balloon: still coloured
