import argparse
from pathlib import Path

import pytest
from PIL import Image

from dandadye import colourise, refsheet
from dandadye.colourise import Settings, build_command, validate_with_mflux


def test_command_has_exact_image_order_and_lora(cfg):
    s = Settings.from_config(cfg)
    cmd = build_command(s, Path("/p/page.png"), Path("/p/ref.png"), Path("/p/out.png"), 832, 1216)
    i = cmd.index("--image-paths")
    assert cmd[i + 1:i + 3] == ["/p/page.png", "/p/ref.png"]  # image 1 = page, image 2 = reference
    assert cmd[cmd.index("--model") + 1] == "flux2-klein-base-9b"
    assert cmd[cmd.index("--prompt") + 1] == "mngclranm"
    assert cmd[cmd.index("--lora-paths") + 1].endswith("manga_colorization.safetensors")
    assert cmd[cmd.index("--lora-scales") + 1] == "0.9"
    assert cmd[cmd.index("--steps") + 1] == "50" and cmd[cmd.index("--guidance") + 1] == "4"
    assert cmd[cmd.index("-q") + 1] == "4" and "--low-ram" in cmd
    assert (cmd[cmd.index("--width") + 1], cmd[cmd.index("--height") + 1]) == ("832", "1216")


def test_settings_precedence(cfg):
    cfg.preset.colourise = {"steps": 30, "prompt_extra": "soft light"}
    args = argparse.Namespace(model=None, model_path=None, quantize=8, steps=None, guidance=None,
                              lora_scale=1.0, megapixels=None, seed=None, prompt_extra=None,
                              low_ram=False, no_quantize=False)
    s = Settings.from_config(cfg, args)
    assert (s.steps, s.quantize, s.lora_scale, s.low_ram) == (30, 8, 1.0, False)
    assert s.prompt == "mngclranm, soft light"
    assert Path(s.lora_path).is_absolute()
    cmd = build_command(s, Path("a"), Path("b"), Path("c"), 16, 16)
    assert "--low-ram" not in cmd


def test_model_path_declares_base_model(cfg):
    s = Settings.from_config(cfg)
    s.model_path = "/models/klein-base-q4"
    cmd = build_command(s, Path("a"), Path("b"), Path("c"), 16, 16)
    assert cmd[1:5] == ["--model", "/models/klein-base-q4", "--base-model", "flux2-klein-base-9b"]


def test_prepare_page_is_16_aligned_rgb(tmp_path, data_root):
    src = data_root / "work" / "src" / "10" / "001.png"
    w, h = colourise.prepare_page(src, tmp_path / "in.png", 1.0)
    with Image.open(tmp_path / "in.png") as im:
        assert im.mode == "RGB" and im.size == (w, h) and w % 16 == 0 and h % 16 == 0


def test_dry_run_skips_flagged_pages_and_needs_refsheet(ingested):
    cfg, st = ingested
    logs = []
    s = Settings.from_config(cfg)
    assert colourise.run(cfg, st, ["10"], None, s, dry_run=True, log=logs.append) == []
    refsheet.run(cfg, st, ["10", "11", "1"], log=lambda *a: None)
    cmds = colourise.run(cfg, st, ["10", "11", "1"], None, s, dry_run=True, log=logs.append)
    # ch10: 4 pages; ch11: 1 normal page (others blank/sparse); ch1: colour page skipped
    assert len(cmds) == 4 + 1 + 3
    assert all(c[c.index("--image-paths") + 2].endswith(".png") for c in cmds)
    assert not (cfg.work / "colourise_in").exists()  # dry run writes nothing


def test_mflux_accepts_the_command(cfg):
    pytest.importorskip("mflux")
    s = Settings.from_config(cfg)
    cmd = build_command(s, Path("/p/page.png"), Path("/p/ref.png"), Path("/p/out.png"), 832, 1216)
    assert validate_with_mflux(cmd) == ""
