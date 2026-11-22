"""The diffusers runtime: selection, --dry-run without torch, and a CPU smoke test on a tiny random pipeline."""

from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from dandadye import colourise, refsheet
from dandadye.colourise import Settings, pick_runtime

TINY_REPO = "optimum-intel-internal-testing--tiny-random-flux.2-klein"


# --- runtime selection ------------------------------------------------------------

@pytest.mark.parametrize("system,machine,expected", [
    ("Darwin", "arm64", "mlx"), ("Darwin", "x86_64", "diffusers"),
    ("Linux", "x86_64", "diffusers"), ("Linux", "aarch64", "diffusers"), ("Windows", "AMD64", "diffusers")])
def test_auto_runtime_per_platform(system, machine, expected):
    assert pick_runtime("auto", system, machine) == expected
    assert pick_runtime(None, system, machine) == expected


def test_explicit_runtime_wins_over_platform():
    assert pick_runtime("diffusers", "Darwin", "arm64") == "diffusers"
    assert pick_runtime("mlx", "Linux", "x86_64") == "mlx"
    with pytest.raises(ValueError, match="unknown runtime"):
        pick_runtime("tpu")


def test_settings_runtime(cfg, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert Settings.from_config(cfg).runtime == "mlx"                      # API default is unchanged
    assert Settings.from_config(cfg, SimpleNamespace(runtime="auto")).runtime == "diffusers"
    assert Settings.from_config(cfg, SimpleNamespace(runtime="mlx")).runtime == "mlx"
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    assert Settings.from_config(cfg, SimpleNamespace(runtime="auto")).runtime == "mlx"
    assert Settings.from_config(cfg, SimpleNamespace(runtime="diffusers")).runtime == "diffusers"
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers", device="cpu", dtype="fp32"))
    assert (s.device, s.dtype) == ("cpu", "fp32")


def test_cli_parser_accepts_runtime_flags():
    from dandadye.cli import build_parser
    a = build_parser().parse_args(["colourise", "--runtime", "diffusers", "--device", "rocm", "--dtype", "bf16"])
    assert (a.runtime, a.device, a.dtype) == ("diffusers", "rocm", "bf16")
    assert build_parser().parse_args(["colourise"]).runtime == "auto"


# --- helpers that need no torch ------------------------------------------------------

def test_repo_resolution(cfg):
    from dandadye import colourise_diffusers as cd
    s = Settings.from_config(cfg)
    assert cd.resolve_repo(s) == "black-forest-labs/FLUX.2-klein-base-9B"
    s.model_path = "/models/x"
    assert cd.resolve_repo(s) == "/models/x"
    s.model_path, s.model = None, "someone/their-klein"
    assert cd.resolve_repo(s) == "someone/their-klein"
    s.model = "nonsense"
    with pytest.raises(SystemExit, match="not a known model"):
        cd.resolve_repo(s)


def test_cli_backend_refused_under_diffusers(cfg):
    from dandadye import colourise_diffusers as cd
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers"))
    cd.check_supported(s, "api")
    with pytest.raises(SystemExit, match="mflux"):
        cd.check_supported(s, "cli")


def test_quantize_is_ignored_with_a_note(cfg):
    from dandadye import colourise_diffusers as cd
    from dandadye.device import choose
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers"))  # quantize defaults to 4
    summary = cd.load_summary(s, choose(flags=dict(torch=None)))
    assert summary["quantize"].startswith("ignored")
    assert any("-q 4" in n and "ignored" in n for n in summary["notes"])
    assert summary["cpu_offload"] is False and summary["vae_tiling"] is True


def test_call_kwargs_image_order_and_guidance(cfg):
    from dandadye import colourise_diffusers as cd
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers"))
    kw = cd.call_kwargs(s, Path("/p/page.png"), [Path("/p/ref.png")], 832, 1216, "mngclranm")
    assert kw["image"] == ["/p/page.png", "/p/ref.png"] and kw["guidance_scale"] == 4.0
    s3 = Settings.from_config(cfg, engine="flux3")
    assert "guidance_scale" not in cd.call_kwargs(s3, Path("a"), [Path("b")], 16, 16, "x")


# --- --dry-run with torch and diffusers unavailable -----------------------------------

def test_dry_run_without_torch_or_diffusers(ingested, monkeypatch):
    cfg, st = ingested
    refsheet.run(cfg, st, ["10"], log=lambda *a: None)
    for mod in [m for m in sys.modules if m.split(".")[0] in ("torch", "diffusers")]:
        monkeypatch.delitem(sys.modules, mod)
    real_import = builtins.__import__

    def no_ml(name, *a, **k):
        if name.split(".")[0] in ("torch", "diffusers"):
            raise ImportError(f"blocked {name}")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_ml)
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers"))
    logs = []
    plans = colourise.run(cfg, st, ["10"], {1}, s, dry_run=True, log=logs.append)
    assert len(plans) == 1
    call = plans[0]["call"]
    assert call["image"][0].endswith("colourise_in/10/001.png") and call["image"][1].endswith(".png")
    assert call["prompt"] == "mngclranm" and call["num_inference_steps"] == 50 and call["guidance_scale"] == 4.0
    assert plans[0]["negative_prompt"] == "" and plans[0]["seed"] == 42
    text = "\n".join(logs)
    assert "diffusers runtime, dry run" in text and "cpu (torch not installed)" in text
    assert "-q 4" in text and "diffusers argument check: skipped" in text
    assert not (cfg.work / "colourise_in").exists()                 # nothing written
    assert not any(m.split(".")[0] in ("torch", "diffusers") for m in sys.modules)


def test_dry_run_refuses_cli_backend(ingested):
    cfg, st = ingested
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers"))
    with pytest.raises(SystemExit, match="mflux"):
        colourise.run(cfg, st, ["10"], None, s, dry_run=True, backend="cli", log=lambda *a: None)


def test_mlx_dry_run_still_prints_commands(ingested):
    cfg, st = ingested
    refsheet.run(cfg, st, ["10"], log=lambda *a: None)
    s = Settings.from_config(cfg, SimpleNamespace(runtime="mlx"))
    out = colourise.run(cfg, st, ["10"], {1}, s, dry_run=True, log=lambda *a: None)
    assert isinstance(out[0], list) and out[0][0] == s.mflux_bin


# --- CPU smoke test: the real flux2-lora code path on a tiny random pipeline ----------

def _tiny_snapshot() -> Path | None:
    homes = [os.environ.get("HF_HOME"), "/config/dandadye-data/hf-gpucheck"]
    for home in filter(None, homes):
        snaps = Path(home) / "hub" / f"models--{TINY_REPO}" / "snapshots"
        found = sorted(p for p in snaps.glob("*") if (p / "model_index.json").exists()) if snaps.is_dir() else []
        if found:
            return found[-1]
    return None


def _tiny_lora(pipe, path: Path) -> None:
    """A random rank-2 PEFT-style LoRA on the first few attention projections of the transformer."""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    sd = {}
    for name, mod in pipe.transformer.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(("to_q", "to_k", "to_v")) and len(sd) < 12:
            sd[f"transformer.{name}.lora_A.weight"] = torch.randn(2, mod.in_features, generator=g) * 0.02
            sd[f"transformer.{name}.lora_B.weight"] = torch.randn(mod.out_features, 2, generator=g) * 0.02
    assert sd, "no attention projections found in the tiny transformer"
    save_file(sd, str(path))


@pytest.fixture
def tiny(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("safetensors")
    snap = _tiny_snapshot()
    if snap is None:
        pytest.skip("tiny random FLUX.2 Klein pipeline is not in the offline cache")
    return snap


def _inputs(tmp_path):
    from tests.conftest import bw_page
    page, ref = tmp_path / "page.png", tmp_path / "ref.png"
    bw_page(160, 224).convert("RGB").save(page)
    Image.new("RGB", (64, 64), (200, 80, 90)).save(ref)
    return page, ref


@pytest.mark.parametrize("bake", [True, False])
@pytest.mark.parametrize("guidance", [4.0, 1.0])
def test_tiny_pipeline_cpu_lora_run(tiny, tmp_path, monkeypatch, cfg, bake, guidance):
    import torch
    from dandadye import colourise_diffusers as cd
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers", device="cpu", dtype="fp32", steps=2,
                                                  guidance=guidance, seed=7, no_bake_lora=not bake))
    s.model_path = str(tiny)
    s.lora_path = str(tmp_path / "tiny_lora.safetensors")
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(str(tiny), dtype=torch.float32)
    _tiny_lora(pipe, Path(s.lora_path))
    # the tiny checkpoint is distilled; make it behave like the undistilled base so real CFG runs
    pipe.register_to_config(is_distilled=False)
    page, ref = _inputs(tmp_path)
    logs = []
    eng = cd.DiffusersEngine(s, logs.append, pipe=pipe)
    assert eng.choice.device == "cpu" and eng.cfg is (guidance > 1)
    assert any("-q 4" in line and "ignored" in line for line in logs)
    out = tmp_path / "out" / "001.png"
    eng.generate(page, [ref], out, 64, 96, s.prompt)
    with Image.open(out) as im:
        assert im.size == (64, 96) and im.mode == "RGB"
    assert eng.pipe.text_encoder is None                            # low-ram freed it after encoding
    assert bool(eng.pipe.get_active_adapters()) is (not bake)       # fused: no adapter left; else set_adapters
    first = Image.open(out).tobytes()
    out2 = tmp_path / "out" / "002.png"
    eng.generate(page, [ref], out2, 64, 96, s.prompt)               # second page reuses the cached embeds
    assert Image.open(out2).tobytes() == first                      # same seed, same output


def test_distilled_checkpoint_ignores_guidance(tiny, tmp_path, cfg):
    import torch
    from diffusers import Flux2KleinPipeline
    from dandadye import colourise_diffusers as cd
    s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers", device="cpu", dtype="fp32", steps=2))
    s.model_path = str(tiny)
    s.lora_path = str(tmp_path / "none.safetensors")
    s.quantize = None
    pipe = Flux2KleinPipeline.from_pretrained(str(tiny), dtype=torch.float32)
    assert pipe.config.is_distilled
    logs = []
    import dandadye.colourise as c
    monkey = pytest.MonkeyPatch()
    monkey.setitem(c.ENGINES["flux2-lora"], "uses_lora", False)
    try:
        eng = cd.DiffusersEngine(s, logs.append, pipe=pipe)
    finally:
        monkey.undo()
    assert eng.cfg is False and any("distilled" in line and "ignored" in line for line in logs)


def test_tiny_pipeline_without_lora_differs_from_with(tiny, tmp_path, monkeypatch, cfg):
    """A real LoRA fold must change the output (guards against a silently dropped adapter)."""
    import torch
    from diffusers import Flux2KleinPipeline
    from dandadye import colourise_diffusers as cd
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    lora = tmp_path / "l.safetensors"
    _tiny_lora(Flux2KleinPipeline.from_pretrained(str(tiny), dtype=torch.float32), lora)
    page, ref = _inputs(tmp_path)
    outs = []
    for scale in (0.0, 5.0):
        s = Settings.from_config(cfg, SimpleNamespace(runtime="diffusers", device="cpu", dtype="fp32", steps=2,
                                                      guidance=1.0, lora_scale=scale))
        s.model_path, s.lora_path = str(tiny), str(lora)
        pipe = Flux2KleinPipeline.from_pretrained(str(tiny), dtype=torch.float32)
        eng = cd.DiffusersEngine(s, lambda *_: None, pipe=pipe)
        out = tmp_path / f"o{scale}.png"
        eng.generate(page, [ref], out, 64, 96, s.prompt)
        outs.append(Image.open(out).tobytes())
    assert outs[0] != outs[1]


def test_chunked_sdpa_matches_full_attention(monkeypatch):
    torch = pytest.importorskip("torch")
    from dandadye import colourise_diffusers as cd

    monkeypatch.setattr(cd, "CHUNK_BYTES", 4 * 2 * 40 * 4 * 5)  # forces several chunks
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 2, 40, 8, generator=g) for _ in range(3))
    mask = torch.rand(1, 1, 40, 40, generator=g) > 0.2
    mask[..., 0] = True
    real = torch.nn.functional.scaled_dot_product_attention
    for m in (None, mask):
        want = real(q, k, v, attn_mask=m)
        got = cd.chunked_sdpa(torch, real, q, k, v, attn_mask=m)
        assert torch.allclose(want, got, atol=1e-5)
