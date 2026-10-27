import json
import os
import time

import pytest
from PIL import Image

from dandadye import export, qa
from dandadye.cli import main


def manifest_of(cfg):
    return json.loads((cfg.work / "export" / "manifest.json").read_text())


def test_manifest_exact_format(ingested):
    cfg, st = ingested
    export.run(cfg, st, ["10", "11"], height=440, log=lambda *a: None)
    m = manifest_of(cfg)
    assert list(m) == ["version", "series", "title", "chapters"]
    assert (m["version"], m["series"], m["title"]) == (1, "testseries", "Test Series")
    ch = m["chapters"][0]
    assert list(ch) == ["id", "number", "title", "pages"]
    assert (ch["id"], ch["number"], ch["title"]) == ("10", 10, "Chapter 10")
    assert ch["pages"][0] == {"key": "testseries/10/001.webp", "w": 300, "h": 440, "spread": False}
    assert ch["pages"][1]["spread"] is True and ch["pages"][1]["w"] == 600
    for p in ch["pages"]:
        with Image.open(cfg.work / "export" / p["key"]) as im:
            assert im.format == "WEBP" and im.size == (p["w"], p["h"])
    assert export.validate_manifest(m) == []


def test_validator_matches_reader_rules():
    good = {"version": 1, "series": "s", "title": "T", "chapters": [
        {"id": "245.1", "number": 245.1, "title": "Chapter 245.1",
         "pages": [{"key": "s/245.1/001.v2.webp", "w": 10, "h": 10, "spread": False}]}]}
    assert export.validate_manifest(good) == []
    bad = json.loads(json.dumps(good))
    bad["chapters"][0]["pages"][0]["spred"] = True
    bad["chapters"][0]["pages"].append(dict(good["chapters"][0]["pages"][0]))
    bad["chapters"][0]["id"] = "ch1"
    errs = export.validate_manifest(bad)
    assert any('unknown field "spred"' in e for e in errs)
    assert any("duplicate" in e for e in errs)
    assert any(".id:" in e for e in errs)
    assert export.validate_manifest({**good, "chapters": [{**good["chapters"][0], "pages": []}]})
    assert export.validate_manifest({**good, "extra": 1})


def test_never_upscales_and_fractional_chapter_numbers(ingested):
    cfg, st = ingested
    export.run(cfg, st, ["10"], height=5000, log=lambda *a: None)
    assert manifest_of(cfg)["chapters"][0]["pages"][0]["h"] == 880
    assert export.chapter_entry("245.1", [])["number"] == 245.1


def test_published_keys_are_never_reused(ingested):
    cfg, st = ingested
    export.run(cfg, st, ["10"], height=440, log=lambda *a: None)
    ledger = export.Ledger(cfg.work / "export" / "published.json")
    ledger.mark("testseries/10/001.webp")
    # page 1 gets a colourised version -> new key; page 2 changes but was never published -> same key
    for idx in (1, 2):
        d = cfg.work / "inkfix" / "flux2-lora" / "10"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{idx:03d}.png"
        Image.new("RGB", (600, 880), (200, 100, 100)).save(p)
        st.update(f"10/{idx:03d}", "inkfix", {"path": str(p)})
    export.run(cfg, st, ["10"], height=440, log=lambda *a: None)
    keys = [p["key"] for p in manifest_of(cfg)["chapters"][0]["pages"]]
    assert keys[:2] == ["testseries/10/001.v2.webp", "testseries/10/002.webp"]
    assert (cfg.work / "export" / "testseries/10/001.webp").exists()  # old object untouched
    # unchanged on re-export
    export.run(cfg, st, ["10"], height=440, log=lambda *a: None)
    assert [p["key"] for p in manifest_of(cfg)["chapters"][0]["pages"]] == keys


def test_cli_from_filter(ingested):
    cfg, _ = ingested
    main(["export", "--data", str(cfg.data_root), "--preset", str(cfg.data_root / "preset.toml"),
          "--from", "10", "--height", "440"])
    assert [c["id"] for c in manifest_of(cfg)["chapters"]] == ["10", "11"]


def test_upload_dry_run_then_real_with_fake_wrangler(ingested, tmp_path):
    cfg, st = ingested
    export.run(cfg, st, ["10"], height=440, log=lambda *a: None)
    logs = []
    cmds = export.upload(cfg, ["10"], yes=False, log=logs.append)
    assert "nothing was uploaded" in logs[-1]
    assert cmds[0][:6] == ["npx", "wrangler", "r2", "object", "put", "dandadan-manga/testseries/10/001.webp"]
    assert cmds[0][-1] == "--remote" and "image/webp" in cmds[0]
    assert cmds[-1][5] == "dandadan-manga/manifest.json"          # manifest last
    assert cmds[-1][cmds[-1].index("--cache-control") + 1] == "no-cache"
    with pytest.raises(SystemExit):
        export.upload(cfg, ["10"], yes=True, reader_dir=None, log=lambda *a: None)

    calls = []
    export.upload(cfg, ["10"], yes=True, reader_dir=str(tmp_path), log=lambda *a: None,
                  runner=lambda c, **kw: calls.append((c, kw)))
    assert len(calls) == 5 and calls[-1][0][5].endswith("manifest.json")
    assert calls[0][1]["env"]["WRANGLER_SEND_METRICS"] == "false" and calls[0][1]["cwd"] == tmp_path
    # second upload: pages are published, only the manifest goes up
    calls.clear()
    export.upload(cfg, ["10"], yes=True, reader_dir=str(tmp_path), log=lambda *a: None,
                  runner=lambda c, **kw: calls.append(c))
    assert len(calls) == 1
    # a manifest that drops a published chapter is refused
    export.run(cfg, st, ["11"], height=440, log=lambda *a: None)
    with pytest.raises(SystemExit, match="drops published chapters"):
        export.upload(cfg, ["11"], log=lambda *a: None)


def test_source_choice(ingested):
    cfg, st = ingested
    assert export.pick_source(st.get("1/001"), True)[1] == "original"     # official colour page
    assert export.pick_source(st.get("10/001"), True)[1] == "bw-fallback"
    assert export.pick_source(st.get("10/001"), False) == (None, "missing")


def test_qa_heuristic_flags_grey_pages(ingested):
    cfg, st = ingested
    grey = Image.open(st.get("10/001")["ingest"]["src"]).convert("RGB")
    assert qa.HeuristicChecker().check(grey, {}, {}, cfg).status == "fail"
    tinted = Image.new("RGB", grey.size, (220, 120, 140))
    assert qa.HeuristicChecker().check(tinted, {}, {}, cfg).status == "ok"
