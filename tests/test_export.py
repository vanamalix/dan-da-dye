import json

from PIL import Image

from dandadye import export, qa
from dandadye.cli import main


def test_manifest_exact_format(ingested):
    cfg, st = ingested
    export.run(cfg, st, ["10", "11"], height=440, log=lambda *a: None)
    m = json.loads((cfg.work / "export" / "testseries" / "manifest.json").read_text())
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


def test_never_upscales_and_fractional_chapter_numbers(ingested):
    cfg, st = ingested
    export.run(cfg, st, ["10"], height=5000, log=lambda *a: None)
    m = json.loads((cfg.work / "export" / "testseries" / "manifest.json").read_text())
    assert m["chapters"][0]["pages"][0]["h"] == 880
    assert export.chapter_entry("s", "245.1", [])["number"] == 245.1


def test_cli_from_filter_and_upload_stub(ingested, capsys):
    cfg, _ = ingested
    main(["export", "--data", str(cfg.data_root), "--preset", str(cfg.data_root / "preset.toml"),
          "--from", "10", "--height", "440"])
    m = json.loads((cfg.work / "export" / "testseries" / "manifest.json").read_text())
    assert [c["id"] for c in m["chapters"]] == ["10", "11"]
    main(["upload", "--data", str(cfg.data_root), "--preset", str(cfg.data_root / "preset.toml"), "--from", "10"])
    out = capsys.readouterr().out
    assert "would PUT testseries/10/001.webp" in out and "Nothing was uploaded" in out


def test_source_choice(ingested):
    cfg, st = ingested
    assert export.pick_source(st.get("1/001"), True)[1] == "original"     # official colour page
    assert export.pick_source(st.get("10/001"), True)[1] == "bw-fallback"
    assert export.pick_source(st.get("10/001"), False) == (None, "missing")


def test_qa_heuristic_flags_grey_pages(ingested):
    cfg, st = ingested
    grey = Image.open(st.get("10/001")["ingest"]["src"]).convert("RGB")
    res = qa.HeuristicChecker().check(grey, {}, {}, cfg)
    assert res.status == "fail"
    tinted = Image.new("RGB", grey.size, (220, 120, 140))
    assert qa.HeuristicChecker().check(tinted, {}, {}, cfg).status == "ok"
