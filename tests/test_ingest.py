from pathlib import Path

from dandadye import ingest
from dandadye.state import State
from dandadye.util import lab_to_rgb, natural_key, rgb_to_lab, select_chapters

import numpy as np


def test_natural_sort_and_chapter_selection():
    assert sorted(["p10", "p2", "p1"], key=natural_key) == ["p1", "p2", "p10"]
    ids = ["150", "9", "245.1", "245", "extras"]
    assert select_chapters(ids) == ["9", "150", "245", "245.1", "extras"]
    assert select_chapters(ids, start=150) == ["150", "245", "245.1"]
    assert select_chapters(ids, start=150, end=245) == ["150", "245"]
    assert select_chapters(ids, only=["9", "245.1"]) == ["9", "245.1"]


def test_lab_roundtrip():
    rgb = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    back = lab_to_rgb(rgb_to_lab(rgb))
    assert np.abs(back.astype(int) - rgb).max() <= 1


def test_discover_orders_and_splits_volumes(cfg):
    found = ingest.discover(cfg)
    assert list(found) == ["1", "2", "10", "11"]
    assert [p.numbers for p in found["1"]] == [(0,), (1,), (2,), (10,)]
    assert found["2"][1].numbers == (12, 13)
    assert [p.numbers for p in found["10"]] == [(1,), (2, 3), (4,), (6,)]
    assert [p.numbers for p in found["11"]] == [(1,), (3,), (4,), (5,)]


def test_ingest_flags_and_gaps(ingested):
    cfg, st = ingested
    ch10, ch11, ch1 = st.chapter("10"), st.chapter("11"), st.chapter("1")
    assert ch10["pages"] == 4 and ch10["spreads"] == [2]
    assert ch10["missing_numbers"] == [5]
    assert "missing file" in ch10["gap_notes"][0]
    assert ch11["missing_numbers"] == [2]
    assert "spread" in ch11["gap_notes"][0]
    assert ch11["blank"] == [2, 3] and ch11["sparse"] == [4]
    assert ch1["colour"] == [1] and ch1["missing_numbers"] == list(range(3, 10))
    pairs = sorted(p.name for p in (cfg.work / "training_pairs").iterdir())
    assert pairs == ["1_001_bw.png", "1_001_colour.png"]


def test_ingest_is_idempotent(ingested):
    cfg, st = ingested
    msgs = []
    ingest.run(cfg, workers=1, log=msgs.append)
    assert "0 to analyse" in msgs[0]
    # a stage result survives a re-ingest of unchanged pages
    st.update("10/001", "detect", {"x": 1})
    ingest.run(cfg, workers=1, log=lambda *a: None)
    assert State(cfg.state_path).get("10/001")["detect"] == {"x": 1}


def test_pages_in_reading_order(ingested):
    _, st = ingested
    keys = [k for k, *_ in st.pages()]
    assert keys[:4] == ["1/001", "1/002", "1/003", "1/004"]
    assert Path(st.get("1/004")["ingest"]["src"]).name.startswith("Series - c001 (v01) - p010")
