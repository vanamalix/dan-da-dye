import json

import numpy as np
from PIL import Image

from dandadye import detect
from dandadye.detect.bank import build_bank
from dandadye.detect.cluster import cluster_embeddings, cluster_votes
from dandadye.detect.magi_v2 import MagiV2Detector


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_cluster_embeddings_groups_similar_vectors():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=32), rng.normal(size=32)
    embs = [_unit(a + 0.1 * rng.normal(size=32)) for _ in range(4)] + \
           [_unit(b + 0.1 * rng.normal(size=32)) for _ in range(3)]
    labels = cluster_embeddings(embs, threshold=0.6)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1
    assert labels[0] != labels[4]
    assert cluster_embeddings([]) == []


def test_cluster_votes_needs_clear_majority():
    votes = cluster_votes([0, 0, 0, 1, 1, 2, 2, 2], ["Alice", "Alice", "Other", "Bob", "Alice",
                                                     "Bob", "Carol", "Other"])
    assert votes == {0: "Alice"}


def test_bank_uses_preset_art(cfg):
    bank = build_bank(cfg)
    names = [n for n, _ in bank]
    assert names == ["Alice", "Alice", "Bob", "Bob", "Carol", "Carol"]
    with Image.open(bank[0][1]) as im:
        assert im.mode == "L"


def test_magi_job_and_normalise(cfg, tmp_path):
    det = MagiV2Detector.from_config(cfg)
    job = det.job([tmp_path / "a.png"], [("Alice", tmp_path / "alice.png")])
    assert job["bank"] == {"images": [str(tmp_path / "alice.png")], "names": ["Alice"]}
    page = MagiV2Detector.normalise({"texts": [[0, 0, 1, 1], [2, 2, 3, 3]], "is_essential_text": [True, False],
                                     "characters": []})
    assert page["texts"] == [[0, 0, 1, 1]] and page["texts_other"] == [[2, 2, 3, 3]]


class FakeDetector:
    """Pretends page 1 has Alice twice-ish and page 2 has an unnamed lookalike."""
    name = "fake"

    @classmethod
    def from_config(cls, cfg, **kw):
        return cls()

    def detect_chapter(self, pages, bank):
        e = _unit([1, 0, 0]).tolist()
        e2 = _unit([0.95, 0.05, 0]).tolist()
        out = []
        for i, _ in enumerate(pages):
            out.append({"panels": [[0, 0, 100, 100]], "texts": [[10, 10, 50, 30]], "texts_other": [],
                        "characters": [[5, 5, 40, 90]],
                        "character_names": ["Alice" if i < 2 else "Other"],
                        "embeddings": [e if i % 2 == 0 else e2]})
        return out


def test_detect_run_with_pluggable_detector(ingested, monkeypatch):
    cfg, st = ingested
    monkeypatch.setitem(detect.DETECTORS, "fake", FakeDetector)
    detect.run(cfg, st, ["10"], detector="fake", viz=True, log=lambda *a: None)
    pages = [r[3]["detect"] for r in st.pages(["10"])]
    assert all(p["name_counts"] == {"Alice": 1} for p in pages)  # page 3/4 filled by cluster vote
    assert pages[2]["characters"][0]["magi_name"] == "Other"
    assert (cfg.work / "detect_viz" / "10" / "001.jpg").exists()
    # a human override for the cluster wins
    cluster = pages[0]["characters"][0]["cluster"]
    (cfg.work / "detect" / "10" / "names.json").write_text(json.dumps({str(cluster): "Bob"}))
    detect.run(cfg, st, ["10"], detector="fake", log=lambda *a: None)
    assert st.get("10/001")["detect"]["name_counts"] == {"Bob": 1}
