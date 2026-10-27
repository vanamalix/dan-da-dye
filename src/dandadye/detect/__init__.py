"""Stage 2 - detect: panels, text and character boxes, and who each character is.

Detectors are pluggable (``DETECTORS``). The default, Magi v2, runs in its own
venv via a subprocess and names characters against a *character bank* built
from the preset's official art. On top of Magi's per-page clusters we cluster
characters chapter-wide on Magi's crop embeddings, so a human can name a whole
cluster at once in ``work/detect/<chapter>/names.json`` ({"<cluster>": "Momo"}).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from ..config import Config
from ..state import State
from .bank import build_bank
from .base import Detector
from .cluster import cluster_embeddings, cluster_votes
from .magi_v2 import MagiV2Detector

DETECTORS: dict[str, type] = {"magi-v2": MagiV2Detector}


def make_detector(name: str, cfg: Config, device: str = "auto", eta: float = 0.75) -> Detector:
    return DETECTORS[name].from_config(cfg, device=device, eta=eta)


def assign_names(pages: list[dict], overrides: dict[str, str], threshold: float = 0.55) -> None:
    """Add chapter-wide 'cluster' and final 'name' to every character, in place."""
    flat = [(pi, ci) for pi, p in enumerate(pages) for ci in range(len(p["characters"]))]
    embs = [pages[pi]["embeddings"][ci] for pi, ci in flat] if all(p.get("embeddings") for p in pages
                                                                   if p["characters"]) else []
    labels = cluster_embeddings(embs, threshold) if embs else list(range(len(flat)))
    magi_names = [(pages[pi].get("character_names") or ["Other"] * len(pages[pi]["characters"]))[ci]
                  for pi, ci in flat]
    votes = cluster_votes(labels, magi_names)
    for (pi, ci), lab, mname in zip(flat, labels, magi_names):
        p = pages[pi]
        p.setdefault("cluster", [None] * len(p["characters"]))
        p.setdefault("names", [None] * len(p["characters"]))
        p["cluster"][ci] = int(lab)
        name = overrides.get(str(lab)) or mname
        if name == "Other" and votes.get(lab):
            name = votes[lab]
        p["names"][ci] = name


def draw_viz(src: Path, page: dict, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    for b in page.get("panels", []):
        d.rectangle(b, outline=(0, 160, 0), width=3)
    for b in page.get("texts", []):
        d.rectangle(b, outline=(0, 90, 255), width=2)
    for b, name, cl in zip(page["characters"], page.get("names", []), page.get("cluster", [])):
        d.rectangle(b, outline=(230, 0, 0), width=3)
        d.text((b[0] + 4, b[1] + 2), f"{name} #{cl}", fill=(230, 0, 0))
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, quality=85)


def run(cfg: Config, state: State, chapters: list[str], pages: set[int] | None = None,
        detector: str = "magi-v2", device: str = "auto", force: bool = False, viz: bool = False,
        eta: float = 0.75, log=print) -> None:
    det = None
    bank = None
    for cid in chapters:
        rows = [r for r in state.pages([cid]) if not pages or r[2] in pages]
        rows = [r for r in rows if not r[3]["ingest"].get("error")]
        if not rows:
            continue
        out_dir = cfg.stage_dir(f"detect/{cid}")
        raw_path = out_dir / "raw.json"
        wanted = [r[3]["ingest"]["src"] for r in rows]
        raw = json.loads(raw_path.read_text()) if raw_path.exists() else None
        if force or raw is None or raw.get("srcs") != wanted or raw.get("detector") != detector:
            det = det or make_detector(detector, cfg, device, eta)
            bank = bank if bank is not None else build_bank(cfg)
            log(f"detect: {cid}: {len(rows)} pages with {detector} (bank: {len(bank)} images)")
            results = det.detect_chapter([Path(s) for s in wanted], bank)
            raw = {"detector": detector, "srcs": wanted, "pages": results}
            raw_path.write_text(json.dumps(raw))
        names_path = out_dir / "names.json"
        overrides = json.loads(names_path.read_text()) if names_path.exists() else {}
        assign_names(raw["pages"], overrides)
        for (key, _, idx, data), page in zip(rows, raw["pages"]):
            counts: dict[str, int] = {}
            for n in page["names"]:
                counts[n] = counts.get(n, 0) + 1
            state.update(key, "detect", {
                "detector": detector,
                "panels": page["panels"],
                "texts": page["texts"],
                "texts_other": page.get("texts_other", []),
                "characters": [{"box": b, "name": n, "cluster": c, "magi_name": m}
                               for b, n, c, m in zip(page["characters"], page["names"], page["cluster"],
                                                     page.get("character_names") or [None] * len(page["names"]))],
                "name_counts": counts,
            })
            if viz:
                draw_viz(Path(data["ingest"]["src"]), page, cfg.work / "detect_viz" / cid / f"{idx:03d}.jpg")
        n_chars = sum(len(p["characters"]) for p in raw["pages"])
        n_named = sum(1 for p in raw["pages"] for n in p["names"] if n != "Other")
        log(f"detect: {cid}: {n_chars} character boxes, {n_named} named, "
            f"{len(set(c for p in raw['pages'] for c in p['cluster']))} chapter clusters")
