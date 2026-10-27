"""Stage 6 - export: webp pages + manifest.json for the reader site.

Layout under ``<data>/work/export/``::

    <series>/manifest.json
    <series>/<chapter>/<NNN>.webp      (the manifest "key" is this path)

Manifest format (shared with the reader, keep it exact)::

    {"version":1,"series":"dandadan","title":"Dandadan","chapters":[
      {"id":"150","number":150,"title":"Chapter 150",
       "pages":[{"key":"dandadan/150/001.webp","w":1600,"h":2300,"spread":false}]}]}
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from .config import Config
from .state import State
from .util import chapter_number, chapter_sort_key


def pick_source(data: dict, allow_bw: bool) -> tuple[Path | None, str]:
    """Which image to publish for a page, and why."""
    ing = data.get("ingest", {})
    if ing.get("colour") or ing.get("blank") or ing.get("sparse"):
        return Path(ing["src"]), "original"
    fixed = data.get("inkfix", {}).get("path")
    if fixed and Path(fixed).exists():
        return Path(fixed), "colourised"
    if allow_bw:
        return Path(ing["src"]), "bw-fallback"
    return None, "missing"


def resize_for_web(im: Image.Image, height: int) -> Image.Image:
    im = im.convert("RGB")
    if im.height > height:
        im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    return im


def chapter_entry(series: str, cid: str, pages: list[dict]) -> dict:
    n = chapter_number(cid)
    number = (int(n) if n is not None and n.is_integer() else n) if n is not None else cid
    return {"id": cid, "number": number, "title": f"Chapter {cid}", "pages": pages}


def write_manifest(path: Path, series: str, title: str, chapters: list[dict]) -> dict:
    """Write the manifest for exactly these chapters (the selection is the publication)."""
    manifest = {"version": 1, "series": series, "title": title,
                "chapters": sorted(chapters, key=lambda c: chapter_sort_key(c["id"]))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest


def run(cfg: Config, state: State, chapters: list[str], height: int = 1600, quality: int = 85,
        allow_bw: bool = True, force: bool = False, log=print) -> dict:
    series = cfg.preset.series
    root = cfg.work / "export"
    entries, tally = [], {}
    for cid in chapters:
        page_entries = []
        for key, _, idx, data in state.pages([cid]):
            src, why = pick_source(data, allow_bw)
            tally[why] = tally.get(why, 0) + 1
            if src is None:
                continue
            rel = f"{series}/{cid}/{idx:03d}.webp"
            dst = root / rel
            sig = f"{src}:{src.stat().st_mtime_ns}:{height}:{quality}"
            prev = data.get("export", {})
            if force or not dst.exists() or prev.get("sig") != sig:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as im:
                    out = resize_for_web(im, height)
                out.save(dst, "WEBP", quality=quality, method=6)
                w, h = out.size
                state.update(key, "export", {"key": rel, "w": w, "h": h, "sig": sig, "source": why})
            else:
                w, h = prev["w"], prev["h"]
            page_entries.append({"key": rel, "w": w, "h": h,
                                 "spread": bool(data.get("ingest", {}).get("spread", False))})
        entries.append(chapter_entry(series, cid, page_entries))
    manifest = write_manifest(root / series / "manifest.json", series, cfg.preset.title, entries)
    log(f"export: {len(entries)} chapters -> {root / series}; page sources {tally}; "
        f"manifest has {len(manifest['chapters'])} chapters")
    return manifest


def upload_stub(cfg: Config, chapters: list[str], log=print) -> list[tuple[str, Path]]:
    """Print what an upload would send. The real R2 method is defined elsewhere."""
    root = cfg.work / "export"
    series = cfg.preset.series
    manifest_path = root / series / "manifest.json"
    if not manifest_path.exists():
        log("upload: no manifest yet - run `dandadye export` first")
        return []
    manifest = json.loads(manifest_path.read_text())
    plan, total = [], 0
    for ch in manifest["chapters"]:
        if ch["id"] not in chapters:
            continue
        for p in ch["pages"]:
            f = root / p["key"]
            plan.append((p["key"], f))
            total += f.stat().st_size if f.exists() else 0
    plan.append((f"{series}/manifest.json", manifest_path))
    for k, f in plan[:5]:
        log(f"  would PUT {k}  <-  {f}")
    if len(plan) > 5:
        log(f"  ... {len(plan) - 5} more")
    log(f"upload (stub): {len(plan)} objects, {total / 1e6:.1f} MB. Nothing was uploaded.")
    return plan
