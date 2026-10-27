"""Stage 1 - ingest: index source pages into chapters and classify each page.

Per page we record size, whether it is a double-page spread, whether it is
already in colour (official colour pages: skip colourising, keep original and
export as a training pair), and whether it is blank or a sparse credit/text page.
We also report gaps in the page numbering (missing files, or numbers absorbed
by a spread).
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .state import State
from .util import chapter_number, chapter_sort_key, natural_key, rgb_to_lab

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Classification thresholds (measured on a thumbnail <= 384 px).
SPREAD_ASPECT = 1.15        # width / height above this = double-page spread
COLOUR_CHROMA = 12.0        # LAB chroma for a pixel to count as "coloured"
COLOUR_FRACTION = 0.08      # fraction of coloured pixels for a colour page, or ...
COLOUR_OF_INKED = 0.25      # ... share of non-white pixels that are coloured (pin-ups on white)
COLOUR_MIN = 0.01           # (with at least this overall fraction, to ignore tiny logos)
BLANK_STD = 2.5             # luminance std below this = blank
SPARSE_INK = 0.012          # fraction of ink pixels below this = credit/text page


@dataclass
class SourcePage:
    path: Path
    numbers: tuple[int, ...] | None  # page number(s) parsed from the filename; 2 = spread range


def _chapter_id(raw: str) -> str:
    n = float(raw)
    return str(int(n)) if n.is_integer() else str(n)


def _numbers_from_stem(stem: str, chapter: str) -> tuple[int, ...] | None:
    nums = [int(x) for x in re.findall(r"\d+", stem)]
    cnum = chapter_number(chapter)
    if len(nums) > 1 and cnum is not None and nums[0] == cnum:
        nums = nums[1:]  # "0225-001": leading chapter prefix
    if len(nums) == 1:
        return (nums[0],)
    if len(nums) == 2 and nums[1] == nums[0] + 1:
        return (nums[0], nums[1])  # "006-007": one file covering two pages
    return None


def discover(cfg: Config) -> dict[str, list[SourcePage]]:
    """Map chapter id -> source pages in reading order."""
    p = cfg.preset
    vol_re = re.compile(p.volume_dir_pattern)
    ch_re = re.compile(p.volume_chapter_regex)
    pg_re = re.compile(p.volume_page_regex)
    chapters: dict[str, list[tuple[tuple, SourcePage]]] = {}
    for d in sorted(cfg.source_dir.iterdir(), key=lambda x: natural_key(x.name)):
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS and not f.name.startswith(".")]
        if vol_re.match(d.name):
            for f in files:
                cm = ch_re.search(f.name)
                if not cm:
                    continue
                pm = pg_re.search(f.name.replace(cm.group(0), ""))
                num = None
                if pm:
                    nums = [int(g) for g in pm.groups() if g is not None]
                    num = (nums[0], nums[-1]) if len(nums) > 1 else (nums[0],)
                key = (num[0] if num else 10**9, natural_key(f.name))
                chapters.setdefault(_chapter_id(cm.group(1)), []).append((key, SourcePage(f, num)))
        else:
            cid = d.name
            for f in files:
                num = _numbers_from_stem(f.stem, cid)
                chapters.setdefault(cid, []).append(((0, natural_key(f.name)), SourcePage(f, num)))
    return {c: [sp for _, sp in sorted(v, key=lambda t: t[0])]
            for c, v in sorted(chapters.items(), key=lambda kv: chapter_sort_key(kv[0]))}


def analyse_image(path: Path) -> dict:
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
        im.draft("RGB", (768, 768))  # fast JPEG downscale while decoding
        im = im.convert("RGB")
        im.thumbnail((384, 384))
        rgb = np.asarray(im)
    lab = rgb_to_lab(rgb)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    coloured = chroma > COLOUR_CHROMA
    inked = L < 92
    stats = {
        "colour_fraction": round(float(np.mean(coloured)), 4),
        "colour_of_inked": round(float(coloured[inked].mean()) if inked.any() else 0.0, 4),
        "ink_fraction": round(float(np.mean(L < 35)), 4),
        "white_fraction": round(float(np.mean(L > 90)), 4),
        "lum_std": round(float(L.std()), 2),
    }
    return {"w": w, "h": h, "mode": mode, **stats, **classify(w, h, stats)}


def classify(w: int, h: int, s: dict) -> dict:
    spread = w / h > SPREAD_ASPECT
    colour = s["colour_fraction"] > COLOUR_FRACTION or (
        s["colour_fraction"] > COLOUR_MIN and s.get("colour_of_inked", 0) > COLOUR_OF_INKED)
    blank = s["lum_std"] < BLANK_STD
    sparse = not blank and not colour and s["ink_fraction"] < SPARSE_INK
    return {"spread": bool(spread), "colour": bool(colour), "blank": bool(blank), "sparse": bool(sparse)}


def find_gaps(pages: list[SourcePage]) -> list[int]:
    covered = {n for p in pages if p.numbers for n in range(p.numbers[0], p.numbers[-1] + 1)}
    if not covered:
        return []
    return sorted(set(range(min(covered), max(covered) + 1)) - covered)


def explain_gaps(missing: list[int], pages: list[SourcePage], rows: list) -> list[str]:
    """Say whether each missing page number sits next to a spread (likely absorbed by it)."""
    notes = []
    for n in missing:
        hint = "missing file"
        for (_, _, idx, data), sp in zip(rows, pages):
            if sp.numbers and data["ingest"].get("spread") and (n + 1 in sp.numbers or n - 1 in sp.numbers):
                hint = f"probably part of the spread {sp.path.name} (page {idx})"
                break
        notes.append(f"{n}: {hint}")
    return notes


def _file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


def _analyse_job(path_str: str) -> dict:
    try:
        return analyse_image(Path(path_str))
    except Exception as e:  # unreadable file: record, don't crash the run
        return {"error": str(e)}


def run(cfg: Config, chapters: list[str] | None = None, force: bool = False,
        export_pairs: bool = True, workers: int | None = None, log=print) -> dict:
    state = State(cfg.state_path)
    found = discover(cfg)
    if chapters is not None:
        found = {c: v for c, v in found.items() if c in chapters}

    jobs: dict[str, tuple[str, int, SourcePage]] = {}
    for cid, pages in found.items():
        for i, sp in enumerate(pages, start=1):
            key = f"{cid}/{i:03d}"
            prev = state.get(key)
            sig = _file_sig(sp.path)
            if not force and prev and prev.get("ingest", {}).get("src") == str(sp.path) \
                    and prev["ingest"].get("sig") == sig:
                continue
            jobs[key] = (cid, i, sp)

    log(f"ingest: {sum(len(v) for v in found.values())} pages in {len(found)} chapters, "
        f"{len(jobs)} to analyse")
    keys = list(jobs)
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 4) - 1)) as ex:
        results = ex.map(_analyse_job, [str(jobs[k][2].path) for k in keys], chunksize=8)
        for n, (key, res) in enumerate(zip(keys, results), start=1):
            cid, i, sp = jobs[key]
            prev = state.get(key) or {}
            if prev.get("ingest", {}).get("src") != str(sp.path):
                prev = {}  # page order changed: earlier stage results are stale
            prev["ingest"] = {"src": str(sp.path), "sig": _file_sig(sp.path),
                              "numbers": list(sp.numbers) if sp.numbers else None, **res}
            state.put(cid, i, prev)
            if n % 250 == 0:
                log(f"  analysed {n}/{len(keys)}")

    summary = {}
    for cid, pages in found.items():
        state.delete_chapter_pages(cid, set(range(1, len(pages) + 1)))
        rows = state.pages([cid])
        info = {
            "pages": len(rows),
            "spreads": [r[2] for r in rows if r[3]["ingest"].get("spread")],
            "colour": [r[2] for r in rows if r[3]["ingest"].get("colour")],
            "blank": [r[2] for r in rows if r[3]["ingest"].get("blank")],
            "sparse": [r[2] for r in rows if r[3]["ingest"].get("sparse")],
            "errors": [r[2] for r in rows if r[3]["ingest"].get("error")],
            "missing_numbers": find_gaps(pages),
            "unnumbered": [i for i, sp in enumerate(pages, 1) if sp.numbers is None],
        }
        info["gap_notes"] = explain_gaps(info["missing_numbers"], pages, rows)
        state.put_chapter(cid, info)
        summary[cid] = info

    if export_pairs:
        n = export_training_pairs(cfg, state, list(found))
        log(f"ingest: {n} colour pages exported as training pairs -> {cfg.work / 'training_pairs'}")
    state.close()
    return summary


def export_training_pairs(cfg: Config, state: State, chapters: list[str]) -> int:
    """Copy official colour pages + a greyscale version as (input, target) pairs."""
    out = cfg.stage_dir("training_pairs")
    n = 0
    for key, cid, idx, data in state.pages(chapters):
        ing = data.get("ingest", {})
        if not ing.get("colour") or ing.get("error"):
            continue
        stem = f"{cid}_{idx:03d}"
        target, source = out / f"{stem}_colour.png", out / f"{stem}_bw.png"
        if not target.exists():
            with Image.open(ing["src"]) as im:
                rgb = im.convert("RGB")
                rgb.save(target)
                rgb.convert("L").save(source)
        n += 1
    return n


def format_summary(summary: dict) -> str:
    lines = []
    for cid, s in summary.items():
        extras = []
        for k in ("spreads", "colour", "blank", "sparse", "errors", "gap_notes", "unnumbered"):
            if s.get(k):
                extras.append(f"{k}={s[k]}")
        lines.append(f"{cid:>6}: {s['pages']:3d} pages  " + "  ".join(extras))
    return "\n".join(lines)
