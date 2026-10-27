"""Stage 6 - export (webp + manifest.json) and upload (R2 via wrangler).

The reader's contract (version 1, strict - unknown fields are refused)::

    {"version":1,"series":"dandadan","title":"Dandadan","chapters":[
      {"id":"150","number":150,"title":"Chapter 150",
       "pages":[{"key":"dandadan/150/001.webp","w":1600,"h":2300,"spread":false}]}]}

Local layout mirrors the bucket: ``work/export/manifest.json`` (bucket root) and
``work/export/<key>``.

Keys are immutable once uploaded (the reader caches them forever). If a page's
image changes after its key was published, it gets a new key ``001.v2.webp``,
``001.v3.webp``... Published keys are recorded in ``work/export/published.json``.
Upload order: all page images first, the manifest last, so the manifest only
ever references objects that exist.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from PIL import Image

from .config import Config
from .state import State
from .util import chapter_number, chapter_sort_key

KEY_PATTERN = re.compile(r"^(?!.*\.\.)[A-Za-z0-9_-][A-Za-z0-9._-]*(?:/[A-Za-z0-9_-][A-Za-z0-9._-]*)*\.(?:webp|avif|jpe?g|png)$")
CHAPTER_ID_PATTERN = re.compile(r"^\d{1,5}(?:\.\d{1,3})?$")
DEFAULT_BUCKET = "dandadan-manga"


# --- manifest ---------------------------------------------------------------------

def validate_manifest(m) -> list[str]:
    """Python port of the reader's validateManifest (src/manifest.ts). [] = valid."""
    errs: list[str] = []

    def text(v):
        return isinstance(v, str) and v.strip() != "" and len(v) <= 200

    def dim(v):
        return isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 30000

    if not isinstance(m, dict):
        return ["manifest: must be a JSON object"]
    errs += [f'manifest: unknown field "{k}"' for k in m if k not in ("version", "series", "title", "chapters")]
    if m.get("version") != 1 or isinstance(m.get("version"), bool):
        errs.append("manifest.version: must be 1")
    if not text(m.get("series")):
        errs.append("manifest.series: must be a non-empty string")
    if not text(m.get("title")):
        errs.append("manifest.title: must be a non-empty string")
    chapters = m.get("chapters")
    if not isinstance(chapters, list):
        return errs + ["manifest.chapters: must be an array"]
    ids, keys = set(), set()
    for ci, ch in enumerate(chapters):
        at = f"chapters[{ci}]"
        if not isinstance(ch, dict):
            errs.append(f"{at}: must be an object")
            continue
        errs += [f'{at}: unknown field "{k}"' for k in ch if k not in ("id", "number", "title", "pages")]
        cid = ch.get("id")
        if not isinstance(cid, str) or not CHAPTER_ID_PATTERN.match(cid):
            errs.append(f'{at}.id: must be a string like "150" or "245.1"')
        elif cid in ids:
            errs.append(f'{at}.id: duplicate "{cid}"')
        else:
            ids.add(cid)
        num = ch.get("number")
        if not isinstance(num, (int, float)) or isinstance(num, bool) or num < 0:
            errs.append(f"{at}.number: must be a non-negative number")
        if not text(ch.get("title")):
            errs.append(f"{at}.title: must be a non-empty string")
        pages = ch.get("pages")
        if not isinstance(pages, list) or not pages:
            errs.append(f"{at}.pages: must be a non-empty array")
            continue
        for pi, p in enumerate(pages):
            pat = f"{at}.pages[{pi}]"
            if not isinstance(p, dict):
                errs.append(f"{pat}: must be an object")
                continue
            errs += [f'{pat}: unknown field "{k}"' for k in p if k not in ("key", "w", "h", "spread")]
            k = p.get("key")
            if not isinstance(k, str) or len(k) > 512 or not KEY_PATTERN.match(k):
                errs.append(f"{pat}.key: must be a relative R2 key ending .webp/.avif/.jpg/.png")
            elif k in keys:
                errs.append(f'{pat}.key: duplicate "{k}"')
            else:
                keys.add(k)
            if not dim(p.get("w")):
                errs.append(f"{pat}.w: must be a positive integer")
            if not dim(p.get("h")):
                errs.append(f"{pat}.h: must be a positive integer")
            if not isinstance(p.get("spread"), bool):
                errs.append(f"{pat}.spread: must be true or false")
    return errs


def chapter_entry(cid: str, pages: list[dict]) -> dict:
    n = chapter_number(cid)
    number = (int(n) if n.is_integer() else n) if n is not None else cid
    return {"id": cid, "number": number, "title": f"Chapter {cid}", "pages": pages}


def write_manifest(path: Path, series: str, title: str, chapters: list[dict]) -> dict:
    """Write the manifest for exactly these chapters (the selection is the publication)."""
    manifest = {"version": 1, "series": series, "title": title,
                "chapters": sorted(chapters, key=lambda c: chapter_sort_key(c["id"]))}
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("manifest invalid:\n- " + "\n- ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest


# --- published-key ledger -------------------------------------------------------

class Ledger:
    """Which keys have been uploaded (and so must never be overwritten)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {"keys": {}, "manifest": None}

    def published(self, key: str) -> bool:
        return key in self.data["keys"]

    def mark(self, key: str, sig: str | None = None) -> None:
        self.data["keys"][key] = {"sig": sig, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


def versioned_key(base: str, version: int) -> str:
    return f"{base}.webp" if version <= 1 else f"{base}.v{version}.webp"


def assign_key(base: str, prev: dict, sig: str, ledger: Ledger) -> tuple[str, int]:
    """Same image -> same key. Changed image -> reuse the key only if it was never published."""
    if prev.get("key") and prev.get("sig") == sig:
        return prev["key"], prev.get("version", 1)
    version = prev.get("version", 1)
    if prev.get("key") and not ledger.published(prev["key"]):
        return prev["key"], version
    if prev.get("key"):
        version += 1
    while ledger.published(versioned_key(base, version)):
        version += 1
    return versioned_key(base, version), version


# --- export ---------------------------------------------------------------------

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


def export_root(cfg: Config) -> Path:
    return cfg.work / "export"


def run(cfg: Config, state: State, chapters: list[str], height: int = 1600, quality: int = 85,
        allow_bw: bool = True, force: bool = False, keep_blank: bool = False, log=print) -> dict:
    series = cfg.preset.series
    root = export_root(cfg)
    ledger = Ledger(root / "published.json")
    entries, tally = [], {}
    for cid in chapters:
        page_entries = []
        for key, _, idx, data in state.pages([cid]):
            if data.get("ingest", {}).get("blank") and not keep_blank:
                tally["blank-skipped"] = tally.get("blank-skipped", 0) + 1
                continue
            src, why = pick_source(data, allow_bw)
            tally[why] = tally.get(why, 0) + 1
            if src is None:
                continue
            sig = f"{src}:{src.stat().st_mtime_ns}:{height}:{quality}"
            prev = data.get("export", {})
            obj, version = assign_key(f"{series}/{cid}/{idx:03d}", prev, sig, ledger)
            dst = root / obj
            if force and ledger.published(obj) and prev.get("sig") == sig:
                pass  # never rewrite a published object, even with --force
            elif force or not dst.exists() or prev.get("sig") != sig or prev.get("key") != obj:
                dst.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(src) as im:
                    out = resize_for_web(im, height)
                out.save(dst, "WEBP", quality=quality, method=6)
                prev = {"key": obj, "version": version, "w": out.width, "h": out.height,
                        "sig": sig, "source": why}
                state.update(key, "export", prev)
            page_entries.append({"key": obj, "w": prev["w"], "h": prev["h"],
                                 "spread": bool(data.get("ingest", {}).get("spread", False))})
        if page_entries:
            entries.append(chapter_entry(cid, page_entries))
    manifest = write_manifest(root / "manifest.json", series, cfg.preset.title, entries)
    log(f"export: {len(entries)} chapters, {sum(len(c['pages']) for c in entries)} pages -> {root}; "
        f"sources {tally}")
    return manifest


# --- upload ---------------------------------------------------------------------

def upload_plan(cfg: Config, chapters: list[str] | None = None, allow_removal: bool = False) -> dict:
    """Work out what to upload; raise if the manifest would be unsafe to publish."""
    root = export_root(cfg)
    mpath = root / "manifest.json"
    if not mpath.exists():
        raise SystemExit("upload: no manifest yet - run `dandadye export` first")
    manifest = json.loads(mpath.read_text())
    errors = validate_manifest(manifest)
    if errors:
        raise SystemExit("upload: manifest invalid:\n- " + "\n- ".join(errors))
    if chapters is not None:
        extra = [c["id"] for c in manifest["chapters"] if c["id"] not in chapters]
        if extra:
            raise SystemExit(f"upload: manifest contains chapters outside the selection {extra}; "
                             "re-run export with the same selection")
    ledger = Ledger(root / "published.json")
    pages, missing = [], []
    for ch in manifest["chapters"]:
        for p in ch["pages"]:
            if ledger.published(p["key"]):
                continue
            f = root / p["key"]
            (pages if f.exists() else missing).append((p["key"], f))
    if missing:
        raise SystemExit(f"upload: {len(missing)} manifest keys have no local file, e.g. {missing[0][0]}")
    prev = ledger.data.get("manifest")
    if prev and not allow_removal:
        gone = sorted({c["id"] for c in prev["chapters"]} - {c["id"] for c in manifest["chapters"]},
                      key=chapter_sort_key)
        if gone:
            raise SystemExit(f"upload: new manifest drops published chapters {gone[:10]}... "
                             "(pass --allow-removal if intended)")
    return {"manifest": manifest, "manifest_path": mpath, "pages": pages, "ledger": ledger}


def wrangler_put(bucket: str, key: str, file: Path, content_type: str, cache_control: str | None = None) -> list[str]:
    cmd = ["npx", "wrangler", "r2", "object", "put", f"{bucket}/{key}", "--file", str(file),
           "--content-type", content_type]
    if cache_control:
        cmd += ["--cache-control", cache_control]
    return cmd + ["--remote"]


def upload(cfg: Config, chapters: list[str] | None, yes: bool = False, reader_dir: str | None = None,
           bucket: str = DEFAULT_BUCKET, allow_removal: bool = False, log=print,
           runner=subprocess.run) -> list[list[str]]:
    """Upload new pages, then the manifest. Without ``yes`` only prints the plan."""
    plan = upload_plan(cfg, chapters, allow_removal)
    cmds = [(key, wrangler_put(bucket, key, f, "image/webp")) for key, f in plan["pages"]]
    cmds.append(("manifest.json", wrangler_put(bucket, "manifest.json", plan["manifest_path"],
                                               "application/json", "no-cache")))
    size = sum(f.stat().st_size for _, f in plan["pages"])
    log(f"upload: {len(plan['pages'])} new page objects ({size / 1e6:.1f} MB), then manifest.json "
        f"({len(plan['manifest']['chapters'])} chapters) -> r2://{bucket}")
    for _, c in cmds[:3] + ([("...", ["..."])] if len(cmds) > 4 else []) + cmds[-1:]:
        log("  " + " ".join(c))
    if not yes:
        log("upload: dry run - nothing was uploaded. Re-run with --yes to upload.")
        return [c for _, c in cmds]

    if not reader_dir:
        raise SystemExit("upload --yes needs --reader-dir (the reader project, where wrangler is configured)")
    rdir = Path(reader_dir)
    validator = rdir / "scripts" / "validate-manifest.mjs"
    if validator.exists() and shutil.which("node"):
        runner(["node", str(validator), str(plan["manifest_path"])], cwd=rdir, check=True)
    env = {**os.environ, "WRANGLER_SEND_METRICS": "false"}
    ledger = plan["ledger"]
    for key, c in cmds:
        runner(c, cwd=rdir, env=env, check=True)
        if key != "manifest.json":
            ledger.mark(key)
    ledger.data["manifest"] = plan["manifest"]
    ledger.save()
    log(f"upload: done ({len(cmds)} objects)")
    return [c for _, c in cmds]
