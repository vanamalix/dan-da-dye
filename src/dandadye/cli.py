"""Command line: ``dandadye <stage> [options]``.

Every stage reads and writes the per-page state database under
``<data>/work/state.sqlite`` and skips work that is already done unless
``--force`` is given, so any stage can be re-run or resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import load_config
from .util import parse_index_ranges, select_chapters


def _add_common(p: argparse.ArgumentParser, chapters: bool = True) -> None:
    p.add_argument("--data", help="data root (default: $DANDADYE_DATA)")
    p.add_argument("--preset", default="dandadan", help="preset name or path to preset.toml")
    if chapters:
        p.add_argument("--from", dest="start", type=float, help="first chapter number (inclusive)")
        p.add_argument("--to", dest="end", type=float, help="last chapter number (inclusive)")
        p.add_argument("--chapters", help="comma-separated chapter ids, e.g. 150,245.1")
        p.add_argument("--pages", help="page indices within each chapter, e.g. 1-5,9")
        p.add_argument("--force", action="store_true", help="redo work that is already done")


def _chapters(args, available: list[str]) -> list[str]:
    only = args.chapters.split(",") if getattr(args, "chapters", None) else None
    return select_chapters(available, args.start, args.end, only)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dandadye", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("ingest", help="index pages; flag spreads, colour, blank and credit pages")
    _add_common(p)
    p.add_argument("--no-pairs", action="store_true", help="don't export colour pages as training pairs")
    p.add_argument("--workers", type=int)
    p.add_argument("--json", action="store_true", help="print the summary as JSON")

    p = sub.add_parser("detect", help="panels, characters and names per page (Magi v2)")
    _add_common(p)
    p.add_argument("--detector", default="magi-v2")
    p.add_argument("--device", default="auto", help="auto | mps | cpu")
    p.add_argument("--viz", action="store_true", help="also write box overlays to work/detect_viz/")
    p.add_argument("--eta", type=float, default=0.75, help="Magi 'none of the above' cost; lower = more 'Other'")

    p = sub.add_parser("refsheet", help="build per-page colour reference sheets")
    _add_common(p)
    p.add_argument("--builder", default="cast_strip")
    p.add_argument("--max-characters", type=int, default=4)

    p = sub.add_parser("colourise", help="run FLUX.2 Klein edit + LoRA on each page")
    _add_common(p)
    p.add_argument("--engine", choices=["flux2-lora", "flux3"], default="flux2-lora",
                   help="flux2-lora = FLUX.2 Klein base 9B + manga LoRA; flux3 = FLUX 3 Image Klein multi-reference")
    p.add_argument("--dry-run", action="store_true", help="print the mflux commands, don't run")
    p.add_argument("--backend", choices=["api", "cli"], default="api",
                   help="api = load the model once per batch (fast); cli = one mflux process per page")
    p.add_argument("--model")
    p.add_argument("--model-path", help="local (e.g. pre-quantised) model directory")
    p.add_argument("-q", "--quantize", type=int, choices=[3, 4, 5, 6, 8])
    p.add_argument("--no-quantize", action="store_true")
    p.add_argument("--steps", type=int)
    p.add_argument("--guidance", type=float)
    p.add_argument("--lora-scale", type=float)
    p.add_argument("--megapixels", type=float, help="working resolution of the page (default 1.0)")
    p.add_argument("--seed", type=int)
    p.add_argument("--prompt-extra")
    p.add_argument("--low-ram", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--include-skipped", action="store_true",
                   help="also colourise pages flagged colour/blank/sparse")

    p = sub.add_parser("inkfix", help="restore original linework and lettering over the colour")
    _add_common(p)
    p.add_argument("--engine", choices=["flux2-lora", "flux3"], default="flux2-lora",
                   help="which colourise output to use (work/colour/<engine>/)")
    p.add_argument("--colour-dir", help="take colour images from here instead of work/colour/<engine>")

    p = sub.add_parser("export", help="write webp pages + manifest.json")
    _add_common(p)
    p.add_argument("--height", type=int, default=1600, help="target page height (never upscales)")
    p.add_argument("--quality", type=int, default=85)
    p.add_argument("--allow-bw", action=argparse.BooleanOptionalAction, default=True,
                   help="fall back to the original page when no colour version exists")
    p.add_argument("--keep-blank", action="store_true", help="also publish pages flagged blank")

    p = sub.add_parser("upload", help="upload new pages then manifest.json to R2 (dry run unless --yes)")
    _add_common(p)
    p.add_argument("--yes", action="store_true", help="really upload (runs wrangler r2 object put --remote)")
    p.add_argument("--reader-dir", default=os.environ.get("DANDADYE_READER_DIR"),
                   help="reader project dir where wrangler runs (default: $DANDADYE_READER_DIR)")
    p.add_argument("--bucket", default="dandadan-manga")
    p.add_argument("--allow-removal", action="store_true",
                   help="allow a manifest that drops chapters already published")

    p = sub.add_parser("qa", help="(placeholder) check colour drift against the palette")
    _add_common(p)
    p.add_argument("--checker", default="heuristic")

    p = sub.add_parser("status", help="show per-stage progress")
    _add_common(p)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.data, args.preset)

    if args.stage == "ingest":
        from . import ingest
        only = None
        if args.chapters or args.start is not None or args.end is not None:
            only = _chapters(args, list(ingest.discover(cfg)))
        summary = ingest.run(cfg, only, force=args.force, export_pairs=not args.no_pairs,
                             workers=args.workers, log=lambda m: print(m, file=sys.stderr))
        print(json.dumps(summary, indent=1) if args.json else ingest.format_summary(summary))
        return 0

    from .state import State
    state = State(cfg.state_path)
    chapters = _chapters(args, state.chapters())
    pages = parse_index_ranges(getattr(args, "pages", None))
    if not chapters:
        print("no chapters selected (did you run `dandadye ingest`?)", file=sys.stderr)
        return 1

    if args.stage == "detect":
        from .detect import run as detect_run
        detect_run(cfg, state, chapters, pages, detector=args.detector, device=args.device,
                   force=args.force, viz=args.viz, eta=args.eta)
    elif args.stage == "refsheet":
        from . import refsheet
        refsheet.run(cfg, state, chapters, pages, builder=args.builder,
                     max_characters=args.max_characters, force=args.force)
    elif args.stage == "colourise":
        from . import colourise
        settings = colourise.Settings.from_config(cfg, args)
        colourise.run(cfg, state, chapters, pages, settings, dry_run=args.dry_run,
                      backend=args.backend, force=args.force, include_skipped=args.include_skipped)
    elif args.stage == "inkfix":
        from . import inkfix
        inkfix.run(cfg, state, chapters, pages, force=args.force, colour_dir=args.colour_dir,
                   engine=args.engine)
    elif args.stage == "export":
        from . import export
        export.run(cfg, state, chapters, height=args.height, quality=args.quality,
                   allow_bw=args.allow_bw, force=args.force, keep_blank=args.keep_blank)
    elif args.stage == "upload":
        from . import export
        export.upload(cfg, chapters, yes=args.yes, reader_dir=args.reader_dir, bucket=args.bucket,
                      allow_removal=args.allow_removal)
    elif args.stage == "qa":
        from . import qa
        qa.run(cfg, state, chapters, pages, checker=args.checker, force=args.force)
    elif args.stage == "status":
        _status(state, chapters)
    state.close()
    return 0


def _status(state, chapters: list[str]) -> None:
    stages = ["ingest", "detect", "refsheet", "colourise", "inkfix", "export", "qa"]
    print(f"{'chapter':>8} {'pages':>5} " + " ".join(f"{s:>9}" for s in stages))
    for cid in chapters:
        rows = state.pages([cid])
        counts = [sum(1 for r in rows if s in r[3]) for s in stages]
        print(f"{cid:>8} {len(rows):>5} " + " ".join(f"{c:>9}" for c in counts))


if __name__ == "__main__":
    sys.exit(main())
