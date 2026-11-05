# dan-da-dye

Colours black-and-white manga on a Mac, one page at a time.

It was built for one job: turning *Dandadan*, a manga drawn in black and white on purpose, into a colour
edition nobody asked for, to be read in a private web reader on a folding phone. It runs entirely on Apple
Silicon. There is no cloud GPU, no subscription and no hurry, which is fortunate, because it is not fast.

> **This is AI slop, and it says so on the tin.** The code was written with Claude (Anthropic's model),
> mostly by describing what was wanted and then reading what came back. The pages are painted by a
> diffusion model making educated guesses about what colour a ghost is. A human reviewed both, mostly
> out of a desire to read the manga. It works. It is also occasionally, confidently, wrong about hair.

**Status:** early. Every stage runs; the colouring stage has been dry-run more often than it has been run.

## What it is for

- Colouring manga pages **you already own**, for **your own reading**. The repo ships no pages, no
  coloured output and no official art, and its `.gitignore` refuses image files on principle.
- Apple Silicon Macs (M1 or later). The models run on the GPU through
  [MLX](https://github.com/ml-explore/mlx) (via [mflux](https://github.com/filipstrand/mflux)) and PyTorch's
  MPS backend, both of which sit on Metal. There is no CUDA path, and nobody has missed it.
- Not for: selling anything, publishing anything, or reading anything you did not buy. The models' own
  licences say non-commercial (see Credits), and so does common sense.

What you need: macOS on Apple Silicon, Python 3.12, about 90 GB of disk for the models, and 16 GB of memory
at the very least (see "Memory" below). Patience is not listed as a dependency but is required.

## How it works

```
ingest -> detect -> refsheet -> colourise -> inkfix -> export -> upload
                                                  \-> qa
```

| Stage | What it does |
|---|---|
| `ingest` | Indexes every page into chapters (natural sort; volume folders are split into chapters by the `cNNN` in the filename). Records size and flags **spreads** (wide aspect), **colour pages** (official colour inserts: kept as they are, never colourised, and also saved to `work/training_pairs/` as colour/greyscale pairs for a future LoRA), **blank** pages and **sparse** credit/afterword pages. Reports gaps in the page numbering and whether a gap is explained by a neighbouring spread. |
| `detect` | Runs [Magi v2](https://huggingface.co/ragavsachdeva/magiv2) per chapter for panels, text and character boxes. It names characters against a **character bank** built from the preset's official art (a full figure and an upper-body crop per character). On top of that, characters are clustered chapter-wide on Magi's crop embeddings, so you can fix a whole cluster at once in `work/detect/<chapter>/names.json` (`{"3": "Momo"}`). `--viz` writes box overlays for checking. The detector is pluggable (`dandadye.detect.DETECTORS`). |
| `refsheet` | Builds a reference image per page from the official art of the characters on it, plus a palette swatch strip under each figure. Falls back to the main cast. Builders are pluggable (`refsheet.BUILDERS`). Pages with the same cast share one sheet. |
| `colourise` | Runs mflux. Engine `flux2-lora` (default): FLUX.2 Klein base 9B + LoRA, **image 1 = page, image 2 = reference sheet**, both at ~1 MP with 16-px aligned sides, prompt `mngclranm`, 50 steps, guidance 4, LoRA 0.9. Engine `flux3`: FLUX 3 Image Klein, the page plus one reference per character (up to 9), a full caption naming each character's palette, 16 steps, no LoRA. `--backend api` loads the model once per batch (and frees the text encoder after encoding the fixed LoRA prompt); `--backend cli` runs one mflux process per page. `--dry-run` prints the exact commands and checks them with mflux's own argument parser. |
| `inkfix` | Resizes the colour to the page, removes the model's own thin lines (morphological closing), **multiplies the original greyscale page over it**, forces ink to neutral black and near-white paper to pure white, restores lettering inside Magi's text boxes from the original, and flood-fills the rest of each speech balloon white. Gutters outside Magi's panels become white. |
| `export` | Writes `work/export/<series>/<chapter>/<NNN>.webp` (≤1600 px tall, never upscaled, quality 85) and `work/export/manifest.json` in the reader's exact v1 format, validated by a Python port of the reader's strict validator. Official colour pages, blank and credit pages are published as the originals. Pages without a colour version fall back to B&W (`--no-allow-bw` to omit them). Blank pages are dropped unless `--keep-blank` is given. |
| `upload` | Uploads to R2 with `wrangler r2 object put --remote`: new page objects first, **manifest last**. Without `--yes` it only prints the plan. Keys are immutable: once a key is published (recorded in `work/export/published.json`), a changed page gets a new key such as `001.v2.webp`. Refuses a manifest that would drop published chapters unless `--allow-removal` is given. |
| `qa` | Placeholder interface for a later vision-model check of colour drift. The current `heuristic` checker flags pages that stayed grey and measures each named character's distance to their palette. |
| `status` | Per-chapter progress per stage. |

All state lives in `<data>/work/state.sqlite` (one JSON record per page), so every stage is
idempotent: it skips finished work, and `--force` redoes it. The common options are `--from 150 --to 160`,
`--chapters 150,245.1` and `--pages 1-5`.

## Setup

**Moved the repo?** The `.venv` remembers where it was installed from (`pip install -e`). Either reinstall
from the new place (`"$DANDADYE_DATA/.venv/bin/pip" install -e .`) or leave a symlink at the old path.

Apple Silicon Mac, Python 3.12.

```sh
# data root: pages, models, art, outputs (outside the repo)
export DANDADYE_DATA="/path/to/data-root"
export HF_HOME="$DANDADYE_DATA/hf"

python3.12 -m venv "$DANDADYE_DATA/.venv"
"$DANDADYE_DATA/.venv/bin/pip" install -e ".[mlx,dev]"     # mflux + this package

# Magi v2 in its own venv (torch); dandadye calls it through a subprocess
python3.12 -m venv "$DANDADYE_DATA/.venv-magi"
"$DANDADYE_DATA/.venv-magi/bin/pip" install -r requirements-magi.txt
```

Expected data-root layout (what the Dandadan preset points at):

```
<data>/work/src/<chapter or vNN>/...     source pages
<data>/palette/palette.json              character -> region -> base/shadow/highlight hex
<data>/palette/img/<name>/<name>.png     official setting art (transparent PNG)
<data>/models/magiv2/                    ragavsachdeva/magiv2 snapshot
<data>/lora/manga-colorization/          thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA
<data>/hf/                               Hugging Face cache (FLUX.2-klein-base-9B, flux-3-image-klein)
```

Then:

```sh
dandadye ingest
dandadye detect   --chapters 150 --viz
dandadye refsheet --from 150
dandadye colourise --from 150 --dry-run          # check the commands
dandadye colourise --from 150 -q 4               # real run (slow)
dandadye inkfix   --from 150
dandadye export   --from 150
dandadye upload   --from 150                      # prints the plan
dandadye upload   --from 150 --yes --reader-dir /path/to/reader-project
pytest
```

### Memory (16 GB Macs)

FLUX.2 Klein base 9B in bf16 does not fit in 16 GB with its 8B text encoder. Use `-q 4` (the default)
and `--low-ram` (on by default). With the `api` backend, the fixed LoRA prompt is encoded once and the
text encoder is then freed. Quantising when the model loads still briefly needs the full weights. Saving
a quantised copy once with `mflux-save --model flux2-klein-base-9b -q 4 --path ...` and passing
`--model-path` avoids that.

## Adding a series

Copy `presets/dandadan/preset.toml`. Set `series`, `title`, `source_dir`, the filename regexes
for volume folders, the palette path, and one `[[characters]]` entry per character (`art`,
`palette_key`, `main_cast`). Optional `[colourise]` overrides (e.g. `prompt_extra`,
`lora_scale`), plus `[colourise.flux3]` for the FLUX 3 engine.

## Credits and licences

- **Code**: Apache-2.0.
- **FLUX.2 [klein] 9B / base 9B** (Black Forest Labs): FLUX Non-Commercial License. **Personal,
  non-commercial use only.** The same applies to **FLUX 3 Image Klein**; check its model card.
- **Manga colourisation LoRA** by thedeoxen
  ([FLUX.2-klein-9B-manga-colorization-by-reference-LORA](https://huggingface.co/thedeoxen/FLUX.2-klein-9B-manga-colorization-by-reference-LORA)):
  Apache-2.0. It still inherits the base model's non-commercial terms when used with it.
- **Magi v2**: Ragav Sachdeva, Gyungin Shin, Andrew Zisserman, *Tails Tell Tales: Chapter-Wide
  Manga Transcriptions with Character Names* (arXiv:2408.00298). The model card has no standard
  licence. It states use is "unrestricted ... in personal, research, non-commercial, and
  not-for-profit endeavors"; any other use needs the author's permission.
- **[mflux](https://github.com/filipstrand/mflux)**: MLX implementation of FLUX (MIT).
- Manga pages and official character art belong to their rights holders. This project ships none of
  them. Colourised pages are for personal reading.
