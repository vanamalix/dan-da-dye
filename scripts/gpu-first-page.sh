#!/usr/bin/env bash
# Time the full colourise stage on ONE synthetic page with the real FLUX.2 Klein base 9B + manga LoRA.
#
#   scripts/gpu-first-page.sh <data-root> [--dry-run] [-- extra `dandadye colourise` args]
#
# <data-root> must contain .venv/ (torch + diffusers + this package), hf/ (the Hugging Face cache with the
# weights) and lora/manga-colorization/manga_colorization.safetensors. Nothing in it is modified: the script
# builds a throwaway data root in a temp dir (symlinks to hf/ and lora/, a synthetic page, its own state
# database and preset) and runs the pipeline there. Offline: HF_HUB_OFFLINE=1.
#
# Stages run on the synthetic page: ingest (no model needed), refsheet (no model needed; the throwaway preset
# has one character without art, so the sheet is a plain white placeholder with no detect output; the detect
# stage needs the Magi model and is skipped), then colourise (the timed part).
#
# Timing: "colourise" is wall clock around the CLI. The CLI logs one per-page time that includes the model load
# (the engine is built inside the first page's timer), so set PAGES=2 to separate them: page 2 is pure
# generation, load is roughly page 1 minus page 2. Env: PAGES (default 1), KEEP_WORK=0 to delete the temp dir
# afterwards (default: keep, so the output image can be inspected), GPU_FIRST_PAGE_WORK=<dir> to pick the
# (empty or new) temp dir yourself.
#
# --dry-run: runs ingest + refsheet for real (cheap, CPU) and `colourise --dry-run`, which prints the load plan
# and the exact pipeline arguments without loading any weights.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
[[ $# -ge 1 && "$1" != -h && "$1" != --help ]] || usage 0
DATA=$(cd "$1" && pwd) || usage; shift
DRY=0
if [[ ${1:-} == --dry-run ]]; then DRY=1; shift; fi
[[ ${1:-} == -- ]] && shift
EXTRA=("$@")
PAGES=${PAGES:-1}

PY="$DATA/.venv/bin/python"
CLI="$DATA/.venv/bin/dandadye"
LORA_REL=lora/manga-colorization/manga_colorization.safetensors
[[ -x $PY && -x $CLI ]] || { echo "no $DATA/.venv (run scripts/setup.sh first)" >&2; exit 1; }
[[ -d $DATA/hf ]] || { echo "no $DATA/hf (Hugging Face cache)" >&2; exit 1; }
[[ -f $DATA/$LORA_REL ]] || { echo "no $DATA/$LORA_REL" >&2; exit 1; }

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${GPU_FIRST_PAGE_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/dandadye-first-page.XXXXXX")}
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
case $WORK/ in "$REPO"/*) echo "refusing to work inside the repo: $WORK" >&2; exit 1;; esac

# Throwaway data root: real weights/LoRA by symlink, everything else new.
ROOT=$WORK/data
mkdir -p "$ROOT/work/src/1" "$ROOT/lora"
ln -sfn "$DATA/hf" "$ROOT/hf"
ln -sfn "$DATA/lora/manga-colorization" "$ROOT/lora/manga-colorization"
cat > "$WORK/preset.toml" <<'TOML'
series = "synthetic"
title = "Synthetic test page"
source_dir = "work/src"

[[characters]]
name = "Test"
main_cast = true

[colourise]
prompt_extra = ""
TOML

# Synthetic line-art-like grayscale pages, 1000x1500: panels, hatching, faces, balloons with text.
"$PY" -I - "$ROOT/work/src/1" "$PAGES" <<'PYEOF'
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

out, n = Path(sys.argv[1]), int(sys.argv[2])
W, H = 1000, 1500
try:
    font = ImageFont.load_default(size=34)
except TypeError:
    font = ImageFont.load_default()

def page(seed):
    rnd = random.Random(seed)
    im = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(im)
    panels = [(40, 40, 960, 520), (40, 560, 480, 1000), (520, 560, 960, 1000), (40, 1040, 960, 1460)]
    for (x0, y0, x1, y1) in panels:
        d.rectangle([x0, y0, x1, y1], outline=0, width=8)
        for _ in range(6):                       # hatching / speed lines
            x = rnd.randint(x0 + 10, x1 - 10)
            d.line([x, y0 + 10, x + rnd.randint(-60, 60), y1 - 10], fill=60, width=2)
        cx, cy = (x0 + x1) // 2 + rnd.randint(-60, 60), (y0 + y1) // 2 + 30
        r = min(x1 - x0, y1 - y0) // 4
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=6, fill=235)   # head
        d.polygon([(cx - r, cy - r // 2), (cx, cy - r - 50), (cx + r, cy - r // 2), (cx, cy - r // 3)],
                  outline=0, fill=40)                                               # hair
        for ex in (-r // 3, r // 3):
            d.ellipse([cx + ex - 14, cy - 10, cx + ex + 14, cy + 18], outline=0, width=4, fill=255)
            d.ellipse([cx + ex - 5, cy - 2, cx + ex + 5, cy + 10], fill=0)
        d.arc([cx - r // 3, cy + r // 5, cx + r // 3, cy + r // 2], 20, 160, fill=0, width=4)
        d.rectangle([cx - r // 2, cy + r, cx + r // 2, y1 - 10], outline=0, width=5, fill=200)  # torso
        d.line([cx - r // 2, cy + r + 30, cx + r // 2, cy + r + 30], fill=0, width=4)
    for (bx, by, text) in [(80, 80, "WHAT IS THAT?!"), (600, 600, "RUN!"), (90, 1080, "I CAN'T BELIEVE\nIT WORKED")]:
        w, h = d.multiline_textbbox((0, 0), text, font=font)[2:]
        box = [bx, by, bx + w + 50, by + h + 40]
        d.ellipse(box, outline=0, width=5, fill=255)
        d.multiline_text((bx + 25, by + 20), text, font=font, fill=0, align="center")
    return im

out.mkdir(parents=True, exist_ok=True)
for i in range(1, n + 1):
    page(i).save(out / f"{i:03d}.png")
PYEOF

export HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export DANDADYE_DATA="$ROOT"
ARGS=(--data "$ROOT" --preset "$WORK/preset.toml")

now() { date +%s.%N; }
secs() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.1f", b-a}'; }

echo "work dir: $WORK  (data root: $DATA, pages: $PAGES, dry-run: $DRY)"
T0=$(now)
"$CLI" ingest "${ARGS[@]}" --no-pairs --workers 1 --json | "$PY" -I -c '
import json, sys; d = json.load(sys.stdin)
print("ingest:", {c: {k: v for k, v in s.items() if k in ("pages", "colour", "blank", "sparse", "errors")} for c, s in d.items()})'
T1=$(now)
"$CLI" refsheet "${ARGS[@]}"
T2=$(now)
echo "phase ingest: $(secs "$T0" "$T1")s   refsheet: $(secs "$T1" "$T2")s"

COL=("$CLI" colourise "${ARGS[@]}" --runtime diffusers)
[[ $DRY == 1 ]] && COL+=(--dry-run)
COL+=("${EXTRA[@]}")
echo "+ ${COL[*]}"
T3=$(now)
"${COL[@]}" 2>&1 | while IFS= read -r line; do printf '[+%6.1fs] %s\n' "$(secs "$T3" "$(now)")" "$line"; done
RC=${PIPESTATUS[0]}
T4=$(now)
[[ $RC == 0 ]] || { echo "colourise failed (exit $RC)" >&2; exit "$RC"; }

echo "phase colourise (wall, includes python/torch import, model load, $PAGES page(s)): $(secs "$T3" "$T4")s"
echo "total: $(secs "$T0" "$T4")s"
if [[ $DRY == 0 ]]; then
  echo "output: $ROOT/work/colour/flux2-lora/1/"
  ls -l "$ROOT"/work/colour/flux2-lora/1/*.png
  echo "per-page seconds from state (page 1 includes model load):"
  "$PY" -I -c '
import json, sqlite3, sys
for k, d in sqlite3.connect(sys.argv[1]).execute("select key, data from pages order by key"):
    c = json.loads(d).get("colourise", {}).get("flux2-lora")
    if c: print(f"  {k}: {c[\"seconds\"]}s {c[\"width\"]}x{c[\"height\"]} on {c.get(\"device_detail\", c.get(\"device\"))} ({c.get(\"dtype\")})")
' "$ROOT/work/state.sqlite"
fi
if [[ ${KEEP_WORK:-1} == 0 ]]; then rm -rf "$WORK"; else echo "kept: $WORK (delete it when done)"; fi
