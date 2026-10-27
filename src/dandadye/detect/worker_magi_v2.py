"""Magi v2 worker - runs inside its own venv (torch + transformers), not dandadye's.

Usage: python -I worker_magi_v2.py job.json result.json

job.json: {"model_dir": ..., "device": "auto|mps|cpu", "eta": 0.75,
           "pages": [path, ...], "bank": {"images": [path, ...], "names": [name, ...]}}

This file must stay standalone: it is executed by a different interpreter and
cannot import dandadye.
"""

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoConfig, AutoModel  # noqa: E402


def read_image(path):
    with Image.open(path) as im:
        return np.array(im.convert("L").convert("RGB"))


def use_native_solver():
    """Magi solves its name assignment with pulp, whose bundled CBC binary is x86-only
    on macOS. Point pulp's default at HiGHS (``pip install highspy``) when available."""
    import pulp
    try:
        solver = pulp.HiGHS(msg=False)
        if solver.available():
            pulp.LpSolverDefault = solver
            pulp.pulp.LpSolverDefault = solver
    except Exception:
        pass


def pick_device(name):
    if name == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return name


def run(job, device):
    # OCR off (we only need boxes); the timm ResNet backbone weights are inside the
    # checkpoint, so don't let timm try to download ImageNet weights.
    config = AutoConfig.from_pretrained(job["model_dir"], trust_remote_code=True, disable_ocr=True)
    config.detection_model_config.use_pretrained_backbone = False
    model = AutoModel.from_pretrained(job["model_dir"], config=config, trust_remote_code=True)
    model = model.to(device).eval()
    pages = [read_image(p) for p in job["pages"]]
    bank = {"images": [read_image(p) for p in job["bank"]["images"]], "names": job["bank"]["names"]}
    with torch.no_grad():
        results = model.do_chapter_wide_prediction(pages, bank, eta=job.get("eta", 0.75),
                                                   batch_size=job.get("batch_size", 4), do_ocr=False)
        # Crop embeddings for chapter-wide clustering and confidence scores.
        embs = model.predict_crop_embeddings(pages, [r["characters"] for r in results])
        bank_emb = None
        if bank["images"]:
            bank_emb = model.predict_crop_embeddings(
                bank["images"], [[[0, 0, im.shape[1], im.shape[0]]] for im in bank["images"]])
            bank_emb = torch.nn.functional.normalize(torch.cat(bank_emb, 0), dim=1)
    out = []
    for r, e in zip(results, embs):
        e = torch.nn.functional.normalize(e, dim=1) if len(e) else e
        page = {k: r[k] for k in ("panels", "texts", "characters", "tails", "character_cluster_labels",
                                  "is_essential_text", "character_names", "text_character_associations")}
        page["embeddings"] = [[round(float(x), 4) for x in row] for row in e.cpu().tolist()]
        if bank_emb is not None and len(e):
            sims = (e @ bank_emb.T).cpu().numpy()
            best = sims.argmax(1)
            page["bank_best"] = [[bank["names"][j], round(float(sims[i, j]), 4)] for i, j in enumerate(best)]
        else:
            page["bank_best"] = []
        out.append(page)
    return out


def main():
    job = json.load(open(sys.argv[1]))
    use_native_solver()
    device = pick_device(job.get("device", "auto"))
    try:
        pages = run(job, device)
    except Exception as e:  # some ops are flaky on MPS; CPU always works
        if device == "cpu":
            raise
        print(f"magi worker: {device} failed ({e!r}); retrying on cpu", file=sys.stderr)
        device = "cpu"
        pages = run(job, device)
    json.dump({"device": device, "pages": pages}, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
