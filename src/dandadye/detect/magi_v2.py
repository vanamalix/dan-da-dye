"""Magi v2 (ragavsachdeva/magiv2) detector, run in a separate venv via subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from ..config import Config

WORKER = Path(__file__).with_name("worker_magi_v2.py")


class MagiV2Detector:
    name = "magi-v2"

    def __init__(self, python: Path, model_dir: Path, device: str = "auto", eta: float = 0.75,
                 hf_home: Path | None = None):
        self.python, self.model_dir, self.device, self.eta, self.hf_home = python, model_dir, device, eta, hf_home

    @classmethod
    def from_config(cls, cfg: Config, device: str = "auto", eta: float = 0.75) -> "MagiV2Detector":
        return cls(python=cfg.data_root / ".venv-magi" / "bin" / "python",
                   model_dir=cfg.data_root / "models" / "magiv2", device=device, eta=eta,
                   hf_home=cfg.data_root / "hf")

    def job(self, pages: list[Path], bank: list[tuple[str, Path]]) -> dict:
        return {"model_dir": str(self.model_dir), "device": self.device, "eta": self.eta,
                "pages": [str(p) for p in pages],
                "bank": {"images": [str(p) for _, p in bank], "names": [n for n, _ in bank]}}

    @staticmethod
    def normalise(page: dict) -> dict:
        """Split Magi's texts into dialogue (essential) and other (SFX etc.)."""
        essential = page.get("is_essential_text") or [True] * len(page["texts"])
        page["texts_other"] = [t for t, e in zip(page["texts"], essential) if not e]
        page["texts"] = [t for t, e in zip(page["texts"], essential) if e]
        return page

    def detect_chapter(self, pages: list[Path], bank: list[tuple[str, Path]]) -> list[dict]:
        if not self.python.exists():
            raise SystemExit(f"Magi venv not found: {self.python} (see README: 'Magi v2 setup')")
        with tempfile.TemporaryDirectory() as td:
            job_path, out_path = Path(td) / "job.json", Path(td) / "out.json"
            job_path.write_text(json.dumps(self.job(pages, bank)))
            env = {"HF_HOME": str(self.hf_home)} if self.hf_home else {}
            subprocess.run([str(self.python), "-I", str(WORKER), str(job_path), str(out_path)],
                           check=True, env={**os.environ, **env, "HF_HUB_OFFLINE": "1"})
            result = json.loads(out_path.read_text())
        return [self.normalise(p) for p in result["pages"]]
