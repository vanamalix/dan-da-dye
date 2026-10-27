"""Detector interface. Swap Magi v2 for another model by implementing this."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Detector(Protocol):
    """Detect one chapter at a time (some models use cross-page context).

    ``detect_chapter`` returns one dict per page, in order, with:
      panels:      [[x1, y1, x2, y2], ...]  in reading order, original pixels
      texts:       dialogue/lettering boxes (kept clean by inkfix)
      texts_other: other text, e.g. sound effects (may be coloured)
      characters:  [[x1, y1, x2, y2], ...]
      character_names: name per character from the bank, "Other" if unknown,
                       or None if the detector cannot name characters
      embeddings:  optional unit vectors per character for chapter clustering
    ``bank`` is a list of (name, image_path); several images may share a name.
    """

    name: str

    def detect_chapter(self, pages: list[Path], bank: list[tuple[str, Path]]) -> list[dict]: ...
