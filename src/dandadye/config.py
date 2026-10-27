"""Configuration: data root, work directory and the series preset.

The data root holds everything that must never be committed (pages, models,
official art, outputs). It comes from ``--data`` or the ``DANDADYE_DATA``
environment variable. The preset (``presets/<name>/preset.toml``) describes one
series: where its pages are and which characters it has.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_PRESETS = Path(__file__).resolve().parents[2] / "presets"


@dataclass
class Character:
    name: str
    art: Path | None = None
    palette_key: str | None = None
    main_cast: bool = False


@dataclass
class Preset:
    series: str
    title: str
    source_dir: str = "work/src"
    volume_dir_pattern: str = r"^v\d+$"
    volume_chapter_regex: str = r"c(\d+(?:\.\d+)?)"
    volume_page_regex: str = r"p(\d+)(?:-p?(\d+))?"
    palette: str | None = None
    characters: list[Character] = field(default_factory=list)
    colourise: dict = field(default_factory=dict)

    @property
    def main_cast(self) -> list[Character]:
        return [c for c in self.characters if c.main_cast]

    def character(self, name: str) -> Character | None:
        return next((c for c in self.characters if c.name == name), None)


@dataclass
class Config:
    data_root: Path
    preset: Preset

    @property
    def work(self) -> Path:
        return self.data_root / "work"

    @property
    def source_dir(self) -> Path:
        return self.data_root / self.preset.source_dir

    @property
    def state_path(self) -> Path:
        return self.work / "state.sqlite"

    def stage_dir(self, stage: str) -> Path:
        d = self.work / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def resolve(self, rel: str | Path | None) -> Path | None:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.data_root / p


def find_preset(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.is_file():
        return p
    if p.is_dir() and (p / "preset.toml").is_file():
        return p / "preset.toml"
    candidate = REPO_PRESETS / name_or_path / "preset.toml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"preset not found: {name_or_path}")


def load_preset(name_or_path: str, data_root: Path | None = None) -> Preset:
    raw = tomllib.loads(find_preset(name_or_path).read_text())
    chars = []
    for c in raw.pop("characters", []):
        art = c.get("art")
        if art and data_root is not None and not Path(art).is_absolute():
            art = data_root / art
        chars.append(Character(name=c["name"], art=Path(art) if art else None,
                               palette_key=c.get("palette_key"), main_cast=c.get("main_cast", False)))
    known = {k: raw[k] for k in raw if k in Preset.__dataclass_fields__}
    return Preset(characters=chars, **known)


def load_config(data_root: str | Path | None = None, preset: str = "dandadan") -> Config:
    root = data_root or os.environ.get("DANDADYE_DATA")
    if not root:
        raise SystemExit("set DANDADYE_DATA or pass --data /path/to/data-root")
    root = Path(root).expanduser()
    return Config(data_root=root, preset=load_preset(preset, root))
