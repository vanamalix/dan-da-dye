"""Small shared helpers: sorting, chapter selection and colour-space maths."""

from __future__ import annotations

import re

import numpy as np

_NUM = re.compile(r"(\d+)")


def natural_key(s: str) -> list:
    """Sort key that orders "p2" before "p10"."""
    return [int(t) if t.isdigit() else t.lower() for t in _NUM.split(s)]


def chapter_number(chapter_id: str) -> float | None:
    try:
        return float(chapter_id)
    except ValueError:
        return None


def chapter_sort_key(chapter_id: str) -> tuple:
    n = chapter_number(chapter_id)
    return (0, n, "") if n is not None else (1, 0.0, chapter_id)


def select_chapters(ids: list[str], start: float | None = None, end: float | None = None,
                    only: list[str] | None = None) -> list[str]:
    """Filter chapter ids by an inclusive numeric range and/or an explicit list."""
    out = []
    for cid in sorted(ids, key=chapter_sort_key):
        n = chapter_number(cid)
        if only and cid not in only:
            continue
        if start is not None and (n is None or n < start):
            continue
        if end is not None and (n is None or n > end):
            continue
        out.append(cid)
    return out


def parse_index_ranges(spec: str | None) -> set[int] | None:
    """"1-3,7" -> {1, 2, 3, 7}; None -> None (all pages)."""
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


# --- sRGB <-> CIELAB (D65), float arrays ---------------------------------------
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_MI = np.linalg.inv(_M)
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 or [0,1] float RGB (H,W,3) -> LAB with L in [0,100]."""
    c = rgb.astype(np.float64)
    if rgb.dtype == np.uint8:
        c /= 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """LAB -> uint8 RGB (clipped)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = np.clip(xyz @ _MI.T, 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
