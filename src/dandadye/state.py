"""Per-page pipeline state in a small SQLite database.

One row per page, keyed ``"<chapter>/<NNN>"``. ``data`` is a JSON object that
stages merge their results into (``data["ingest"]``, ``data["detect"]`` ...),
so every stage can tell what is already done and resume.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .util import chapter_sort_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    key TEXT PRIMARY KEY,
    chapter TEXT NOT NULL,
    idx INTEGER NOT NULL,
    data TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS pages_chapter ON pages(chapter);
CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL DEFAULT '{}'
);
"""


def page_key(chapter: str, idx: int) -> str:
    return f"{chapter}/{idx:03d}"


class State:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    # pages -----------------------------------------------------------------
    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT data FROM pages WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, chapter: str, idx: int, data: dict) -> None:
        self.db.execute(
            "INSERT INTO pages(key, chapter, idx, data) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET data=excluded.data",
            (page_key(chapter, idx), chapter, idx, json.dumps(data)),
        )
        self.db.commit()

    def update(self, key: str, section: str, value: dict) -> None:
        data = self.get(key)
        if data is None:
            raise KeyError(key)
        data[section] = value
        self.db.execute("UPDATE pages SET data=? WHERE key=?", (json.dumps(data), key))
        self.db.commit()

    def delete_chapter_pages(self, chapter: str, keep_idx: set[int]) -> None:
        """Drop rows for pages that no longer exist after a re-ingest."""
        rows = self.db.execute("SELECT key, idx FROM pages WHERE chapter=?", (chapter,)).fetchall()
        for key, idx in rows:
            if idx not in keep_idx:
                self.db.execute("DELETE FROM pages WHERE key=?", (key,))
        self.db.commit()

    def pages(self, chapters: list[str] | None = None) -> list[tuple[str, str, int, dict]]:
        """Return (key, chapter, idx, data) in reading order."""
        rows = self.db.execute("SELECT key, chapter, idx, data FROM pages").fetchall()
        out = [(k, c, i, json.loads(d)) for k, c, i, d in rows if chapters is None or c in chapters]
        out.sort(key=lambda r: (chapter_sort_key(r[1]), r[2]))
        return out

    # chapters --------------------------------------------------------------
    def put_chapter(self, chapter: str, data: dict) -> None:
        self.db.execute(
            "INSERT INTO chapters(id, data) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            (chapter, json.dumps(data)),
        )
        self.db.commit()

    def chapter(self, chapter: str) -> dict | None:
        row = self.db.execute("SELECT data FROM chapters WHERE id=?", (chapter,)).fetchone()
        return json.loads(row[0]) if row else None

    def chapters(self) -> list[str]:
        ids = [r[0] for r in self.db.execute("SELECT id FROM chapters").fetchall()]
        return sorted(ids, key=chapter_sort_key)
