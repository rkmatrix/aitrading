from __future__ import annotations
import sqlite3
from pathlib import Path

class MetaDB:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, key: str, default: str | None = None) -> str | None:
        with sqlite3.connect(self.path) as con:
            cur = con.execute("SELECT v FROM kv WHERE k=?", (key,))
            row = cur.fetchone()
            return row[0] if row else default

    def set(self, key: str, value: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO kv(k,v) VALUES(?,?)", (key, value))
