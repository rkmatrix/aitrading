# ai/online/alpha_online_buffer.py
from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional


logger = logging.getLogger(__name__)


class AlphaOnlineBuffer:
    """
    AlphaOnlineBuffer

    - Maintains a rolling in-memory buffer of recent transitions.
    - Streams new rows from an append-only JSONL replay file.
    - Each row in the replay should be a dict with at least:
        {
            "obs": [...],
            "action": ...,
            "reward": float,
            "done": bool,
            "next_obs": [...],
            ...
        }
    """

    def __init__(
        self,
        *,
        path: str,
        maxlen: int = 10000,
    ) -> None:
        self.path = Path(path)
        self.maxlen = maxlen

        self._buffer: Deque[Dict] = deque(maxlen=maxlen)
        self._file_pos: int = 0      # byte offset in file
        self.total_seen: int = 0     # total lines seen (including evicted ones)

        # ensure directory exists
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        # initial load if file exists
        if self.path.exists():
            logger.info("📥 AlphaOnlineBuffer: loading existing replay from %s", self.path)
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("⚠️ AlphaOnlineBuffer: skipping malformed line: %r", line[:200])
                        continue
                    self._buffer.append(row)
                    self.total_seen += 1
                self._file_pos = f.tell()
            logger.info(
                "📥 AlphaOnlineBuffer: loaded %d rows (total_seen=%d)",
                len(self._buffer),
                self.total_seen,
            )
        else:
            logger.info("📄 AlphaOnlineBuffer: replay file does not exist yet: %s", self.path)

    def __len__(self) -> int:
        return len(self._buffer)

    def get_recent(self, n: int) -> List[Dict]:
        """
        Return up to the last `n` transitions (may be fewer if buffer smaller).
        """
        if n <= 0:
            return []
        n = min(n, len(self._buffer))
        # deque doesn't support slicing, so we convert to list
        return list(self._buffer)[-n:]

    def push(self, row: Dict) -> None:
        """
        Manual push (in case you want to use it directly).
        """
        self._buffer.append(row)
        self.total_seen += 1

        # also append to disk
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            self._file_pos = f.tell()

    def refresh_from_disk(self) -> int:
        """
        Read newly appended lines from JSONL replay file and add to buffer.

        Returns:
            count_new (int): number of new rows added.
        """
        if not self.path.exists():
            # no file yet, nothing to read
            return 0

        count_new = 0
        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self._file_pos)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("⚠️ AlphaOnlineBuffer: skipping malformed appended line")
                    continue
                self._buffer.append(row)
                self.total_seen += 1
                count_new += 1
            self._file_pos = f.tell()

        if count_new > 0:
            logger.info(
                "📥 AlphaOnlineBuffer: loaded %d new rows (len(buffer)=%d, total_seen=%d)",
                count_new,
                len(self._buffer),
                self.total_seen,
            )
        return count_new
