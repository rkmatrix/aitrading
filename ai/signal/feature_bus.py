from collections import defaultdict, deque
from typing import Dict, Deque, Tuple
from datetime import datetime
from .schemas import Feature

class FeatureBus:
    """In-memory, thread-safe-ish (single process) feature bus."""
    def __init__(self, maxlen: int = 256):
        self._store: Dict[Tuple[str,str], Deque[Feature]] = defaultdict(lambda: deque(maxlen=maxlen))

    def publish(self, symbol: str, feature: Feature):
        self._store[(symbol, feature.name)].append(feature)

    def latest(self, symbol: str, name: str) -> Feature | None:
        q = self._store.get((symbol, name))
        if not q or not len(q):
            return None
        return q[-1]

    def snapshot(self, symbol: str, names: list[str]) -> dict[str, Feature | None]:
        return {n: self.latest(symbol, n) for n in names}
