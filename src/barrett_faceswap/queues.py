"""A small queue that keeps the newest item and drops what went stale.

Live capture is faster than a swap. Holding every frame would add lag, so
the capture side keeps at most ``maxsize`` items and discards the oldest.
"""

from __future__ import annotations

import threading
from collections import deque


class LatestQueue:
    """Thread-safe bounded queue. ``put_latest`` drops the oldest item when full."""

    def __init__(self, maxsize: int = 1) -> None:
        self.maxsize = max(1, int(maxsize))
        self._items: deque = deque()
        self._cond = threading.Condition()
        self.dropped = 0

    def put_latest(self, item: object) -> None:
        with self._cond:
            while len(self._items) >= self.maxsize:
                self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cond.notify()

    def get(self, timeout: float) -> object | None:
        """Return the next item, or ``None`` when ``timeout`` passes."""

        with self._cond:
            if not self._items:
                self._cond.wait(timeout=timeout)
            if not self._items:
                return None
            return self._items.popleft()

    def empty(self) -> bool:
        with self._cond:
            return not self._items
