from collections import deque
from threading import Lock
import time


class RateLimiter:
    """Sliding-window rate limiter — counts requests per API key in the last 60 s."""

    def __init__(self, requests_per_minute: int):
        self._rpm = requests_per_minute
        self._windows: dict[str, deque] = {}
        self._lock = Lock()

    def is_allowed(self, key: str) -> bool:
        if self._rpm <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            dq = self._windows.setdefault(key, deque())
            while dq and dq[0] < cutoff:
                dq.popleft()
            if len(dq) >= self._rpm:
                return False
            dq.append(now)
            return True
