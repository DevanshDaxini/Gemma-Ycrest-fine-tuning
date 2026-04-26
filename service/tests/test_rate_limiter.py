import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rate_limiter import RateLimiter


def test_allows_within_limit():
    rl = RateLimiter(requests_per_minute=5)
    for _ in range(5):
        assert rl.is_allowed("key-a") is True


def test_blocks_when_over_limit():
    rl = RateLimiter(requests_per_minute=3)
    for _ in range(3):
        rl.is_allowed("key-a")
    assert rl.is_allowed("key-a") is False


def test_different_keys_independent():
    rl = RateLimiter(requests_per_minute=2)
    rl.is_allowed("key-a")
    rl.is_allowed("key-a")
    # key-a exhausted
    assert rl.is_allowed("key-a") is False
    # key-b untouched
    assert rl.is_allowed("key-b") is True


def test_zero_rpm_always_allows():
    rl = RateLimiter(requests_per_minute=0)
    for _ in range(1000):
        assert rl.is_allowed("key-a") is True
