"""Bounded in-memory cache for history-independent answer results."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace

from clone.types import AnswerResult, RetrievalMode


class AnswerCache:
    """Thread-safe LRU cache with a fixed time-to-live."""

    def __init__(
        self,
        max_entries: int = 200,
        ttl_seconds: float = 1_800.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1 or ttl_seconds <= 0:
            raise ValueError("Cache limits must be positive")
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str, int, str], tuple[float, AnswerResult]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    @staticmethod
    def key(
        query: str,
        mode: RetrievalMode,
        top_k: int,
        persona_version: str,
    ) -> tuple[str, str, int, str]:
        normalized = " ".join(query.casefold().split())
        return normalized, RetrievalMode(mode).value, top_k, persona_version

    def get(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        persona_version: str,
    ) -> AnswerResult | None:
        key = self.key(query, mode, top_k, persona_version)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, result = entry
            if self._clock() >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return replace(result, latency_ms=dict(result.latency_ms))

    def put(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int,
        persona_version: str,
        result: AnswerResult,
    ) -> None:
        key = self.key(query, mode, top_k, persona_version)
        with self._lock:
            self._entries[key] = (self._clock() + self.ttl_seconds, result)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
