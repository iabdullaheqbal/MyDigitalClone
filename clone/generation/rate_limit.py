"""Thread-safe rolling token budgets for Groq model requests."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenReservation:
    """Reference to one accounted request in the rolling budget."""

    model: str
    entry: list[float]


class TokenBudget:
    """Reserve and reconcile a per-model rolling 60-second token allowance."""

    def __init__(
        self,
        tokens_per_minute: int = 6_000,
        *,
        window_seconds: float = 60.0,
        max_wait_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if tokens_per_minute < 1 or window_seconds <= 0 or max_wait_seconds < 0:
            raise ValueError("Token budget limits must be positive")
        self.tokens_per_minute = tokens_per_minute
        self.window_seconds = window_seconds
        self.max_wait_seconds = max_wait_seconds
        self._clock = clock
        self._sleep = sleep
        self._usage: dict[str, deque[list[float]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def reserve(self, model: str, tokens: int) -> tuple[bool, float]:
        """Reserve estimated tokens, waiting at most the configured duration."""
        reservation, retry_after = self.reserve_estimate(model, tokens)
        return reservation is not None, retry_after

    def reserve_estimate(self, model: str, tokens: int) -> tuple[TokenReservation | None, float]:
        """Return a reservation handle or the time until enough budget expires."""
        if tokens < 1:
            raise ValueError("Estimated token count must be positive")
        if tokens > self.tokens_per_minute:
            return None, self.window_seconds
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                usage = self._usage[model]
                while usage and now - usage[0][0] >= self.window_seconds:
                    usage.popleft()
                used = sum(int(entry[1]) for entry in usage)
                if used + tokens <= self.tokens_per_minute:
                    entry = [now, float(tokens)]
                    usage.append(entry)
                    return TokenReservation(model, entry), 0.0
                retry_after = max(0.0, usage[0][0] + self.window_seconds - now)
                delay = min(retry_after, self.max_wait_seconds - waited)
                if delay <= 0:
                    return None, retry_after
                self._sleep(delay)
                waited += delay

    def reconcile(
        self,
        model: str,
        reserved_tokens: int,
        *,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        remaining_tokens: int | None = None,
        reservation: TokenReservation | None = None,
    ) -> None:
        """Replace the latest reservation with provider usage when available."""
        actual = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else reserved_tokens
        )
        with self._lock:
            usage = self._usage[model]
            entry = reservation.entry if reservation and reservation.model == model else (
                usage[-1] if usage else None
            )
            if entry is not None:
                entry[1] = float(actual)
            if remaining_tokens is not None and remaining_tokens >= 0:
                current_used = sum(int(item[1]) for item in usage)
                total_provider_used = self.tokens_per_minute - remaining_tokens
                unexplained_used = max(0, total_provider_used - current_used)
                if entry is not None and unexplained_used:
                    entry[1] += float(unexplained_used)

    def used_tokens(self, model: str) -> int:
        """Return the unexpired accounted token total for tests and diagnostics."""
        with self._lock:
            now = self._clock()
            usage = self._usage[model]
            while usage and now - usage[0][0] >= self.window_seconds:
                usage.popleft()
            return sum(int(entry[1]) for entry in usage)


def estimate_prompt_tokens(messages: list[dict[str, str]], max_tokens: int) -> int:
    """Estimate prompt plus completion tokens conservatively from characters."""
    characters = sum(len(message.get("content", "")) for message in messages)
    return max(1, (characters + 3) // 4 + max_tokens)
