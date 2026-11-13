"""Groq chat-completions client with token budgets and bounded fallbacks."""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Protocol

import requests

from clone.generation.rate_limit import TokenBudget, estimate_prompt_tokens


class GroqError(RuntimeError):
    """Base class for safe, user-presentable Groq failures."""


class GroqRateLimitError(GroqError):
    """The API or local token budget rejected a request."""

    def __init__(self, retry_after_seconds: float = 25.0) -> None:
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__("Groq rate limit was reached")


class GroqTimeoutError(GroqError):
    """A Groq request timed out."""


class GroqEmptyOutputError(GroqError):
    """Groq returned an empty completion."""


class GroqRequestError(GroqError):
    """The API request failed or returned an invalid response."""


class ChatClient(Protocol):
    """Interface used by answer generation and query rewriting."""

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> str: ...


class RateLimiter:
    """Serialize request starts with a minimum interval between them."""

    def __init__(
        self,
        interval: float = 1.1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isfinite(interval) or interval < 1.1:
            raise ValueError("API request interval must be at least 1.1 seconds")
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._next_request_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Wait until this request can start without violating the interval."""
        with self._lock:
            delay = self._next_request_at - self._clock()
            if delay > 0:
                self._sleep(delay)
            self._next_request_at = self._clock() + self.interval


def _header_float(headers: object, name: str) -> float | None:
    try:
        raw = headers.get(name)  # type: ignore[attr-defined]
    except AttributeError:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _retry_after_seconds(headers: object) -> float | None:
    value = _header_float(headers, "retry-after")
    if value is not None:
        return value
    try:
        raw = headers.get("retry-after")  # type: ignore[attr-defined]
        retry_at = parsedate_to_datetime(raw)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=UTC)
    return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())


def _reset_seconds(headers: object) -> float | None:
    raw = None
    try:
        raw = headers.get("x-ratelimit-reset-tokens")  # type: ignore[attr-defined]
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        pass
    total = 0.0
    for amount, unit in re.findall(r"([\d.]+)\s*([hms])", str(raw), re.I):
        total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit.casefold()]
    return total if total else None


class GroqClient:
    """Groq-compatible client that avoids logging prompt or credential contents."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
        models: tuple[str, ...] | None = None,
        timeout: float = 30.0,
        max_attempts: int = 2,
        session: requests.Session | None = None,
        rate_limiter: RateLimiter | None = None,
        token_budget: TokenBudget | None = None,
        tokens_per_minute: int = 6_000,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        selected_models = tuple(
            dict.fromkeys(item.strip() for item in (models or (model,)) if item.strip())
        )
        if not selected_models or not base_url.strip():
            raise ValueError("Groq model and base URL must not be empty")
        if not isfinite(timeout) or timeout <= 0 or max_attempts < 1:
            raise ValueError("Timeout and max_attempts must be positive")
        self._api_key = api_key.strip()
        self.models = selected_models
        self.model = selected_models[0]
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = timeout
        self.max_attempts = min(max_attempts, 2)
        self._session = session or requests.Session()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._token_budget = token_budget or TokenBudget(tokens_per_minute)
        self._sleep = sleep

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> str:
        """Call the first eligible model, retry 429 once, then try alternatives."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        url = f"{self.base_url}/chat/completions"
        request_headers = {"Authorization": f"Bearer {self._api_key}"}
        estimated = estimate_prompt_tokens(messages, max_tokens)
        last_retry_after = 0.0

        for model_index, model in enumerate(self.models):
            attempts = self.max_attempts if model_index == 0 else 1
            for attempt in range(attempts):
                reservation, retry_after = self._token_budget.reserve_estimate(model, estimated)
                if reservation is None:
                    last_retry_after = max(last_retry_after, retry_after)
                    break
                self._rate_limiter.wait()
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "temperature": temperature,
                    "reasoning_effort": "low",
                }
                try:
                    response = self._session.post(
                        url,
                        headers=request_headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                except requests.Timeout:
                    raise GroqTimeoutError("Groq request timed out") from None
                except requests.RequestException:
                    raise GroqRequestError("Groq request failed") from None

                status = response.status_code
                if status == 429:
                    retry_after = _retry_after_seconds(response.headers)
                    reset_after = _reset_seconds(response.headers)
                    retry_after = (
                        retry_after
                        if retry_after is not None
                        else reset_after
                        if reset_after is not None
                        else 25.0
                    )
                    remaining = _header_float(response.headers, "x-ratelimit-remaining-tokens")
                    self._token_budget.reconcile(
                        model,
                        estimated,
                        remaining_tokens=int(remaining) if remaining is not None else None,
                        reservation=reservation,
                    )
                    last_retry_after = max(last_retry_after if model_index else 0, retry_after)
                    if attempt + 1 < attempts and retry_after <= 3:
                        self._sleep(retry_after)
                        continue
                    break
                if 500 <= status < 600:
                    if attempt + 1 < attempts:
                        self._sleep(1.0)
                        continue
                    raise GroqRequestError(f"Groq server failed with status {status}")
                if not 200 <= status < 300:
                    raise GroqRequestError(f"Groq request failed with status {status}")

                payload_data = self._response_json(response)
                usage = payload_data.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
                remaining = _header_float(response.headers, "x-ratelimit-remaining-tokens")
                self._token_budget.reconcile(
                    model,
                    estimated,
                    prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                    completion_tokens=(
                        completion_tokens if isinstance(completion_tokens, int) else None
                    ),
                    remaining_tokens=int(remaining) if remaining is not None else None,
                    reservation=reservation,
                )
                return self._response_content(payload_data)

        raise GroqRateLimitError(last_retry_after or 25.0)

    @staticmethod
    def _response_json(response: requests.Response) -> dict[str, object]:
        try:
            payload = response.json()
        except ValueError:
            raise GroqRequestError("Groq returned an invalid response") from None
        if not isinstance(payload, dict):
            raise GroqRequestError("Groq returned an invalid response")
        return payload

    @staticmethod
    def _response_content(payload: dict[str, object]) -> str:
        try:
            choices = payload["choices"]
            if not isinstance(choices, list):
                raise TypeError
            first = choices[0]
            if not isinstance(first, dict):
                raise TypeError
            message = first["message"]
            if not isinstance(message, dict):
                raise TypeError
            content = message["content"]
        except (KeyError, IndexError, TypeError):
            raise GroqRequestError("Groq returned an invalid response") from None

        if isinstance(content, str):
            result = content.strip()
        elif isinstance(content, list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            result = "".join(text_parts).strip()
        else:
            result = ""
        if not result:
            raise GroqEmptyOutputError("Groq returned an empty response")
        return result
