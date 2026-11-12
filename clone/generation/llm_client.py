"""Groq chat-completions client with bounded retries and rate limiting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from math import isfinite
from typing import Protocol

import requests


class GroqError(RuntimeError):
    """Base class for safe, user-presentable Groq failures."""


class GroqRateLimitError(GroqError):
    """The API rate limit remained exceeded after retries."""


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


class GroqClient:
    """Small Groq-compatible client that never logs request contents."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        *,
        timeout: float = 30.0,
        max_attempts: int = 3,
        session: requests.Session | None = None,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")
        if not model.strip() or not base_url.strip():
            raise ValueError("Groq model and base URL must not be empty")
        if not isfinite(timeout) or timeout <= 0 or max_attempts < 1:
            raise ValueError("Timeout and max_attempts must be positive")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self._session = session or requests.Session()
        self._rate_limiter = rate_limiter or RateLimiter()
        self._sleep = sleep

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        """Request one chat completion, retrying rate limits and server errors."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(self.max_attempts):
            self._rate_limiter.wait()
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                if attempt + 1 == self.max_attempts:
                    raise GroqRequestError("Groq request failed after retries") from None
                self._sleep(2**attempt)
                continue

            status = response.status_code
            if status == 429 or 500 <= status < 600:
                if attempt + 1 == self.max_attempts:
                    if status == 429:
                        raise GroqRateLimitError("Groq rate limit persisted after retries")
                    raise GroqRequestError(f"Groq server failed with status {status}")
                self._sleep(2**attempt)
                continue
            if not 200 <= status < 300:
                raise GroqRequestError(f"Groq request failed with status {status}")
            return self._response_content(response)

        raise GroqRequestError("Groq request exhausted retries")

    @staticmethod
    def _response_content(response: requests.Response) -> str:
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
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
            raise GroqRequestError("Groq returned an empty response")
        return result
