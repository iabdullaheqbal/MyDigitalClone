import pytest
import requests

from clone.generation.llm_client import (
    GroqClient,
    GroqRateLimitError,
    GroqRequestError,
    RateLimiter,
)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleeps.append(duration)
        self.now += duration


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class FakeSession:
    def __init__(self, responses: list[object], clock: FakeTime | None = None) -> None:
        self.responses = iter(responses)
        self.clock = clock
        self.calls: list[dict[str, object]] = []
        self.started_at: list[float] = []

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.clock is not None:
            self.started_at.append(self.clock.clock())
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, FakeResponse)
        return response


def _client(session: FakeSession, clock: FakeTime, max_attempts: int = 3) -> GroqClient:
    limiter = RateLimiter(clock=clock.clock, sleep=clock.sleep)
    return GroqClient(
        "test-secret",
        "openai/gpt-oss-20b",
        session=session,
        rate_limiter=limiter,
        sleep=clock.sleep,
        max_attempts=max_attempts,
    )


def _success(content: object) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_client_retries_429_and_respects_request_spacing() -> None:
    clock = FakeTime()
    session = FakeSession(
        [FakeResponse(429, {}), _success("answer")],
        clock,
    )

    assert _client(session, clock).chat([{"role": "user", "content": "question"}]) == "answer"
    assert len(session.calls) == 2
    assert session.started_at[1] - session.started_at[0] >= 1.1
    assert session.calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["max_completion_tokens"] == 512


def test_client_does_not_retry_bad_request() -> None:
    clock = FakeTime()
    session = FakeSession([FakeResponse(400, {})], clock)

    with pytest.raises(GroqRequestError, match="status 400"):
        _client(session, clock).chat([{"role": "user", "content": "question"}])
    assert len(session.calls) == 1


def test_client_handles_list_content_and_hides_credentials_on_network_errors() -> None:
    clock = FakeTime()
    list_session = FakeSession([_success([{"type": "text", "text": "Hello "}, {"text": "world"}])])
    assert _client(list_session, clock).chat([]) == "Hello world"

    secret = "do-not-leak-this-key"
    failed_session = FakeSession([requests.RequestException(secret)])
    with pytest.raises(GroqRequestError) as error:
        _client(failed_session, clock, max_attempts=1).chat([])
    assert secret not in str(error.value)


def test_client_retries_server_errors_but_raises_rate_limit_type() -> None:
    clock = FakeTime()
    session = FakeSession([FakeResponse(503, {}), _success("recovered")], clock)
    assert _client(session, clock).chat([]) == "recovered"

    rate_limited = FakeSession([FakeResponse(429, {})] * 2)
    with pytest.raises(GroqRateLimitError):
        _client(rate_limited, clock, max_attempts=2).chat([])


def test_rate_limiter_rejects_intervals_below_safety_floor() -> None:
    with pytest.raises(ValueError, match="1.1 seconds"):
        RateLimiter(interval=1.0)
