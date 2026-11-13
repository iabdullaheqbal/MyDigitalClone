import pytest
import requests

from clone.generation.llm_client import (
    GroqClient,
    GroqEmptyOutputError,
    GroqRateLimitError,
    GroqRequestError,
    RateLimiter,
)
from clone.generation.rate_limit import TokenBudget


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
    def __init__(
        self,
        status_code: int,
        payload: object,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

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


def _client(
    session: FakeSession,
    clock: FakeTime,
    max_attempts: int = 3,
    *,
    models: tuple[str, ...] | None = None,
    token_budget: TokenBudget | None = None,
) -> GroqClient:
    limiter = RateLimiter(clock=clock.clock, sleep=clock.sleep)
    return GroqClient(
        "test-secret",
        "openai/gpt-oss-20b",
        models=models,
        session=session,
        rate_limiter=limiter,
        token_budget=token_budget,
        sleep=clock.sleep,
        max_attempts=max_attempts,
    )


def _success(content: object) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


def test_client_retries_429_and_respects_request_spacing() -> None:
    clock = FakeTime()
    session = FakeSession(
        [
            FakeResponse(429, {}, {"retry-after": "1"}),
            _success("answer"),
        ],
        clock,
    )

    assert _client(session, clock).chat([{"role": "user", "content": "question"}]) == "answer"
    assert len(session.calls) == 2
    assert session.started_at[1] - session.started_at[0] >= 1.1
    assert session.calls[0]["url"] == "https://api.groq.com/openai/v1/chat/completions"
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["max_completion_tokens"] == 700
    assert payload["reasoning_effort"] == "low"


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


def test_token_budget_rejects_overage_then_allows_expired_usage() -> None:
    clock = FakeTime()
    budget = TokenBudget(
        tokens_per_minute=100,
        max_wait_seconds=0,
        clock=clock.clock,
        sleep=clock.sleep,
    )

    assert budget.reserve("model-a", 80) == (True, 0)
    admitted, retry_after = budget.reserve("model-a", 30)
    assert not admitted
    assert retry_after == 60

    clock.sleep(60)
    assert budget.reserve("model-a", 30) == (True, 0)


def test_client_token_budget_prevents_api_call_and_reads_usage_headers() -> None:
    clock = FakeTime()
    budget = TokenBudget(
        tokens_per_minute=100,
        max_wait_seconds=0,
        clock=clock.clock,
        sleep=clock.sleep,
    )
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": "answer"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
                },
                {"x-ratelimit-remaining-tokens": "83"},
            )
        ]
    )
    client = _client(session, clock, token_budget=budget)

    assert client.chat([], max_tokens=50) == "answer"
    assert budget.used_tokens("openai/gpt-oss-20b") == 17

    small_budget = TokenBudget(
        tokens_per_minute=40,
        max_wait_seconds=0,
        clock=clock.clock,
        sleep=clock.sleep,
    )
    limited_session = FakeSession([])
    limited_client = _client(limited_session, clock, token_budget=small_budget)
    with pytest.raises(GroqRateLimitError):
        limited_client.chat([], max_tokens=50)
    assert not limited_session.calls


def test_client_uses_next_model_after_rate_limit_and_parses_empty_output() -> None:
    clock = FakeTime()
    session = FakeSession(
        [
            FakeResponse(429, {}, {"retry-after": "25"}),
            _success("fallback answer"),
        ]
    )
    client = _client(
        session,
        clock,
        models=("model-a", "model-b"),
    )

    assert client.chat([]) == "fallback answer"
    assert [call["json"]["model"] for call in session.calls] == ["model-a", "model-b"]

    empty_session = FakeSession([_success("")])
    with pytest.raises(GroqEmptyOutputError):
        _client(empty_session, clock).chat([])
