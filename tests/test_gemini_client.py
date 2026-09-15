"""Resilience unit tests for Gemini key rotation + transient retries.

`execute_with_key_rotation` is the QA-suite name for
`common.gemini.call_with_key_rotation` (aliased in common/gemini.py).
`google.genai.Client` is always mocked — no live API calls.
"""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from google import genai

from common.gemini import (
    call_with_key_rotation,
    execute_with_key_rotation,
)


class FakeApiError(Exception):
    """Minimal stand-in for google.genai errors (code / status_code / message)."""

    def __init__(self, message: str, *, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = code


@pytest.fixture()
def two_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "KEY-1")
    monkeypatch.setenv("GEMINI_API_KEYS", "KEY-2")
    return ["KEY-1", "KEY-2"]


@pytest.fixture()
def single_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "ONLY-KEY")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    return ["ONLY-KEY"]


def _mock_genai_clients(mocker, per_key: dict):
    """Patch google.genai.Client so factory calls return per-key Mock clients."""
    mock_client_cls = mocker.patch("google.genai.Client")
    clients = {key: Mock(name=f"client[{key}]") for key in per_key}
    mock_client_cls.side_effect = lambda api_key: clients[api_key]
    return mock_client_cls, clients


def test_rotation_alias_matches_implementation():
    assert execute_with_key_rotation is call_with_key_rotation


def test_failover_to_key2_on_429_resource_exhausted(mocker, two_keys):
    """Key 1 raises 429 RESOURCE_EXHAUSTED -> request transparently uses key 2."""
    mock_client_cls, clients = _mock_genai_clients(
        mocker, {"KEY-1": None, "KEY-2": None}
    )

    def operation(client):
        if client is clients["KEY-1"]:
            raise FakeApiError("429 RESOURCE_EXHAUSTED: quota exceeded", code=429)
        return "success-from-key-2"

    result = execute_with_key_rotation(
        operation,
        lambda api_key: genai.Client(api_key=api_key),
        operation_name="test-quota-failover",
    )

    assert result == "success-from-key-2"
    assert mock_client_cls.call_count == 2
    mock_client_cls.assert_any_call(api_key="KEY-1")
    mock_client_cls.assert_any_call(api_key="KEY-2")


def test_invalid_key_error_also_fails_over(mocker, two_keys):
    """A dead/revoked first key (400 API_KEY_INVALID) also activates key 2."""
    mock_client_cls, clients = _mock_genai_clients(
        mocker, {"KEY-1": None, "KEY-2": None}
    )

    def operation(client):
        if client is clients["KEY-1"]:
            raise FakeApiError(
                "API key not valid. Please pass a valid API key.", code=400
            )
        return "success-from-key-2"

    result = execute_with_key_rotation(
        operation,
        lambda api_key: genai.Client(api_key=api_key),
        operation_name="test-invalid-key-failover",
    )

    assert result == "success-from-key-2"
    assert mock_client_cls.call_count == 2


def test_503_retries_with_exponential_backoff_then_succeeds(mocker, single_key):
    """503 Service Unavailable -> same-key retry with backoff -> 200 success."""
    mock_client_cls, clients = _mock_genai_clients(mocker, {"ONLY-KEY": None})
    sleep_calls: list[float] = []
    mocker.patch("common.gemini.time.sleep", side_effect=sleep_calls.append)

    attempts: list[str] = []

    def operation(client):
        attempts.append("attempt")
        if len(attempts) == 1:
            raise FakeApiError("503 Service Unavailable: overloaded", code=503)
        return "recovered-200"

    result = execute_with_key_rotation(
        operation,
        lambda api_key: genai.Client(api_key=api_key),
        operation_name="test-503-retry",
    )

    assert result == "recovered-200"
    assert len(attempts) == 2
    # Exponential backoff: 1.0s before the first retry.
    assert sleep_calls == [1.0]
    # Same key retried — no second client constructed.
    assert mock_client_cls.call_count == 1


def test_503_exhausts_retries_and_propagates(mocker, single_key):
    """Persistent 503s exhaust both retries (1s, 2s) and then raise."""
    _mock_genai_clients(mocker, {"ONLY-KEY": None})
    sleep_calls: list[float] = []
    mocker.patch("common.gemini.time.sleep", side_effect=sleep_calls.append)

    def operation(client):
        raise FakeApiError("503 Service Unavailable: overloaded", code=503)

    with pytest.raises(FakeApiError, match="503"):
        execute_with_key_rotation(
            operation,
            lambda api_key: genai.Client(api_key=api_key),
            operation_name="test-503-exhausted",
        )

    assert sleep_calls == [1.0, 2.0]


def test_non_retryable_error_propagates_without_rotation(mocker, two_keys):
    """Programming errors (e.g. ValueError) raise immediately on key 1."""
    mock_client_cls, _ = _mock_genai_clients(mocker, {"KEY-1": None, "KEY-2": None})

    def operation(client):
        raise ValueError("bad prompt shape — retrying with a new key cannot help")

    with pytest.raises(ValueError, match="bad prompt shape"):
        execute_with_key_rotation(
            operation,
            lambda api_key: genai.Client(api_key=api_key),
            operation_name="test-no-rotation",
        )

    assert mock_client_cls.call_count == 1
