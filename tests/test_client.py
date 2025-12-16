from types import SimpleNamespace
from typing import Any, Dict
from unittest import mock

import pytest
import requests

from client import classify


class DummyResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any] | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def raise_for_status(self):
        if 400 <= self.status_code:
            http_error = requests.HTTPError(response=SimpleNamespace(text=self.text))
            raise http_error

    def json(self):
        return self._payload


def test_classify_success():
    dummy_payload = {"emotion": "happy", "confidence": 0.9}
    mock_resp = DummyResponse(status_code=200, payload=dummy_payload)
    with mock.patch("requests.post", return_value=mock_resp) as mocked_post:
        result = classify("http://example.com/classify", "hi")

    mocked_post.assert_called_once()
    assert result == dummy_payload


def test_classify_http_error():
    mock_resp = DummyResponse(status_code=503, text="Service unavailable")
    with mock.patch("requests.post", return_value=mock_resp):
        with pytest.raises(requests.HTTPError):
            classify("http://example.com/classify", "hi")

