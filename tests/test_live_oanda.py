"""
Tests for the live trial's OANDA client (maestro/live/oanda.py): it must never
reach anything but a practice account, and must never resend an order.
"""
import pytest
import requests

from maestro.live.oanda import PRACTICE_URL, PracticeClient, PracticeOnlyError


class FakeSession:
    def __init__(self, post_response=None, fail_post=False):
        self.headers, self.posts, self.post_response, self.fail_post = {}, [], post_response, fail_post

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if self.fail_post:
            raise requests.ConnectionError("timeout")
        return FakeResponse(self.post_response)


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


@pytest.fixture(autouse=True)
def practice_env(monkeypatch):
    monkeypatch.setenv("OANDA_ENVIRONMENT", "practice")
    monkeypatch.delenv("OANDA_BASE_URL", raising=False)


def _client(session=None, **kw):
    return PracticeClient(api_key="k", account_id="101-001-1-001", session=session or FakeSession(), **kw)


@pytest.mark.parametrize("url", ["https://api-fxtrade.oanda.com", "https://api-fxpractice.oanda.com.evil.io"])
def test_refuses_any_server_but_practice(url):
    with pytest.raises(PracticeOnlyError):
        _client(base_url=url)


def test_refuses_a_live_environment_setting(monkeypatch):
    monkeypatch.setenv("OANDA_ENVIRONMENT", "live")
    with pytest.raises(PracticeOnlyError):
        _client()


def test_refuses_a_live_url_from_the_environment(monkeypatch):
    monkeypatch.setenv("OANDA_BASE_URL", "https://api-fxtrade.oanda.com")
    with pytest.raises(PracticeOnlyError):
        _client()


def test_market_order_posts_once_to_the_practice_server():
    fill = {"orderFillTransaction": {"units": "-10000", "price": "1.10005", "time": "2026-10-01T10:00:01Z",
                                     "fullPrice": {"bids": [{"price": "1.10005"}], "asks": [{"price": "1.10015"}]}}}
    session = FakeSession(post_response=fill)
    result = _client(session).market_order("EUR_USD", -10000, tag="maestro-1")
    assert len(session.posts) == 1 and session.posts[0][0].startswith(PRACTICE_URL)
    order = session.posts[0][1]["order"]
    assert order["units"] == "-10000" and order["timeInForce"] == "FOK"
    assert order["clientExtensions"]["id"] == "maestro-1"
    assert result.units == -10000 and result.bid == 1.10005 and result.ask == 1.10015


def test_a_failed_order_is_not_resent():
    session = FakeSession(fail_post=True)
    with pytest.raises(requests.ConnectionError):
        _client(session).market_order("EUR_USD", 10000, tag="maestro-2")
    assert len(session.posts) == 1


def test_an_unfilled_order_raises():
    session = FakeSession(post_response={"orderCancelTransaction": {"reason": "MARKET_HALTED"}})
    with pytest.raises(RuntimeError, match="MARKET_HALTED"):
        _client(session).market_order("EUR_USD", 10000, tag="maestro-3")
