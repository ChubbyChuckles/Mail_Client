# MAIL_CLIENT_TEST/tests/test_exchange.py
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest

from src.exchange import (check_rate_limit, fetch_klines, fetch_ticker_price,
                          fetch_trade_details, handle_ban_error)


@pytest.fixture
def mock_bitvavo(monkeypatch):
    """Mock the ccxt.bitvavo client."""
    mock_client = MagicMock()
    monkeypatch.setattr("MAIL_CLIENT_TEST.src.exchange.bitvavo", mock_client)
    return mock_client


def test_fetch_ticker_price_success(mock_bitvavo):
    """Test successful ticker price fetch."""
    mock_bitvavo.fetch_ticker.return_value = {"last": 50000.0}
    price = fetch_ticker_price("BTC/EUR")
    assert price == 50000.0
    mock_bitvavo.fetch_ticker.assert_called_once_with("BTC/EUR")


def test_fetch_ticker_price_invalid_response(mock_bitvavo, caplog):
    """Test handling of invalid ticker response."""
    mock_bitvavo.fetch_ticker.return_value = {}
    with caplog.at_level(logging.ERROR):
        price = fetch_ticker_price("BTC/EUR")
        assert price is None
        assert "Invalid ticker response" in caplog.text


def test_fetch_ticker_price_permission_denied(mock_bitvavo, caplog):
    """Test handling of PermissionDenied error."""
    mock_bitvavo.fetch_ticker.side_effect = ccxt.PermissionDenied("403: errorCode 105")
    with caplog.at_level(logging.WARNING):
        price = fetch_ticker_price("BTC/EUR")
        assert price is None
        assert "Permission denied" in caplog.text


def test_fetch_klines_success(mock_bitvavo):
    """Test successful OHLCV data fetch."""
    mock_bitvavo.fetch_ohlcv.return_value = [
        [1625097600000, 1000, 1100, 900, 1050, 100],
        [1625097660000, 1050, 1150, 950, 1100, 150],
    ]
    df = fetch_klines("BTC/EUR", timeframe="1m", limit=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
    ]
    assert df["symbol"].iloc[0] == "BTC/EUR"


def test_fetch_klines_empty_response(mock_bitvavo):
    """Test handling of empty OHLCV response."""
    mock_bitvavo.fetch_ohlcv.return_value = []
    df = fetch_klines("BTC/EUR")
    assert df.empty


def test_fetch_trade_details_success(mock_bitvavo):
    """Test successful trade details fetch."""
    mock_bitvavo.fetch_trades.return_value = [
        {"timestamp": 1625097600000, "amount": 1.0, "price": 50000.0}
    ]
    trade_count, largest_volume = fetch_trade_details(
        "BTC/EUR", datetime(2021, 6, 30), datetime(2021, 7, 1)
    )
    assert trade_count == 1
    assert largest_volume == 50000.0


def test_handle_ban_error_with_ban(mock_bitvavo, caplog):
    """Test handling of ban error."""
    error = ccxt.NetworkError("403: errorCode:105, The ban expires at 1625097600000")
    with caplog.at_level(logging.WARNING):
        ban_expiry = handle_ban_error(error)
        assert ban_expiry == 1625097600.0
        assert "API ban detected" in caplog.text


def test_check_rate_limit_near_limit(monkeypatch, caplog):
    """Test rate limit check when approaching limit."""
    monkeypatch.setattr("MAIL_CLIENT_TEST.src.exchange.weight_used", 900)
    monkeypatch.setattr("MAIL_CLIENT_TEST.src.exchange.last_reset_time", 1625097600)
    monkeypatch.setattr("time.time", lambda: 1625097600 + 30)
    with caplog.at_level(logging.INFO):
        check_rate_limit(100)
        assert "Approaching rate limit" in caplog.text
