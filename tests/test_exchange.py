import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest
from tenacity import RetryError

from src.exchange import (RATE_LIMIT_WEIGHT, ban_expiry_time, check_rate_limit,
                          fetch_klines, fetch_ticker_price,
                          fetch_trade_details, handle_ban_error, is_banned,
                          last_reset_time, wait_until_ban_lifted, weight_used)


# Fixture for mocking ccxt.bitvavo
@pytest.fixture
def mock_bitvavo():
    with patch("src.exchange.bitvavo", new=MagicMock()) as mock:
        yield mock


# Test handle_ban_error
def test_handle_ban_error():
    """Test handling of ban error with 403 and errorCode 105."""
    exception = Exception(
        '403 Forbidden: {"errorCode":105, "error":"The ban expires at 1697059200000"}'
    )
    with patch("src.exchange.logger") as mock_logger:
        ban_expiry = handle_ban_error(exception)
        assert ban_expiry == 1697059200.0
        assert is_banned is False
        assert ban_expiry_time == 0


def test_handle_ban_error_no_ban():
    """Test handle_ban_error with non-ban exception."""
    exception = Exception("Generic error")
    with patch("src.exchange.logger") as mock_logger:
        result = handle_ban_error(exception)
        assert result is None


# Test wait_until_ban_lifted
def test_wait_until_ban_lifted():
    """Test waiting until ban is lifted."""
    future_time = time.time() + 2
    with patch("src.exchange.logger") as mock_logger, patch("time.sleep") as mock_sleep:
        wait_until_ban_lifted(future_time)
        mock_logger.info.assert_any_call(
            f"Waiting 2.00 seconds until ban lifts at {datetime.utcfromtimestamp(future_time)}"
        )
        mock_sleep.assert_called_with(pytest.approx(2, abs=0.1))
        assert is_banned is False


# Test check_rate_limit
def test_check_rate_limit_under_limit():
    """Test rate limit check when under limit."""
    global weight_used, last_reset_time
    weight_used = 0
    last_reset_time = time.time()
    with patch("src.exchange.logger") as mock_logger:
        check_rate_limit(100)
        assert weight_used == 0
        mock_logger.debug.assert_called()


# Test fetch_ticker_price
def test_fetch_ticker_price_success(mock_bitvavo):
    """Test successful ticker price fetch."""
    mock_bitvavo.fetch_ticker.return_value = {"last": 50000.0}
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"):
        price = fetch_ticker_price("BTC/EUR")
        assert price == 50000.0
        mock_bitvavo.fetch_ticker.assert_called_with("BTC/EUR")


def test_fetch_ticker_price_invalid_price(mock_bitvavo):
    """Test fetch_ticker_price with invalid (zero) price."""
    mock_bitvavo.fetch_ticker.return_value = {"last": 0}
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.exchange.logger"
    ) as mock_logger:
        price = fetch_ticker_price("BTC/EUR")
        assert price is None


def test_fetch_ticker_price_ban(mock_bitvavo):
    """Test fetch_ticker_price with ban error."""
    mock_bitvavo.fetch_ticker.side_effect = ccxt.PermissionDenied(
        '403 Forbidden: {"errorCode":105, "error":"The ban expires at 1697059200000"}'
    )
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.exchange.wait_until_ban_lifted"
    ):
        price = fetch_ticker_price("BTC/EUR")
        assert price is None


def test_fetch_ticker_price_fallback(mock_bitvavo):
    """Test fetch_ticker_price fallback to last known price."""
    mock_bitvavo.fetch_ticker.side_effect = Exception("Network error")
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.price_monitor.PriceMonitorManager"
    ) as mock_pmm:
        mock_pmm.return_value.last_prices.get.return_value = 49000.0
        price = fetch_ticker_price("BTC/EUR")
        assert price == 49000.0


# Test fetch_klines
def test_fetch_klines_success(mock_bitvavo):
    """Test successful OHLCV data fetch."""
    mock_bitvavo.fetch_ohlcv.return_value = [
        [1697059200000, 50000, 51000, 49000, 50500, 10]
    ]
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"):
        df = fetch_klines("BTC/EUR", "1m", 10)
        assert not df.empty
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
        assert df["close"].iloc[0] == 50500.0


def test_fetch_klines_invalid_prices(mock_bitvavo):
    """Test fetch_klines with invalid (negative) prices."""
    mock_bitvavo.fetch_ohlcv.return_value = [
        [1697059200000, -50000, 51000, 49000, 50500, 10]
    ]
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.exchange.logger"
    ) as mock_logger:
        df = fetch_klines("BTC/EUR", "1m", 10)
        assert df.empty
        mock_logger.error.assert_called_with(
            "Invalid prices (zero or negative) in OHLCV data for BTC/EUR. Discarding data."
        )


def test_fetch_klines_ban(mock_bitvavo):
    """Test fetch_klines with ban error."""
    mock_bitvavo.fetch_ohlcv.side_effect = ccxt.PermissionDenied(
        '403 Forbidden: {"errorCode":105, "error":"The ban expires at 1697059200000"}'
    )
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.exchange.wait_until_ban_lifted"
    ):
        df = fetch_klines("BTC/EUR")
        assert df.empty


# Test fetch_trade_details
def test_fetch_trade_details_success(mock_bitvavo):
    """Test successful trade details fetch."""
    mock_bitvavo.fetch_trades.return_value = [
        {"timestamp": 1697059200000, "amount": 1, "price": 50000},
        {"timestamp": 1697059201000, "amount": 2, "price": 51000},
    ]
    start_time = datetime.utcfromtimestamp(1697059200)
    end_time = datetime.utcfromtimestamp(1697059202)
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"):
        trade_count, largest_volume = fetch_trade_details(
            "BTC/EUR", start_time, end_time
        )
        assert trade_count == 0
        assert largest_volume == 102000.0  # 2 * 51000


def test_fetch_trade_details_no_trades(mock_bitvavo):
    """Test fetch_trade_details with no trades."""
    mock_bitvavo.fetch_trades.return_value = []
    start_time = datetime.utcfromtimestamp(1697059200)
    end_time = datetime.utcfromtimestamp(1697059202)
    with patch("src.exchange.check_rate_limit"), patch("src.exchange.semaphore"), patch(
        "src.exchange.logger"
    ) as mock_logger:
        trade_count, largest_volume = fetch_trade_details(
            "BTC/EUR", start_time, end_time
        )
        assert trade_count == 0
        assert largest_volume == 0.0
        mock_logger.warning.assert_called_with(
            f"No trades returned for BTC/EUR from {start_time} to {end_time}"
        )
