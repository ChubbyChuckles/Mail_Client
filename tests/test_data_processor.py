# tests/test_data_processor.py
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_processor import colorize_value, verify_and_analyze_data

# Sample configuration values for testing
MIN_VOLUME_EUR = 10000
PRICE_INCREASE_THRESHOLD = 5.0
MAX_SLIPPAGE_BUY = 0.5
MAX_SLIPPAGE_SELL = -0.5
ALLOCATION_PER_TRADE = 0.1
PRICE_RANGE_PERCENT = 2.0
MIN_TOTAL_SCORE = 0.7


@pytest.fixture
def sample_ohlcv_data():
    """Fixture for sample OHLCV data."""
    current_time = datetime.utcnow()
    five_min_ago = current_time - timedelta(minutes=5)
    ten_min_ago = current_time - timedelta(minutes=10)
    data = {
        "timestamp": [
            five_min_ago,
            five_min_ago,
            ten_min_ago,
            five_min_ago,
            five_min_ago,
            ten_min_ago,
        ],
        "symbol": ["BTC/EUR", "ETH/EUR", "BTC/EUR", "ADA/EUR", "XRP/EUR", "ADA/EUR"],
        "open": [50000.0, 3000.0, 49000.0, 1.0, 0.5, 0.95],
        "close": [52500.0, 3150.0, 49500.0, 1.05, 0.52, 0.98],
        "volume": [10.0, 1000.0, 15.0, 50000.0, 20000.0, 60000.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def price_monitor_manager():
    """Fixture for a mock price monitor manager."""
    return MagicMock()


@pytest.fixture
def mock_logger():
    """Fixture for mocking the logger."""
    with patch("src.data_processor.logger") as mock:
        yield mock


@pytest.fixture
def mock_check_rate_limit():
    """Fixture for mocking check_rate_limit."""
    with patch("src.data_processor.check_rate_limit") as mock:
        yield mock


@pytest.fixture
def mock_calculate_order_book_metrics():
    """Fixture for mocking calculate_order_book_metrics."""
    with patch("src.data_processor.calculate_order_book_metrics") as mock:
        mock.side_effect = lambda market, **kwargs: {
            "market": market,
            "slippage_buy": 0.3,
            "slippage_sell": -0.2,
            "total_score": 0.8,
            "recommendation": "Strong Buy",
        }
        yield mock


@pytest.fixture
def mock_portfolio():
    """Fixture for mock portfolio."""
    return {"cash": 10000, "assets": {"BTC/EUR": 0.1, "ADA/EUR": 1000}}


@pytest.fixture
def mock_portfolio_lock():
    """Fixture for mock portfolio lock."""
    return MagicMock()


@pytest.fixture
def mock_low_volatility_assets():
    """Fixture for mock low_volatility_assets."""
    return set(["ADA/EUR"])


def test_empty_dataframe(mock_logger):
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    price_monitor_manager = MagicMock()
    result = verify_and_analyze_data(df, price_monitor_manager)

    assert result[0] == []
    assert result[1].empty
    assert result[2] == []
    mock_logger.warning.assert_called_with("Input DataFrame is empty.")


def test_old_data(mock_logger):
    """Test handling of data older than 10 minutes."""
    old_time = datetime.utcnow() - timedelta(minutes=15)
    df = pd.DataFrame(
        {
            "timestamp": [old_time],
            "symbol": ["BTC/EUR"],
            "open": [50000.0],
            "close": [51000.0],
            "volume": [10.0],
        }
    )
    price_monitor_manager = MagicMock()
    result = verify_and_analyze_data(df, price_monitor_manager)

    assert result[0] == []
    assert result[1].empty
    assert result[2] == []
    mock_logger.warning.assert_called_with(
        "Data contains no candles from within the last 10 minutes."
    )


def test_invalid_data_handling():
    """Test handling of invalid data (e.g., zero open price)."""
    df = pd.DataFrame(
        {
            "timestamp": [datetime.utcnow()],
            "symbol": ["BTC/EUR"],
            "open": [0.0],
            "close": [51000.0],
            "volume": [10.0],
        }
    )
    price_monitor_manager = MagicMock()
    _, percent_changes, _ = verify_and_analyze_data(df, price_monitor_manager)

    assert percent_changes.empty
