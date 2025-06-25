# MAIL_CLIENT_TEST/tests/test_data_processor.py
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import logging

from src.data_processor import verify_and_analyze_data
from src.config import logger, MIN_VOLUME_EUR, PRICE_INCREASE_THRESHOLD

@pytest.fixture
def mock_price_monitor():
    """Mock PriceMonitorManager."""
    return MagicMock()

@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    now = datetime.utcnow()
    return pd.DataFrame({
        "timestamp": [now - timedelta(minutes=1), now],
        "open": [1000.0, 1050.0],
        "high": [1100.0, 1150.0],
        "low": [900.0, 1000.0],
        "close": [1050.0, 1100.0],
        "volume": [100.0, 150.0],
        "symbol": ["BTC/EUR", "BTC/EUR"]
    })

def test_verify_and_analyze_data_valid(sample_data, mock_price_monitor, caplog):
    """Test data analysis with valid input."""
    with caplog.at_level(logging.INFO):
        above_threshold, percent_changes = verify_and_analyze_data(sample_data, mock_price_monitor)
        assert len(above_threshold) == 1
        assert above_threshold[0]["symbol"] == "BTC/EUR"
        assert not percent_changes.empty
        assert "Coins with price increase" in caplog.text

def test_verify_and_analyze_data_empty(caplog):
    """Test handling of empty DataFrame."""
    with caplog.at_level(logging.WARNING):
        above_threshold, percent_changes = verify_and_analyze_data(pd.DataFrame(), MagicMock())
        assert above_threshold == []
        assert percent_changes.empty
        assert "Input DataFrame is empty" in caplog.text

def test_verify_and_analyze_data_old_data(caplog):
    """Test handling of outdated data."""
    old_data = pd.DataFrame({
        "timestamp": [datetime.utcnow() - timedelta(minutes=15)],
        "open": [1000.0],
        "close": [1050.0],
        "volume": [100.0],
        "symbol": ["BTC/EUR"]
    })
    with caplog.at_level(logging.WARNING):
        above_threshold, percent_changes = verify_and_analyze_data(old_data, MagicMock())
        assert above_threshold == []
        assert percent_changes.empty
        assert "no candles from within the last 10 minutes" in caplog.text