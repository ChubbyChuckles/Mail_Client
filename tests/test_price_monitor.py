# MAIL_CLIENT_TEST/tests/test_price_monitor.py
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.price_monitor import PriceMonitorManager
from src.state import portfolio, portfolio_lock
from src.config import logger

@pytest.fixture
def price_monitor():
    """Create a PriceMonitorManager instance."""
    return PriceMonitorManager()

@pytest.fixture
def sample_portfolio():
    """Set up a sample portfolio."""
    with portfolio_lock:
        portfolio["assets"] = {
            "BTC/EUR": {
                "quantity": 0.1,
                "purchase_price": 50000.0,
                "purchase_time": datetime.utcnow() - timedelta(minutes=10),
                "highest_price": 51000.0,
                "current_price": 50500.0,
                "profit_target": 0.05,
                "sell_price": 52500.0
            }
        }
    yield
    with portfolio_lock:
        portfolio["assets"] = {}

def test_price_monitor_start(price_monitor, sample_portfolio):
    """Test starting a price monitoring thread."""
    candles_df = pd.DataFrame()
    with patch.object(price_monitor, "handle_ticker") as mock_handle:
        price_monitor.start("BTC/EUR", portfolio, portfolio_lock, candles_df)
        assert "BTC/EUR" in price_monitor.running
        assert price_monitor.running["BTC/EUR"]
        mock_handle.assert_called_once()

def test_price_monitor_stop(price_monitor):
    """Test stopping a price monitoring thread."""
    price_monitor.running["BTC/EUR"] = True
    price_monitor.threads["BTC/EUR"] = MagicMock()
    price_monitor.stop("BTC/EUR")
    assert "BTC/EUR" not in price_monitor.running
    assert "BTC/EUR" not in price_monitor.threads

def test_evaluate_candle_sell_signal(price_monitor, sample_portfolio):
    """Test candle evaluation triggering a sell."""
    candle = {"close": 60000.0}
    prices_df = pd.DataFrame({
        "symbol": ["BTC/EUR"],
        "open": [50000.0],
        "high": [60000.0],
        "low": [50000.0],
        "close": [60000.0]
    })
    with patch("MAIL_CLIENT_TEST.src.price_monitor.sell_asset") as mock_sell:
        price_monitor.evaluate_candle(candle, "BTC/EUR", portfolio, portfolio_lock, [], prices_df)
        mock_sell.assert_called_once()