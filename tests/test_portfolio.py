# MAIL_CLIENT_TEST/tests/test_portfolio.py
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
from datetime import datetime, timedelta
import json
import logging

from src.portfolio import manage_portfolio, sell_asset, save_portfolio
from src.state import portfolio, portfolio_lock
from src.config import logger, PORTFOLIO_VALUE, ALLOCATION_PER_TRADE

@pytest.fixture
def mock_price_monitor():
    """Mock PriceMonitorManager."""
    return MagicMock()

@pytest.fixture
def sample_portfolio():
    """Set up a sample portfolio."""
    with portfolio_lock:
        portfolio["cash"] = PORTFOLIO_VALUE
        portfolio["assets"] = {
            "BTC/EUR": {
                "quantity": 0.1,
                "purchase_price": 50000.0,
                "purchase_time": datetime.utcnow() - timedelta(minutes=10),
                "highest_price": 51000.0,
                "current_price": 50500.0,
                "profit_target": 0.05,
                "original_profit_target": 0.05,
                "sell_price": 52500.0
            }
        }
    yield
    with portfolio_lock:
        portfolio["cash"] = PORTFOLIO_VALUE
        portfolio["assets"] = {}

def test_sell_asset_success(sample_portfolio, mock_price_monitor, tmp_path):
    """Test successful asset sale."""
    with portfolio_lock:
        asset = portfolio["assets"]["BTC/EUR"]
    finished_trades = []
    finished_trade = sell_asset(
        "BTC/EUR", asset, 51000.0, portfolio, portfolio_lock, finished_trades,
        "Test sale", mock_price_monitor
    )
    assert finished_trade["Symbol"] == "BTC/EUR"
    assert float(finished_trade["Profit/Loss"]) > 0
    with portfolio_lock:
        assert "BTC/EUR" not in portfolio["assets"]
        assert portfolio["cash"] > PORTFOLIO_VALUE

def test_sell_asset_invalid_price(sample_portfolio, mock_price_monitor, caplog):
    """Test sale with invalid price."""
    with portfolio_lock:
        asset = portfolio["assets"]["BTC/EUR"]
    finished_trades = []
    with caplog.at_level(logging.ERROR):
        finished_trade = sell_asset(
            "BTC/EUR", asset, 0.0, portfolio, portfolio_lock, finished_trades,
            "Test sale", mock_price_monitor
        )
        assert finished_trade is None
        assert "Invalid sell price" in caplog.text

def test_manage_portfolio_buy(mock_price_monitor, tmp_path):
    """Test portfolio management with a buy decision."""
    above_threshold_data = [
        {"symbol": "ETH/EUR", "close_price": 2000.0, "percent_change": 5.0, "volume_eur": 15000.0}
    ]
    percent_changes = pd.DataFrame({
        "symbol": ["ETH/EUR"],
        "close_price": [2000.0],
        "percent_change": [5.0],
        "volume_eur": [15000.0]
    })
    with portfolio_lock:
        portfolio["cash"] = PORTFOLIO_VALUE
        portfolio["assets"] = {}
    with patch("MAIL_CLIENT_TEST.src.portfolio.fetch_trade_details", return_value=(10, 1000.0)):
        manage_portfolio(above_threshold_data, percent_changes, mock_price_monitor)
    with portfolio_lock:
        assert "ETH/EUR" in portfolio["assets"]
        assert portfolio["cash"] < PORTFOLIO_VALUE
        assert portfolio["assets"]["ETH/EUR"]["quantity"] == (PORTFOLIO_VALUE * ALLOCATION_PER_TRADE * (1 - 0.0015)) / 2000.0

def test_save_portfolio(tmp_path, sample_portfolio):
    """Test saving portfolio to file."""
    portfolio_file = tmp_path / "portfolio.json"
    with patch("MAIL_CLIENT_TEST.src.config.PORTFOLIO_FILE", str(portfolio_file)):
        save_portfolio()
    assert portfolio_file.exists()
    with open(portfolio_file, "r") as f:
        saved_data = json.load(f)
    assert saved_data["cash"] == PORTFOLIO_VALUE
    assert "BTC/EUR" in saved_data["assets"]