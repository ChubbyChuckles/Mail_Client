import json
import os
from datetime import datetime, timedelta
from threading import Lock
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.portfolio import (manage_portfolio, save_portfolio, sell_asset,
                           sell_most_profitable_asset)


# Mock configuration values
@pytest.fixture
def config():
    return {
        "ADJUSTED_PROFIT_TARGET": 0.015,
        "ALLOCATION_PER_TRADE": 0.1,
        "ASSET_THRESHOLD": 5,
        "BUY_FEE": 0.002,
        "CAT_LOSS_THRESHOLD": -0.05,
        "FINISHED_TRADES_CSV": "finished_trades.csv",
        "MAX_ACTIVE_ASSETS": 10,
        "MIN_TOTAL_SCORE": 0.7,
        "MIN_HOLDING_MINUTES": 15,
        "MOMENTUM_CONFIRM_MINUTES": 5,
        "MOMENTUM_THRESHOLD": 0.0,
        "PORTFOLIO_FILE": "portfolio.json",
        "PORTFOLIO_VALUE": 10000.0,
        "PROFIT_TARGET": 0.03,
        "PROFIT_TARGET_MULTIPLIER": 3.0,
        "SELL_FEE": 0.002,
        "TIME_STOP_MINUTES": 60,
        "TRAILING_STOP_FACTOR": 0.05,
        "TRAILING_STOP_FACTOR_EARLY": 0.03,
        "MAX_SLIPPAGE_BUY": 0.01,
        "logger": MagicMock(),
    }


@pytest.fixture
def portfolio_data():
    return {
        "cash": 5000.0,
        "assets": {
            "BTC/EUR": {
                "quantity": 0.1,
                "purchase_price": 30000.0,
                "purchase_time": datetime.utcnow() - timedelta(minutes=20),
                "highest_price": 31000.0,
                "current_price": 30500.0,
                "profit_target": 0.03,
                "sell_price": 30900.0,
            }
        },
    }


@pytest.fixture
def portfolio_lock():
    return Lock()


@pytest.fixture
def percent_changes():
    return pd.DataFrame(
        {
            "symbol": ["BTC/EUR", "ETH/EUR"],
            "close_price": [30500.0, 2000.0],
            "percent_change": [0.01, 0.02],
            "high": [31000.0, 2050.0],
            "low": [30000.0, 1950.0],
            "close": [30500.0, 2000.0],
        }
    )


@pytest.fixture
def price_monitor_manager():
    return MagicMock()


@pytest.fixture
def above_threshold_data():
    return [
        {"symbol": "ETH/EUR", "close_price": 2000.0},
        {"symbol": "XRP/EUR", "close_price": 0.5},
    ]


@pytest.fixture
def order_book_metrics_list():
    return [
        {
            "market": "ETH-EUR",
            "total_score": 0.8,
            "slippage_buy": 0.005,
            "bought": False,
        },
        {
            "market": "XRP-EUR",
            "total_score": 0.6,
            "slippage_buy": 0.015,
            "bought": False,
        },
    ]


def test_sell_asset_invalid_price(
    config, portfolio_data, portfolio_lock, price_monitor_manager
):
    finished_trades = []
    symbol = "BTC/EUR"
    asset = portfolio_data["assets"][symbol]
    with patch.dict("src.portfolio", portfolio_data, clear=True):
        result = sell_asset(
            symbol,
            asset,
            0.0,
            portfolio_data,
            portfolio_lock,
            finished_trades,
            "Invalid price",
            price_monitor_manager,
        )
    assert result is None


def test_sell_asset_none_price_monitor(config, portfolio_data, portfolio_lock):
    finished_trades = []
    symbol = "BTC/EUR"
    asset = portfolio_data["assets"][symbol]
    current_price = 31000.0
    with patch.dict("src.portfolio", portfolio_data, clear=True):
        result = sell_asset(
            symbol,
            asset,
            current_price,
            portfolio_data,
            portfolio_lock,
            finished_trades,
            "Profit target reached",
            None,
        )
    assert result is not None


def test_manage_portfolio_low_score(
    config,
    portfolio_data,
    above_threshold_data,
    percent_changes,
    price_monitor_manager,
    order_book_metrics_list,
):
    order_book_metrics_list[0]["total_score"] = 0.5  # Below MIN_TOTAL_SCORE
    portfolio_data["assets"] = {}
    with patch.dict("src.portfolio", portfolio_data, clear=True):
        manage_portfolio(
            above_threshold_data,
            percent_changes,
            price_monitor_manager,
            order_book_metrics_list,
        )
    assert "ETH/EUR" not in portfolio_data["assets"]
