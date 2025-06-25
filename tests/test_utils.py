# MAIL_CLIENT_TEST/tests/test_utils.py
import pytest
import pandas as pd
from unittest.mock import mock_open, patch
import csv
import logging

from src.utils import (
    calculate_ema, calculate_dynamic_ema_period, append_to_buy_trades_csv, append_to_finished_trades_csv
)
from src.config import logger

def test_calculate_ema_valid():
    """Test EMA calculation with valid input."""
    prices = [1000, 1010, 1020, 1030, 1040]
    ema = calculate_ema(prices, 3)
    assert isinstance(ema, float)
    assert 1020 < ema < 1040

def test_calculate_ema_insufficient_data(caplog):
    """Test EMA with insufficient data."""
    prices = [1000, 1010]
    with caplog.at_level(logging.WARNING):
        ema = calculate_ema(prices, 3)
        assert ema is None
        assert "Insufficient data for EMA calculation" in caplog.text

def test_calculate_dynamic_ema_period():
    """Test dynamic EMA period calculation."""
    period = calculate_dynamic_ema_period(30, 60, 15, 12)
    assert period == 8  # Base 5 + 3 for holding time

def test_append_to_buy_trades_csv(tmp_path):
    """Test appending buy trade to CSV."""
    csv_file = tmp_path / "buy_trades.csv"
    trade_data = {
        "Symbol": "BTC/EUR",
        "Buy Quantity": "0.1",
        "Buy Price": "50000.0",
        "Buy Time": "2021-06-30",
        "Buy Fee": "7.5"
    }
    with patch("MAIL_CLIENT_TEST.src.config.BUY_TRADES_CSV", str(csv_file)):
        append_to_buy_trades_csv(trade_data)
    assert csv_file.exists()
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["Symbol"] == "BTC/EUR"