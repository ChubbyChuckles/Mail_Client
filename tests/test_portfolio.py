# tests/test_portfolio.py
import pytest
from src.state import portfolio
from src.portfolio import manage_portfolio
from src.price_monitor import PriceMonitorManager

def test_portfolio_initialization():
    assert portfolio['cash'] == 10000
    assert portfolio['assets'] == {}