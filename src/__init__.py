# trading_bot/src/__init__.py
"""
Trading Bot Package

This package contains a modular cryptocurrency trading bot that interacts with the Bitvavo API.
Modules include configuration, exchange interactions, portfolio management, data processing,
price monitoring, notifications, storage, utilities, and state management.
"""

from .config import logger, API_KEY, API_SECRET, PORTFOLIO_VALUE, LOOP_INTERVAL_SECONDS
from .exchange import bitvavo, fetch_klines, fetch_ticker_price, fetch_trade_details
from .state import (
    portfolio,
    portfolio_lock,
    low_volatility_assets,
    negative_momentum_counts,
    last_sheets_write,
    last_sheets_load,
)
from .portfolio import manage_portfolio, save_portfolio
from .data_processor import verify_and_analyze_data
from .price_monitor import PriceMonitorManager
from .notifications import send_telegram_message, send_trade_notification, run_async
from .storage import save_to_local, write_to_google_sheets, load_active_assets
from .utils import calculate_ema, calculate_dynamic_ema_period, append_to_buy_trades_csv
from .main import main

__version__ = "0.1.0"  # Version of the trading bot package
__all__ = [
    # Config
    'logger',
    'API_KEY',
    'API_SECRET',
    'PORTFOLIO_VALUE',
    'LOOP_INTERVAL_SECONDS',
    # Exchange
    'bitvavo',
    'fetch_klines',
    'fetch_ticker_price',
    'fetch_trade_details',
    # Portfolio
    'portfolio',
    'portfolio_lock',
    'manage_portfolio',
    'price_monitor_manager',
    # Data Processor
    'verify_and_analyze_data',
    # Price Monitor
    'PriceMonitorManager',
    # Notifications
    'send_telegram_message',
    'send_trade_notification',
    'run_async',
    # Storage
    'save_to_local',
    'save_portfolio',
    'write_to_google_sheets',
    'load_active_assets',
    # Utils
    'calculate_ema',
    'calculate_dynamic_ema_period',
    'append_to_buy_trades_csv',
    # Main
    'main',
]