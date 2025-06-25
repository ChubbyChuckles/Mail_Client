# trading_bot/src/__init__.py
"""
Trading Bot Package

This package contains a modular cryptocurrency trading bot that interacts with the Bitvavo API.
Modules include configuration, exchange interactions, portfolio management, data processing,
price monitoring, notifications, storage, utilities, and state management.
"""

from .config import (API_KEY, API_SECRET, LOOP_INTERVAL_SECONDS,
                     PORTFOLIO_VALUE, logger)
from .data_processor import verify_and_analyze_data
from .exchange import (bitvavo, fetch_klines, fetch_ticker_price,
                       fetch_trade_details)
from .main import main
from .portfolio import manage_portfolio, save_portfolio
from .price_monitor import PriceMonitorManager
from .state import (low_volatility_assets, negative_momentum_counts, portfolio,
                    portfolio_lock)
from .storage import save_to_local
from .utils import append_to_finished_trades_csv  # Add this line
from .utils import append_to_order_book_metrics_csv  # Add new function
from .utils import (append_to_buy_trades_csv, calculate_dynamic_ema_period,
                    calculate_ema)

__version__ = "0.1.0"  # Version of the trading bot package
__all__ = [
    # Config
    "logger",
    "API_KEY",
    "API_SECRET",
    "PORTFOLIO_VALUE",
    "LOOP_INTERVAL_SECONDS",
    # Exchange
    "bitvavo",
    "fetch_klines",
    "fetch_ticker_price",
    "fetch_trade_details",
    # Portfolio
    "portfolio",
    "portfolio_lock",
    "manage_portfolio",
    "save_portfolio",
    # Data Processor
    "verify_and_analyze_data",
    # Price Monitor
    "PriceMonitorManager",
    # Notifications
    "send_telegram_message",
    "send_trade_notification",
    "run_async",
    # Storage
    "save_to_local",
    # Utils
    "calculate_ema",
    "calculate_dynamic_ema_period",
    "append_to_buy_trades_csv",
    "append_to_finished_trades_csv",  # Add this line
    "append_to_order_book_metrics_csv"
    # Main
    "main",
]
