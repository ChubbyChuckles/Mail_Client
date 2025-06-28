# trading_bot/src/portfolio.py
import glob
import json
import os
import time
import requests
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tempfile
import shutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from . import config
from .config import logger
from .exchange import fetch_ticker_price, fetch_trade_details
from .state import low_volatility_assets, negative_momentum_counts, portfolio, portfolio_lock
from .utils import append_to_buy_trades_csv, append_to_finished_trades_csv, append_to_order_book_metrics_csv, calculate_dynamic_ema_period, calculate_ema
from .bitvavo_order_metrics import calculate_order_book_metrics, fetch_order_book_with_retry
from .telegram_notifications import TelegramNotifier
from .data_processor import preprocess_market_data
import asyncio

portfolio_values = []  # Store portfolio value history

telegram_notifier = TelegramNotifier(
    bot_token=config.config.TELEGRAM_BOT_TOKEN,
    chat_id=config.config.TELEGRAM_CHAT_ID
)
asyncio.run_coroutine_threadsafe(telegram_notifier.start(), asyncio.get_event_loop())

class ErrorHandler:
    """Handles errors by logging and sending notifications."""
    def __init__(self, telegram_notifier):
        self.telegram_notifier = telegram_notifier

    def handle(self, subject, message, exception=None):
        logger.error(f"{subject}: {message}", exc_info=exception)
        self.telegram_notifier.notify_error(subject, message)

error_handler = ErrorHandler(telegram_notifier)

class StrategyConfig:
    """Configuration for trading strategy rules."""
    def __init__(self):
        # Buy conditions
        self.buy_conditions = {
            "not_in_portfolio": {"enabled": True},
            "sufficient_cash": {"enabled": True, "allocation_per_trade": config.config.ALLOCATION_PER_TRADE, "portfolio_value": config.config.PORTFOLIO_VALUE},
            "below_max_assets": {"enabled": True, "max_active_assets": config.config.MAX_ACTIVE_ASSETS},
            "not_low_volatility": {"enabled": True},
            "min_score": {"enabled": True, "min_total_score": config.config.MIN_TOTAL_SCORE},
            "max_buy_slippage": {"enabled": True, "max_slippage_buy": config.config.MAX_SLIPPAGE_BUY},
            "rsi_oversold": {"enabled": True, "threshold": config.config.RSI_OVERSOLD_THRESHOLD, "period": config.config.RSI_OVERSOLD_PERIOD},
            "min_volume": {"enabled": True, "min_trade_volume_eur": config.config.MIN_VOLUME_EUR},
            "bollinger_lower": {"enabled": True, "period": config.config.BOLLINGER_LOWER_PERIOD, "std_dev": config.config.BOLLINGER_LOWER_STD_DEV},
            "macd_bullish": {"enabled": False, "fast_period": config.config.MACD_BULLISH_FAST_PERIOD, "slow_period": config.config.MACD_BULLISH_SLOW_PERIOD, "signal_period": config.config.MACD_BULLISH_SIGNAL_PERIOD}
        }
        # Sell conditions
        self.sell_conditions = {
            "catastrophic_loss": {"enabled": True, "threshold": config.config.CAT_LOSS_THRESHOLD, "atr_multiplier": config.config.CAT_LOSS_ATR_MULTIPLIER},
            "time_stop": {"enabled": True, "time_stop_minutes": config.config.TIME_STOP_MINUTES},
            "multiplied_profit": {"enabled": True, "multiplier": config.config.PROFIT_TARGET_MULTIPLIER},
            "regular_sell": {
                "enabled": True,
                "min_holding_minutes": config.config.MIN_HOLDING_MINUTES,
                "trailing_stop_factor": config.config.TRAILING_STOP_FACTOR,
                "trailing_stop_factor_early": config.config.TRAILING_STOP_FACTOR_EARLY,
                "momentum_confirm_minutes": config.config.MOMENTUM_CONFIRM_MINUTES
            },
            "rsi_overbought": {"enabled": True, "threshold": config.config.RSI_OVERBOUGHT_THRESHOLD, "period": config.config.RSI_OVERBOUGHT_PERIOD},
            "volume_drop": {"enabled": True, "drop_threshold": config.config.VOLUME_DROP_THRESHOLD},
            "bollinger_upper": {"enabled": True, "period": config.config.BOLLINGER_UPPER_PERIOD, "std_dev": config.config.BOLLINGER_UPPER_STD_DEV},
            "macd_bearish": {"enabled": False, "fast_period": config.config.MACD_BEARISH_FAST_PERIOD, "slow_period": config.config.MACD_BEARISH_SLOW_PERIOD, "signal_period": config.config.MACD_BEARISH_SIGNAL_PERIOD}
        }
        # Grid trading parameters
        self.grid_enabled = False
        # self.grid_enabled = config.config.GRID_ENABLED
        self.grid_range = config.config.GRID_RANGE
        self.grid_step = config.config.GRID_STEP
        self.grid_order_size = config.config.ALLOCATION_PER_TRADE / 5
        self.grid_bb_period = config.config.GRID_BB_PERIOD
        self.grid_bb_std = config.config.GRID_BB_STD_DEV
        # DCA parameters
        self.dca_enabled = config.config.DCA_ENABLED
        self.dca_drop_threshold = config.config.DCA_DROP_THRESHOLD
        self.dca_max_allocation = config.config.DCA_ALLOCATION * config.config.PORTFOLIO_VALUE
        # Volatility-adjusted sizing
        self.volatility_sizing_enabled = config.config.VOLATILITY_SIZING_ENABLED
        self.atr_period = config.config.VOLATILITY_ATR_PERIOD
        self.max_allocation_multiplier = config.config.VOLATILITY_MAX_ALLOCATION_MULTIPLIER
        self.min_allocation_multiplier = config.config.VOLATILITY_MIN_ALLOCATION_MULTIPLIER
        self.max_position_size = config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
        # General parameters
        self.profit_target = config.config.PROFIT_TARGET
        self.adjusted_profit_target = config.config.ADJUSTED_PROFIT_TARGET
        self.momentum_threshold = config.config.MOMENTUM_THRESHOLD
        self.buy_fee = config.config.BUY_FEE
        self.sell_fee = config.config.SELL_FEE
        self.max_slippage_sell = config.config.MAX_SLIPPAGE_SELL
        self.asset_threshold = config.config.ASSET_THRESHOLD
        self.max_drawdown = config.config.MAX_DRAWDOWN

class TradingStrategy:
    """Encapsulates trading strategy logic with configurable conditions."""
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.peak_portfolio_value = max([v["portfolio_value"] for v in portfolio_values], default=self.config.buy_conditions["sufficient_cash"]["portfolio_value"])
        self.grid_orders = {}

    def calculate_rsi(self, symbol, market_data, period=14):
        if "close_price" not in market_data.columns:
            error_handler.handle("Market Data Error", f"No 'close_price' column in market_data for {symbol}")
            return None
        closes = market_data[market_data["symbol"] == symbol]["close_price"].values
        if len(closes) < period:
            return None
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0).sum() / period
        loss = np.where(delta < 0, -delta, 0).sum() / period
        rs = gain / loss if loss != 0 else float("inf")
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(self, symbol, market_data, period=20, std_dev=2.0):
        if "close_price" not in market_data.columns:
            error_handler.handle("Market Data Error", f"No 'close_price' column in market_data for {symbol}")
            return None, None, None
        closes = market_data[market_data["symbol"] == symbol]["close_price"].tail(period)
        if len(closes) < period:
            return None, None, None
        mean = closes.mean()
        std = closes.std()
        upper_band = mean + std_dev * std
        lower_band = mean - std_dev * std
        return mean, upper_band, lower_band

    def calculate_macd(self, symbol, market_data, fast_period=12, slow_period=26, signal_period=9):
        if "close_price" not in market_data.columns:
            error_handler.handle("Market Data Error", f"No 'close_price' column in market_data for {symbol}")
            return None, None
        closes = market_data[market_data["symbol"] == symbol]["close_price"].tail(max(fast_period, slow_period, signal_period))
        if len(closes) < max(fast_period, slow_period):
            return None, None
        ema_fast = pd.Series(closes).ewm(span=fast_period, adjust=False).mean().iloc[-1]
        ema_slow = pd.Series(closes).ewm(span=slow_period, adjust=False).mean().iloc[-1]
        macd_line = ema_fast - ema_slow
        if len(closes) < signal_period:
            return macd_line, None
        macd_series = pd.Series(closes).ewm(span=fast_period, adjust=False).mean() - pd.Series(closes).ewm(span=slow_period, adjust=False).mean()
        signal_line = macd_series.ewm(span=signal_period, adjust=False).mean().iloc[-1]
        return macd_line, signal_line

    def calculate_atr(self, symbol, market_data, period=14):
        if not all(col in market_data.columns for col in ["high", "low", "close_price"]):
            logger.warning(f"Missing 'high' or 'low' columns for {symbol}. Skipping ATR calculation.")
            return None
        symbol_data = market_data[market_data["symbol"] == symbol].tail(period)
        if len(symbol_data) < period:
            logger.warning(f"Insufficient data for ATR calculation for {symbol}: {len(symbol_data)} < {period}")
            return None
        high_low = symbol_data["high"] - symbol_data["low"]
        high_close = abs(symbol_data["high"] - symbol_data["close_price"].shift(1))
        low_close = abs(symbol_data["low"] - symbol_data["close_price"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.mean()

    def is_sideways_market(self, symbol, market_data):
        if not self.config.grid_enabled:
            return False
        if "close_price" not in market_data.columns:
            error_handler.handle("Market Data Error", f"No 'close_price' column in market_data for {symbol}")
            return False
        closes = market_data[market_data["symbol"] == symbol]["close_price"].tail(self.config.grid_bb_period)
        if len(closes) < self.config.grid_bb_period:
            return False
        mean = closes.mean()
        std = closes.std()
        upper_band = mean + self.config.grid_bb_std * std
        lower_band = mean - self.config.grid_bb_std * std
        current_price = closes.iloc[-1]
        return lower_band <= current_price <= upper_band

    def place_grid_orders(self, symbol, current_price, portfolio, market_data):
        orders = []
        base_price = current_price
        for i in range(-5, 6):
            price_offset = i * self.config.grid_step * base_price
            order_price = base_price + price_offset
            allocation = min(self.config.grid_order_size * self.config.buy_conditions["sufficient_cash"]["portfolio_value"], self.config.max_position_size)
            quantity = (allocation / order_price) if order_price > 0 else 0
            if i < 0:
                orders.append({"price": order_price, "type": "buy", "quantity": quantity})
            elif i > 0:
                orders.append({"price": order_price, "type": "sell", "quantity": quantity})
        self.grid_orders[symbol] = orders
        logger.info(f"Placed grid orders for {symbol}: {len(orders)} orders")
        return orders

    def check_grid_orders(self, symbol, current_price, portfolio, portfolio_lock, finished_trades, price_monitor_manager):
        if symbol not in self.grid_orders:
            return []
        executed_trades = []
        orders = self.grid_orders[symbol]
        for order in orders[:]:
            if order["type"] == "buy" and current_price <= order["price"] and portfolio["cash"] >= order["quantity"] * order["price"]:
                trade = self.execute_grid_buy(symbol, order["price"], order["quantity"], portfolio, portfolio_lock, price_monitor_manager)
                if trade:
                    executed_trades.append(trade)
                    orders.remove(order)
            elif order["type"] == "sell" and symbol in portfolio.get("assets", {}) and current_price >= order["price"]:
                trade = sell_asset(symbol, portfolio["assets"][symbol], order["price"], portfolio, portfolio_lock, finished_trades, "Grid sell", price_monitor_manager, sell_slippage=0.0)
                if trade:
                    executed_trades.append(trade)
                    orders.remove(order)
        self.grid_orders[symbol] = orders
        return executed_trades

    def execute_grid_buy(self, symbol, price, quantity, portfolio, portfolio_lock, price_monitor_manager):
        allocation = quantity * price
        buy_fee = allocation * self.config.buy_fee
        net_allocation = allocation - buy_fee
        if net_allocation <= 0:
            return None
        try:
            if not portfolio_lock.acquire(timeout=5):
                error_handler.handle("Portfolio Lock Failure", f"Timeout acquiring portfolio lock for grid buy {symbol}")
                return None
            try:
                buy_trade_data = {
                    "Symbol": symbol,
                    "Buy Quantity": f"{quantity:.10f}",
                    "Buy Price": f"{price:.10f}",
                    "Buy Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "Buy Fee": f"{buy_fee:.2f}",
                    "Buy Slippage": "0.00%",
                    "Allocation": f"{allocation:.2f}",
                    "Actual Cost": f"{quantity * price:.2f}",
                    "Trade Count": 0,
                    "Largest Trade Volume EUR": "0.00"
                }
                append_to_buy_trades_csv(buy_trade_data)
                telegram_notifier.notify_buy_trade(buy_trade_data)
                portfolio["assets"][symbol] = {
                    "quantity": quantity,
                    "purchase_price": price,
                    "purchase_time": datetime.utcnow(),
                    "buy_slippage": 0.0,
                    "buy_fee": buy_fee,
                    "highest_price": price,
                    "current_price": price,
                    "profit_target": self.config.profit_target,
                    "original_profit_target": self.config.profit_target,
                    "sell_price": price * (1 + self.config.profit_target),
                }
                portfolio["cash"] -= allocation
                logger.info(f"Grid buy: {quantity:.4f} {symbol} at {price:.4f} EUR")
                price_monitor_manager.start(symbol, portfolio, portfolio_lock, None)
                return buy_trade_data
            finally:
                portfolio_lock.release()
        except Exception as e:
            error_handler.handle("Grid Buy Failure", f"Error executing grid buy for {symbol}: {e}", e)
            return None

    def should_buy(self, symbol, portfolio, market_data, order_book_metrics, trade_volume_eur):
        conditions = self.config.buy_conditions
        metrics = next((m for m in order_book_metrics if m.get("market") == symbol.replace("/", "-")), {})
        total_score = metrics.get("total_score", 0)
        slippage_buy = metrics.get("slippage_buy", float("inf"))

        checks = [
            (conditions["not_in_portfolio"]["enabled"], lambda: symbol not in portfolio.get("assets", {}), "Already owned"),
            (conditions["sufficient_cash"]["enabled"], lambda: portfolio.get("cash", 0) >= conditions["sufficient_cash"]["portfolio_value"] * conditions["sufficient_cash"]["allocation_per_trade"], "Insufficient cash"),
            (conditions["below_max_assets"]["enabled"], lambda: len(portfolio.get("assets", {})) < conditions["below_max_assets"]["max_active_assets"], "Max assets reached"),
            (conditions["not_low_volatility"]["enabled"], lambda: symbol not in low_volatility_assets, "Low volatility asset"),
            (conditions["min_score"]["enabled"], lambda: total_score >= conditions["min_score"]["min_total_score"], f"Score {total_score:.2f} < {conditions['min_score']['min_total_score']:.2f}"),
            (conditions["max_buy_slippage"]["enabled"], lambda: slippage_buy <= conditions["max_buy_slippage"]["max_slippage_buy"], f"Slippage {slippage_buy:.2f}% > {conditions['max_buy_slippage']['max_slippage_buy']:.2f}%"),
            (conditions["min_volume"]["enabled"], lambda: trade_volume_eur >= conditions["min_volume"]["min_trade_volume_eur"], f"Trade volume {trade_volume_eur:.2f} EUR < {conditions['min_volume']['min_trade_volume_eur']:.2f} EUR")
        ]

        for enabled, condition, reason in checks:
            if enabled and not condition():
                logger.debug(f"Cannot buy {symbol}: {reason}")
                return False, reason

        if conditions["rsi_oversold"]["enabled"]:
            rsi = self.calculate_rsi(symbol, market_data, conditions["rsi_oversold"]["period"])
            if rsi is not None and rsi > conditions["rsi_oversold"]["threshold"]:
                return False, f"RSI {rsi:.2f} > oversold threshold {conditions['rsi_oversold']['threshold']}"

        if conditions["bollinger_lower"]["enabled"]:
            mean, upper, lower = self.calculate_bollinger_bands(symbol, market_data, conditions["bollinger_lower"]["period"], conditions["bollinger_lower"]["std_dev"])
            if mean is None or lower is None:
                return False, "Insufficient data for Bollinger Bands"
            current_price = market_data[market_data["symbol"] == symbol]["close_price"].iloc[-1]
            if current_price > lower:
                return False, f"Price {current_price:.2f} > lower Bollinger Band {lower:.2f}"

        if conditions["macd_bullish"]["enabled"]:
            macd, signal = self.calculate_macd(symbol, market_data, conditions["macd_bullish"]["fast_period"], conditions["macd_bullish"]["slow_period"], conditions["macd_bullish"]["signal_period"])
            if macd is None or signal is None:
                return False, "Insufficient data for MACD"
            if macd <= signal:
                return False, f"MACD {macd:.2f} not above signal {signal:.2f}"

        if self.config.grid_enabled and self.is_sideways_market(symbol, market_data):
            return False, "Sideways market, using grid trading"

        return True, None

    def should_sell(self, symbol, asset, market_data, holding_minutes, negative_momentum_counts, trade_volume_eur, prev_volume_eur):
        current_price = market_data[market_data["symbol"] == symbol]["close_price"].iloc[-1] if "close_price" in market_data.columns else asset["purchase_price"]
        symbol_candles = market_data[market_data["symbol"] == symbol].tail(5)
        atr = 0
        if len(symbol_candles) >= 5 and all(col in symbol_candles.columns for col in ["high", "low"]):
            atr = np.mean(symbol_candles["high"] - symbol_candles["low"])
        else:
            logger.warning(f"Missing 'high' or 'low' columns for {symbol} in should_sell. Using default ATR=0.")
        profit_target = max(0.015, min(0.05, 1.2 * atr / asset["purchase_price"])) if atr > 0 and asset["purchase_price"] > 0 else self.config.profit_target
        total_cost = asset["purchase_price"] * asset["quantity"] + asset.get("buy_fee", 0)
        unrealized_profit = ((current_price * asset["quantity"]) - total_cost) / total_cost if total_cost > 0 else 0
        trailing_stop = (
            self.config.sell_conditions["regular_sell"]["trailing_stop_factor_early"]
            if holding_minutes < 15
            else self.config.sell_conditions["regular_sell"]["trailing_stop_factor"]
        ) * atr / asset["purchase_price"] if atr > 0 else 0.05
        trailing_loss = (asset["highest_price"] - current_price) / asset["highest_price"] if asset["highest_price"] > asset["purchase_price"] else 0

        sell_reasons = []
        conditions = self.config.sell_conditions

        if conditions["catastrophic_loss"]["enabled"] and unrealized_profit <= conditions["catastrophic_loss"]["threshold"] and abs(unrealized_profit) > conditions["catastrophic_loss"]["atr_multiplier"] * atr / asset["purchase_price"]:
            sell_reasons.append(("Catastrophic loss", unrealized_profit))
        if conditions["time_stop"]["enabled"] and holding_minutes >= conditions["time_stop"]["time_stop_minutes"] and unrealized_profit < 0:
            sell_reasons.append(("Time stop", holding_minutes))
        if conditions["multiplied_profit"]["enabled"] and unrealized_profit >= conditions["multiplied_profit"]["multiplier"] * profit_target:
            sell_reasons.append((f"Multiplied profit target ({conditions['multiplied_profit']['multiplier']}x)", unrealized_profit))
        if conditions["regular_sell"]["enabled"] and holding_minutes >= conditions["regular_sell"]["min_holding_minutes"]:
            if trailing_loss >= trailing_stop:
                sell_reasons.append(("Trailing stop", trailing_loss))
            if unrealized_profit >= profit_target:
                sell_reasons.append((f"Dynamic profit target ({profit_target*100:.1f}%)", unrealized_profit))
            if negative_momentum_counts.get(symbol, 0) >= conditions["regular_sell"]["momentum_confirm_minutes"]:
                sell_reasons.append(("Negative momentum", negative_momentum_counts.get(symbol, 0)))
        if conditions["rsi_overbought"]["enabled"]:
            rsi = self.calculate_rsi(symbol, market_data, conditions["rsi_overbought"]["period"])
            if rsi is not None and rsi > conditions["rsi_overbought"]["threshold"]:
                sell_reasons.append((f"RSI overbought {rsi:.2f}", rsi))
        if conditions["volume_drop"]["enabled"] and prev_volume_eur > 0 and trade_volume_eur / prev_volume_eur < conditions["volume_drop"]["drop_threshold"]:
            sell_reasons.append((f"Volume drop {trade_volume_eur/prev_volume_eur:.2f}x", trade_volume_eur))
        if conditions["bollinger_upper"]["enabled"]:
            mean, upper, lower = self.calculate_bollinger_bands(symbol, market_data, conditions["bollinger_upper"]["period"], conditions["bollinger_upper"]["std_dev"])
            if mean is not None and upper is not None and current_price > upper:
                sell_reasons.append((f"Price above upper Bollinger Band {upper:.2f}", current_price))
        if conditions["macd_bearish"]["enabled"]:
            macd, signal = self.calculate_macd(symbol, market_data, conditions["macd_bearish"]["fast_period"], conditions["macd_bearish"]["slow_period"], conditions["macd_bearish"]["signal_period"])
            if macd is not None and signal is not None and macd < signal:
                sell_reasons.append((f"MACD bearish crossover {macd:.2f} < {signal:.2f}", macd))

        return sell_reasons if sell_reasons else None

    def should_dca(self, symbol, asset, portfolio, market_data):
        if not self.config.dca_enabled:
            return False, "DCA disabled"
        price_series = market_data.get("close_price", asset["purchase_price"])
        current_price = price_series.iloc[-1] if isinstance(price_series, pd.Series) else price_series
        if current_price <= 0:
            return False, "Invalid price"
        drop_percentage = (current_price - asset["purchase_price"]) / asset["purchase_price"] if asset["purchase_price"] > 0 else 0
        if drop_percentage >= -self.config.dca_drop_threshold:
            return False, f"Drop {drop_percentage*100:.1f}% < threshold {-self.config.dca_drop_threshold*100:.1f}%"
        total_allocation = portfolio.get("cash", 0) + sum(a["quantity"] * a["current_price"] for a in portfolio.get("assets", {}).values())
        current_allocation = asset["quantity"] * asset["current_price"]
        if current_allocation / total_allocation > self.config.dca_max_allocation:
            return False, f"Allocation {current_allocation/total_allocation*100:.1f}% > max {self.config.dca_max_allocation*100:.1f}%"
        return True, None

    def calculate_metrics(self, asset, market_data, holding_minutes):
        symbol = asset.get("symbol")
        if not symbol:
            logger.error("Missing symbol in asset data")
            return {"profit_target": self.config.profit_target, "trailing_stop": 0.05}
        symbol_candles = market_data[market_data["symbol"] == symbol].tail(5)
        atr = 0
        if len(symbol_candles) >= 5 and all(col in symbol_candles.columns for col in ["high", "low"]):
            atr = np.mean(symbol_candles["high"] - symbol_candles["low"])
        else:
            logger.warning(f"Missing 'high' or 'low' columns for {symbol} in calculate_metrics. Using default ATR=0.")
        profit_target = max(0.015, min(0.05, 1.2 * atr / asset["purchase_price"])) if atr > 0 and asset["purchase_price"] > 0 else self.config.profit_target
        trailing_stop = (
            self.config.sell_conditions["regular_sell"]["trailing_stop_factor_early"]
            if holding_minutes < 15
            else self.config.sell_conditions["regular_sell"]["trailing_stop_factor"]
        ) * atr / asset["purchase_price"] if atr > 0 else 0.05
        return {"profit_target": profit_target, "trailing_stop": trailing_stop}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.RequestException, requests.HTTPError)),
    before_sleep=lambda retry_state: logger.info(f"Retrying fetch_ticker_price after {retry_state.attempt_number} attempts"),
    reraise=True
)
def fetch_ticker_price_with_retry(symbol):
    """Fetches ticker price with retry logic."""
    price = fetch_ticker_price(symbol)
    if price is None:
        raise ValueError(f"Failed to fetch price for {symbol}")
    return float(price)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.RequestException, requests.HTTPError)),
    before_sleep=lambda retry_state: logger.info(f"Retrying fetch_trade_details after {retry_state.attempt_number} attempts"),
    reraise=True
)
def fetch_trade_details_with_retry(symbol, start_time, end_time):
    """Fetches trade details with retry logic."""
    trade_count, largest_trade_volume_eur = fetch_trade_details(symbol, start_time, end_time)
    if trade_count is None or largest_trade_volume_eur is None:
        raise ValueError(f"Failed to fetch trade details for {symbol}")
    return trade_count, largest_trade_volume_eur

def sell_asset(
    symbol,
    asset,
    current_price,
    portfolio,
    portfolio_lock,
    finished_trades,
    reason,
    price_monitor_manager,
    sell_slippage=0.0,
):
    """
    Sells a specified asset and updates the portfolio.
    """
    try:
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"Invalid symbol: {symbol}")
        if not isinstance(asset, dict) or not all(key in asset for key in ["quantity", "purchase_price", "purchase_time"]):
            raise ValueError(f"Invalid asset data for {symbol}")
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError(f"Invalid sell price {current_price} for {symbol}")
        if not isinstance(sell_slippage, (int, float)):
            raise ValueError(f"Invalid sell_slippage {sell_slippage} for {symbol}")

        if not portfolio_lock.acquire(timeout=5):
            error_handler.handle("Portfolio Lock Failure", f"Timeout acquiring portfolio lock for {symbol}")
            return None

        try:
            logger.debug(f"Starting sell process for {symbol}: {reason}")
            sale_value = asset["quantity"] * current_price * (1 - abs(sell_slippage))
            sell_fee = sale_value * config.config.SELL_FEE
            net_sale_value = sale_value - sell_fee
            buy_value = asset["quantity"] * asset["purchase_price"]
            buy_fee = buy_value * config.config.BUY_FEE
            profit_loss = net_sale_value - (buy_value + buy_fee)
            finished_trade = {
                "Symbol": symbol,
                "Buy Quantity": f"{asset['quantity']:.10f}",
                "Buy Price": f"{asset['purchase_price']:.10f}",
                "Buy Time": asset["purchase_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "Buy Fee": f"{buy_fee:.2f}",
                "Sell Quantity": f"{asset['quantity']:.10f}",
                "Sell Price": f"{current_price * (1 - abs(sell_slippage)):.10f}",
                "Sell Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Sell Fee": f"{sell_fee:.2f}",
                "Sell Slippage": f"{sell_slippage:.2f}%",
                "Profit/Loss": f"{profit_loss:.2f}",
                "Reason": reason,
            }
            portfolio["cash"] += net_sale_value
            finished_trades.append(finished_trade)
            logger.info(
                f"Sold {asset['quantity']:.10f} {symbol} at {current_price * (1 - abs(sell_slippage)):.8f} EUR "
                f"(after {sell_slippage:.2f}% slippage and {sell_fee:.2f} fee) for {net_sale_value:.2f} €. Reason: {reason}"
            )
            del portfolio["assets"][symbol]
            low_volatility_assets.discard(symbol)
            negative_momentum_counts.pop(symbol, None)
        finally:
            portfolio_lock.release()

        try:
            if price_monitor_manager:
                price_monitor_manager.stop(symbol)
            append_to_finished_trades_csv(finished_trade)
            telegram_notifier.notify_sell_trade(finished_trade)
        except Exception as e:
            error_handler.handle("Post-Sale Action Failure", f"Failed to process post-sale actions for {symbol}: {e}", e)
        return finished_trade
    except ValueError as e:
        error_handler.handle("Sell Asset Failure", f"Validation error for {symbol}: {e}", e)
        return None
    except Exception as e:
        error_handler.handle("Sell Asset Failure", f"Unexpected error for {symbol}: {e}", e)
        return None

def sell_most_profitable_asset(
    portfolio,
    portfolio_lock,
    market_data,
    finished_trades,
    price_monitor_manager=None,
    asset_sell_slippages=None,
):
    """
    Sells the most profitable asset to free up a portfolio slot.
    """
    try:
        if not isinstance(portfolio, dict) or "assets" not in portfolio:
            raise ValueError("Invalid portfolio structure")
        if not isinstance(market_data, pd.DataFrame) or "symbol" not in market_data.columns:
            raise ValueError("Invalid market_data DataFrame")
        if not isinstance(finished_trades, list):
            raise ValueError("finished_trades must be a list")

        if not portfolio_lock.acquire(timeout=5):
            error_handler.handle("Portfolio Lock Failure", "Timeout acquiring portfolio lock")
            return None

        try:
            current_time = datetime.utcnow()
            strategy_config = StrategyConfig()
            profitable_assets = [
                (symbol, asset)
                for symbol, asset in portfolio["assets"].items()
                if isinstance(asset.get("purchase_time"), datetime)
                and (current_time - asset["purchase_time"]).total_seconds() / 60 >= strategy_config.sell_conditions["regular_sell"]["min_holding_minutes"]
            ]
            if not profitable_assets:
                logger.info("No profitable assets eligible for sale.")
                return None

            max_profit = -float("inf")
            asset_to_sell = None
            for symbol, asset in profitable_assets:
                current_price_series = market_data[market_data["symbol"] == symbol]["close_price"]
                current_price = float(current_price_series.iloc[0]) if not current_price_series.empty else fetch_ticker_price_with_retry(symbol)
                unrealized_profit = (
                    (current_price - asset["purchase_price"]) / asset["purchase_price"]
                    if asset["purchase_price"] > 0 else 0
                )
                sell_slippage = asset_sell_slippages.get(symbol, strategy_config.max_slippage_sell + 0.1) if asset_sell_slippages else (strategy_config.max_slippage_sell + 0.1)
                if unrealized_profit >= 0.01 and unrealized_profit > max_profit and abs(sell_slippage) <= strategy_config.max_slippage_sell:
                    max_profit = unrealized_profit
                    asset_to_sell = (symbol, asset, current_price, sell_slippage)
            if asset_to_sell is None:
                logger.info("No assets with unrealized profit >= 1% or acceptable slippage.")
                return None

            symbol, asset, current_price, sell_slippage = asset_to_sell
            return sell_asset(
                symbol,
                asset,
                current_price,
                portfolio,
                portfolio_lock,
                finished_trades,
                "Sold to free up slot for new buy",
                price_monitor_manager,
                sell_slippage,
            )
        finally:
            portfolio_lock.release()
    except Exception as e:
        error_handler.handle("Sell Profitable Asset Failure", f"Error: {e}", e)
        return None

def save_portfolio():
    """
    Saves the portfolio state to a JSON file, keeping only the 3 latest backups.
    """
    try:
        if not isinstance(portfolio, dict) or "cash" not in portfolio or "assets" not in portfolio:
            raise ValueError("Invalid portfolio structure")

        if not portfolio_lock.acquire(timeout=5):
            error_handler.handle("Portfolio Lock Failure", "Failed to acquire portfolio lock")
            return

        try:
            portfolio_copy = {
                "cash": portfolio["cash"],
                "assets": {
                    symbol: {
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in asset.items()
                    }
                    for symbol, asset in portfolio["assets"].items()
                },
            }
            file_path = config.config.PORTFOLIO_FILE
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(portfolio_copy, temp_file, indent=4)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            shutil.move(temp_file.name, file_path)

            backup_file = f"{file_path}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(portfolio_copy, temp_file, indent=4)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            shutil.move(temp_file.name, backup_file)

            backup_files = glob.glob(f"{file_path}.backup_*")
            backup_files.sort(key=lambda x: x.split("backup_")[-1], reverse=True)
            for old_file in backup_files[3:]:
                try:
                    os.remove(old_file)
                    logger.debug(f"Deleted old backup file: {old_file}")
                except OSError as e:
                    logger.warning(f"Error deleting old backup file {old_file}: {e}")
        finally:
            portfolio_lock.release()
    except Exception as e:
        error_handler.handle("Portfolio Save Failure", f"Error saving portfolio: {e}", e)

def validate_inputs(candidate_assets, market_data, price_monitor_manager, order_book_metrics_list, asset_sell_slippages):
    """Validates inputs for portfolio management."""
    if not isinstance(candidate_assets, list):
        raise ValueError("candidate_assets must be a list")
    if not isinstance(market_data, pd.DataFrame) or not {"symbol", "close_price"}.issubset(market_data.columns):
        raise ValueError("Invalid market_data DataFrame")
    if not price_monitor_manager:
        raise ValueError("price_monitor_manager cannot be None")
    if order_book_metrics_list is None:
        order_book_metrics_list = []
    elif not isinstance(order_book_metrics_list, list):
        raise ValueError("order_book_metrics_list must be a list")
    if asset_sell_slippages is not None and not isinstance(asset_sell_slippages, dict):
        raise ValueError("asset_sell_slippages must be a dictionary")
    return order_book_metrics_list

def handle_orphaned_monitors(price_monitor_manager, portfolio):
    """Stops monitors for assets not in the portfolio."""
    active_monitors = set(price_monitor_manager.running.keys()) if price_monitor_manager.running else set()
    active_assets = set(portfolio.get("assets", {}).keys())
    for symbol in active_monitors - active_assets:
        logger.warning(f"Stopping orphaned monitor for {symbol}.")
        price_monitor_manager.stop(symbol)

def get_current_price(symbol, market_data):
    """Fetches the current price for an asset."""
    current_price_series = market_data[market_data["symbol"] == symbol]["close_price"]
    if current_price_series.empty:
        logger.warning(f"No price in market_data for {symbol}. Fetching ticker price.")
        try:
            return fetch_ticker_price_with_retry(symbol)
        except Exception as e:
            error_handler.handle("Price Fetch Failure", f"Failed to fetch price for {symbol}: {e}", e)
            return None
    return float(current_price_series.iloc[0])

def update_asset_metrics(asset, current_price, market_data, holding_minutes, strategy):
    """Updates asset metrics like highest_price and profit_target."""
    asset["current_price"] = current_price
    asset["highest_price"] = max(asset.get("highest_price", asset["purchase_price"]), current_price)
    metrics = strategy.calculate_metrics(asset, market_data, holding_minutes)
    asset["profit_target"] = metrics["profit_target"]
    sell_prices = [asset["purchase_price"] * (1 + metrics["profit_target"])]
    if asset["highest_price"] > asset["purchase_price"]:
        trailing_stop_price = asset["highest_price"] * (1 - metrics["trailing_stop"])
        if trailing_stop_price > asset["purchase_price"]:
            sell_prices.append(trailing_stop_price)
    asset["sell_price"] = min(sell_prices) if sell_prices else asset["purchase_price"] * (1 + metrics["profit_target"])

def evaluate_and_sell_assets(portfolio, market_data, price_monitor_manager, asset_sell_slippages, strategy, prev_volumes):
    """Evaluates and sells assets based on strategy conditions."""
    finished_trades = []
    current_time = datetime.utcnow()
    skipped_assets = []
    total_asset_value = 0.0

    for symbol in list(portfolio.get("assets", {}).keys()):
        asset = portfolio["assets"][symbol]
        if symbol not in low_volatility_assets and symbol not in price_monitor_manager.running:
            price_monitor_manager.start(symbol, portfolio, portfolio_lock, market_data)

        current_price = get_current_price(symbol, market_data)
        if current_price is None:
            skipped_assets.append(symbol)
            total_asset_value += asset["quantity"] * asset["purchase_price"]
            continue

        price_monitor_manager.last_prices[symbol] = current_price
        holding_minutes = (current_time - asset["purchase_time"]).total_seconds() / 60 if isinstance(asset.get("purchase_time"), datetime) else 0
        update_asset_metrics(asset, current_price, market_data, holding_minutes, strategy)
        total_asset_value += asset["quantity"] * current_price

        momentum = market_data[market_data["symbol"] == symbol]["percent_change"].iloc[0] if not market_data[market_data["symbol"] == symbol].empty else 0
        negative_momentum_counts[symbol] = negative_momentum_counts.get(symbol, 0) + 1 if momentum < strategy.config.momentum_threshold else 0

        try:
            trade_count, trade_volume_eur = fetch_trade_details_with_retry(symbol, current_time - timedelta(minutes=5), current_time)
        except Exception as e:
            error_handler.handle("Trade Details Failure", f"Failed to fetch trade details for {symbol}: {e}", e)
            trade_volume_eur = 0

        sell_reasons = strategy.should_sell(symbol, asset, market_data, holding_minutes, negative_momentum_counts, trade_volume_eur, prev_volumes.get(symbol, 0))
        sell_slippage = asset_sell_slippages.get(symbol, strategy.config.max_slippage_sell + 0.1)
        if sell_reasons and abs(sell_slippage) <= strategy.config.max_slippage_sell:
            reason, value = sell_reasons[0]
            logger.info(
                f"Selling {symbol}: {reason}, Current: {current_price:.4f}, P/L: {(current_price - asset['purchase_price'])/asset['purchase_price']*100:.2f}%, Slippage: {sell_slippage:.2f}%"
            )
            trade = sell_asset(
                symbol, asset, current_price, portfolio, portfolio_lock, finished_trades, reason, price_monitor_manager, sell_slippage
            )
            if trade:
                finished_trades.append(trade)
        else:
            logger.info(
                f"Holding {symbol}: Current: {current_price:.4f}, P/L: {(current_price - asset['purchase_price'])/asset['purchase_price']*100:.2f}%, Slippage: {sell_slippage:.2f}%"
            )
        prev_volumes[symbol] = trade_volume_eur

        # Check grid orders for this asset
        if strategy.config.grid_enabled and strategy.is_sideways_market(symbol, market_data):
            grid_trades = strategy.check_grid_orders(symbol, current_price, portfolio, portfolio_lock, finished_trades, price_monitor_manager)
            finished_trades.extend(grid_trades)

    return finished_trades, total_asset_value, skipped_assets


def buy_new_assets(candidate_assets, portfolio, market_data, price_monitor_manager, order_book_metrics_list, asset_sell_slippages, strategy):
    # Preprocess market_data to ensure correct column names
    market_data = preprocess_market_data(market_data)
    
    current_time = datetime.utcnow()
    five_min_ago = current_time - timedelta(minutes=5)
    total_asset_value = portfolio.get("cash", 0)
    finished_trades = []
    prev_volumes = {}

    for record in candidate_assets:
        if not isinstance(record, dict) or "symbol" not in record or "close_price" not in record:
            logger.warning(f"Invalid record in candidate_assets: {record}")
            continue
        symbol = record["symbol"]
        close_price = record["close_price"]

        # Check drawdown limit
        total_portfolio_value = portfolio.get("cash", 0) + sum(a["quantity"] * a["current_price"] for a in portfolio.get("assets", {}).values())
        strategy.peak_portfolio_value = max(strategy.peak_portfolio_value, total_portfolio_value)
        current_drawdown = (strategy.peak_portfolio_value - total_portfolio_value) / strategy.peak_portfolio_value if strategy.peak_portfolio_value > 0 else 0
        if current_drawdown >= strategy.config.max_drawdown:
            logger.warning(f"Trading paused: Drawdown {current_drawdown*100:.1f}% >= {strategy.config.max_drawdown*100:.1f}%")
            continue

        try:
            trade_count, trade_volume_eur = fetch_trade_details_with_retry(symbol, five_min_ago, current_time)
        except Exception as e:
            error_handler.handle("Trade Details Failure", f"Failed to fetch trade details for {symbol}: {e}", e)
            trade_volume_eur = 0
            continue

        if strategy.config.grid_enabled and strategy.is_sideways_market(symbol, market_data):
            strategy.place_grid_orders(symbol, close_price, portfolio, market_data)
            continue

        can_buy, reason = strategy.should_buy(symbol, portfolio, market_data, order_book_metrics_list, trade_volume_eur)
        if not can_buy:
            logger.info(f"Cannot buy {symbol}: {reason}")
            if len(portfolio.get("assets", {})) >= strategy.config.buy_conditions["below_max_assets"]["max_active_assets"]:
                trade = sell_most_profitable_asset(portfolio, portfolio_lock, market_data, finished_trades, price_monitor_manager, asset_sell_slippages)
                if trade:
                    finished_trades.append(trade)
            continue

        metrics = next((m for m in order_book_metrics_list if m.get("market") == symbol.replace("/", "-")), {})
        slippage_buy = metrics.get("slippage_buy", float("inf"))
        purchase_price = close_price * (1 + slippage_buy / 100)

        # Volatility-adjusted position sizing
        base_allocation = strategy.config.buy_conditions["sufficient_cash"]["portfolio_value"] * strategy.config.buy_conditions["sufficient_cash"]["allocation_per_trade"]
        allocation = base_allocation
        if strategy.config.volatility_sizing_enabled:
            atr = strategy.calculate_atr(symbol, market_data, strategy.config.atr_period)
            if atr is not None and atr > 0 and close_price > 0:
                volatility_factor = min(strategy.config.max_allocation_multiplier, max(strategy.config.min_allocation_multiplier, 1.0 / (atr / close_price)))
                allocation = min(base_allocation * volatility_factor, strategy.config.max_position_size)
                logger.info(f"Adjusted allocation for {symbol} by volatility factor {volatility_factor:.2f} (ATR: {atr:.2f})")
            else:
                allocation = min(base_allocation, strategy.config.max_position_size)
                logger.info(f"Using default allocation for {symbol} due to missing ATR data")

        buy_fee = allocation * strategy.config.buy_fee
        net_allocation = allocation - buy_fee
        quantity = net_allocation / purchase_price if purchase_price > 0 else 0

        if quantity <= 0:
            logger.warning(f"Cannot buy {symbol}: Invalid quantity {quantity}")
            continue

        buy_trade_data = {
            "Symbol": symbol,
            "Buy Quantity": f"{quantity:.10f}",
            "Buy Price": f"{purchase_price:.10f}",
            "Buy Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Buy Fee": f"{buy_fee:.2f}",
            "Buy Slippage": f"{slippage_buy:.2f}%",
            "Allocation": f"{allocation:.2f}",
            "Actual Cost": f"{quantity * purchase_price:.2f}",
            "Trade Count": trade_count,
            "Largest Trade Volume EUR": f"{trade_volume_eur:.2f}",
        }
        append_to_buy_trades_csv(buy_trade_data)
        telegram_notifier.notify_buy_trade(buy_trade_data)
        portfolio["assets"][symbol] = {
            "quantity": quantity,
            "purchase_price": purchase_price,
            "purchase_time": current_time,
            "buy_slippage": slippage_buy,
            "buy_fee": buy_fee,
            "highest_price": purchase_price,
            "current_price": close_price,
            "profit_target": strategy.config.profit_target,
            "original_profit_target": strategy.config.profit_target,
            "sell_price": purchase_price * (1 + strategy.config.profit_target),
            "symbol": symbol
        }
        portfolio["cash"] -= allocation
        total_asset_value += quantity * close_price
        logger.info(
            f"Bought {quantity:.4f} {symbol} at {purchase_price:.4f} EUR (close {close_price:.4f}) for {quantity * purchase_price:.2f} EUR"
        )
        for metrics in order_book_metrics_list:
            if metrics.get("market") == symbol.replace("/", "-"):
                metrics["bought"] = True
        price_monitor_manager.start(symbol, portfolio, portfolio_lock, market_data)
        prev_volumes[symbol] = trade_volume_eur

    # DCA for existing assets
    if strategy.config.dca_enabled:
        for symbol, asset in list(portfolio.get("assets", {}).items()):
            can_dca, reason = strategy.should_dca(symbol, asset, portfolio, market_data)
            if can_dca:
                current_price = get_current_price(symbol, market_data)
                if current_price is None:
                    continue
                base_allocation = strategy.config.buy_conditions["sufficient_cash"]["portfolio_value"] * strategy.config.buy_conditions["sufficient_cash"]["allocation_per_trade"] / 2
                allocation = base_allocation
                if strategy.config.volatility_sizing_enabled:
                    atr = strategy.calculate_atr(symbol, market_data, strategy.config.atr_period)
                    if atr is not None and atr > 0 and current_price > 0:
                        volatility_factor = min(strategy.config.max_allocation_multiplier, max(strategy.config.min_allocation_multiplier, 1.0 / (atr / current_price)))
                        allocation = min(base_allocation * volatility_factor, strategy.config.max_position_size)
                        logger.info(f"DCA: Adjusted allocation for {symbol} by volatility factor {volatility_factor:.2f} (ATR: {atr:.2f})")
                    else:
                        allocation = min(base_allocation, strategy.config.max_position_size)
                buy_fee = allocation * strategy.config.buy_fee
                net_allocation = allocation - buy_fee
                quantity = net_allocation / current_price if current_price > 0 else 0
                if quantity <= 0:
                    continue
                avg_price = (asset["quantity"] * asset["purchase_price"] + quantity * current_price) / (asset["quantity"] + quantity)
                asset["quantity"] += quantity
                asset["purchase_price"] = avg_price
                asset["buy_fee"] += buy_fee
                portfolio["cash"] -= allocation
                total_asset_value += quantity * current_price
                buy_trade_data = {
                    "Symbol": symbol,
                    "Buy Quantity": f"{quantity:.10f}",
                    "Buy Price": f"{current_price:.10f}",
                    "Buy Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Buy Fee": f"{buy_fee:.2f}",
                    "Buy Slippage": "0.00%",
                    "Allocation": f"{allocation:.2f}",
                    "Actual Cost": f"{quantity * current_price:.2f}",
                    "Trade Count": 0,
                    "Largest Trade Volume EUR": "0.00"
                }
                append_to_buy_trades_csv(buy_trade_data)
                telegram_notifier.notify_buy_trade(buy_trade_data)
                logger.info(f"DCA: Bought {quantity:.4f} {symbol} at {current_price:.4f} EUR, new avg price {avg_price:.4f}")

    return finished_trades, total_asset_value

def send_notifications(portfolio, portfolio_values):
    """Sends periodic portfolio notifications."""
    current_time = datetime.utcnow()
    if not hasattr(telegram_notifier, 'last_summary_time') or (current_time - telegram_notifier.last_summary_time).total_seconds() >= 3600:
        telegram_notifier.notify_portfolio_summary(portfolio)
        telegram_notifier.last_summary_time = current_time

    if (current_time - getattr(telegram_notifier, 'last_pinned_time', current_time)).total_seconds() >= 60:
        asyncio.run_coroutine_threadsafe(telegram_notifier.update_pinned_summary(portfolio), asyncio.get_event_loop())
        telegram_notifier.last_pinned_time = current_time

    if (current_time - getattr(telegram_notifier, 'last_chart_time', current_time)).total_seconds() >= 86400:
        asyncio.run_coroutine_threadsafe(telegram_notifier.notify_performance_chart(portfolio_values), asyncio.get_event_loop())
        telegram_notifier.last_chart_time = current_time

    if (current_time - getattr(telegram_notifier, 'last_allocation_time', current_time)).total_seconds() >= 86400:
        asyncio.run_coroutine_threadsafe(telegram_notifier.notify_asset_allocation(portfolio), asyncio.get_event_loop())
        telegram_notifier.last_allocation_time = current_time

    if current_time.hour == 0 and (not hasattr(telegram_notifier, 'last_report_date') or current_time.date() != telegram_notifier.last_report_date):
        asyncio.run_coroutine_threadsafe(telegram_notifier.notify_daily_report(), asyncio.get_event_loop())
        telegram_notifier.last_report_date = current_time.date()

def manage_portfolio(
    candidate_assets,
    market_data,
    price_monitor_manager,
    order_book_metrics_list=None,
    asset_sell_slippages=None,
):
    """
    Manages the portfolio by processing sell and buy decisions.
    """
    try:
        # Validate market_data
        if not isinstance(market_data, (pd.DataFrame, list)):
            logger.error(f"Invalid market_data type: {type(market_data)}. Expected DataFrame or list.")
            return [], 0
        if isinstance(market_data, pd.DataFrame) and market_data.empty:
            logger.warning("market_data is an empty DataFrame")
            return [], 0
        if isinstance(market_data, list) and not market_data:
            logger.warning("market_data is an empty list")
            return [], 0
        
        # Convert market_data to DataFrame if it's a list of dictionaries
        if isinstance(market_data, list):
            logger.warning("market_data is a list of dictionaries; converting to DataFrame")
            market_data = pd.DataFrame(market_data)
        
        # Check for duplicate columns
        if market_data.columns.duplicated().any():
            logger.warning(f"Duplicate columns in market_data: {market_data.columns[market_data.columns.duplicated()].tolist()}")
            market_data = market_data.loc[:, ~market_data.columns.duplicated()]
        
        # Ensure market_data has required columns
        required_columns = {"symbol", "close_price", "volume"}
        if not required_columns.issubset(market_data.columns):
            logger.error(f"market_data missing required columns: {required_columns - set(market_data.columns)}")
            return [], 0
        
        # Log volume for SEI/EUR
        if "symbol" in market_data.columns and "volume" in market_data.columns:
            sei_volume = market_data[market_data["symbol"] == "SEI/EUR"]["volume"].sum() if "SEI/EUR" in market_data["symbol"].values else 0
            logger.debug(f"SEI/EUR volume before preprocessing: {sei_volume:.2f} EUR")
        
        # Preprocess market_data to ensure correct column names and types
        market_data = preprocess_market_data(market_data)
        if market_data.empty:
            logger.warning("preprocess_market_data returned an empty DataFrame")
            return [], 0
        
        # Log volume for SEI/EUR after preprocessing
        if "symbol" in market_data.columns and "volume" in market_data.columns:
            sei_volume = market_data[market_data["symbol"] == "SEI/EUR"]["volume"].sum() if "SEI/EUR" in market_data["symbol"].values else 0
            logger.debug(f"SEI/EUR volume after preprocessing in manage_portfolio: {sei_volume:.2f} EUR")
        
        # Ensure candidate_assets is a list of dictionaries
        if not isinstance(candidate_assets, list):
            logger.warning("candidate_assets is not a list; using market_data.to_dict('records')")
            candidate_assets = market_data.to_dict("records")
        
        order_book_metrics_list = validate_inputs(candidate_assets, market_data, price_monitor_manager, order_book_metrics_list, asset_sell_slippages)
        strategy = TradingStrategy(StrategyConfig())
        prev_volumes = {}

        # Calculate sell slippages for held assets
        asset_sell_slippages = asset_sell_slippages or {}
        for symbol in portfolio.get("assets", {}):
            if symbol not in asset_sell_slippages:
                try:
                    amount_quote = portfolio["assets"][symbol]["quantity"] * portfolio["assets"][symbol]["current_price"]
                    metrics = calculate_order_book_metrics(symbol.replace("/", "-"), amount_quote=amount_quote)
                    asset_sell_slippages[symbol] = metrics.get("slippage_sell", -(strategy.config.max_slippage_sell + 0.1))
                except Exception as e:
                    error_handler.handle("Slippage Calculation Failure", f"Error for {symbol}: {e}", e)
                    asset_sell_slippages[symbol] = -(strategy.config.max_slippage_sell + 0.1)

        if not portfolio_lock.acquire(timeout=5):
            error_handler.handle("Portfolio Lock Failure", "Failed to acquire portfolio lock")
            return [], 0

        try:
            handle_orphaned_monitors(price_monitor_manager, portfolio)

            # Update asset prices
            for symbol, asset in portfolio.get("assets", {}).items():
                current_price = get_current_price(symbol, market_data)
                if current_price is not None:
                    asset["current_price"] = current_price
                    asset["highest_price"] = max(asset["highest_price"], current_price)

            # Adjust profit targets if portfolio is near threshold
            if len(portfolio.get("assets", {})) >= strategy.config.asset_threshold and candidate_assets:
                profitable_assets = [
                    symbol for symbol, asset in portfolio.get("assets", {}).items()
                    if asset.get("current_price", 0) > asset.get("purchase_price", 0) * 1.01
                    and isinstance(asset.get("purchase_time"), datetime)
                    and (datetime.utcnow() - asset["purchase_time"]).total_seconds() / 60 >= strategy.config.sell_conditions["regular_sell"]["min_holding_minutes"]
                ]
                if profitable_assets:
                    for symbol in profitable_assets:
                        portfolio["assets"][symbol]["profit_target"] = min(
                            portfolio["assets"][symbol].get("profit_target", strategy.config.profit_target),
                            strategy.config.adjusted_profit_target
                        )
                        logger.info(f"Adjusted profit target for {symbol} to {strategy.config.adjusted_profit_target}")
                else:
                    trade = sell_most_profitable_asset(
                        portfolio, portfolio_lock, market_data, [], price_monitor_manager, asset_sell_slippages
                    )
                    if trade:
                        portfolio_values.append({
                            "timestamp": datetime.utcnow().isoformat(),
                            "portfolio_value": portfolio.get("cash", 0) + sum(asset["quantity"] * asset["current_price"] for asset in portfolio.get("assets", {}).values())
                        })

            finished_trades, total_asset_value, skipped_assets = evaluate_and_sell_assets(
                portfolio, market_data, price_monitor_manager, asset_sell_slippages, strategy, prev_volumes
            )
            new_trades, new_asset_value = buy_new_assets(
                candidate_assets, portfolio, market_data, price_monitor_manager, order_book_metrics_list, asset_sell_slippages, strategy
            )
            finished_trades.extend(new_trades)
            total_asset_value += new_asset_value - portfolio.get("cash", 0)
        finally:
            portfolio_lock.release()

        if order_book_metrics_list:
            append_to_order_book_metrics_csv(order_book_metrics_list)

        total_portfolio_value = portfolio.get("cash", 0) + total_asset_value
        portfolio_values.append({"timestamp": datetime.utcnow().isoformat(), "portfolio_value": total_portfolio_value})
        if skipped_assets:
            logger.warning(f"Portfolio value may be inaccurate due to missing prices for: {', '.join(skipped_assets)}")
        send_notifications(portfolio, portfolio_values)
        
        logger.info(f"Portfolio management completed. Total asset value: {total_portfolio_value:.2f} EUR")
        return finished_trades, total_portfolio_value
    
    except Exception as e:
        error_handler.handle("Portfolio Management Failure", f"Error: {e}", e)
        logger.error(f"Portfolio management failed: {e}", exc_info=True)
        return [], 0

def send_alert(subject, message):
    """Sends an alert for critical errors."""
    telegram_notifier.notify_error(subject, message)