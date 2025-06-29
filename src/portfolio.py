# trading_bot/src/portfolio.py
import glob
import json
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tempfile
import shutil
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from . import config
from .config import logger
from .exchange import fetch_ticker_price, fetch_trade_details
from .state import low_volatility_assets, negative_momentum_counts, portfolio, portfolio_lock
from .utils import append_to_buy_trades_csv, append_to_finished_trades_csv, append_to_order_book_metrics_csv, calculate_dynamic_ema_period, calculate_ema
from .bitvavo_order_metrics import calculate_order_book_metrics

from .telegram_notifications import TelegramNotifier
import asyncio

portfolio_values = []  # Track portfolio value over time

telegram_notifier = TelegramNotifier(
    bot_token=config.config.TELEGRAM_BOT_TOKEN,
    chat_id=config.config.TELEGRAM_CHAT_ID
)
asyncio.run_coroutine_threadsafe(telegram_notifier.start(), asyncio.get_event_loop())

# --- Metric Classes ---

class Metric:
    """Base class for trading metrics."""
    def __init__(self, name, weight=1.0, enabled=True):
        self.name = name
        self.weight = weight
        self.enabled = enabled

    def compute(self, data):
        """Compute the metric value."""
        raise NotImplementedError(f"Metric {self.name} must implement compute method.")

    def evaluate(self, value):
        """Evaluate if the metric meets the trading condition."""
        raise NotImplementedError(f"Metric {self.name} must implement evaluate method.")

class SlippageBuyMetric(Metric):
    """Evaluates buy slippage against a threshold."""
    def __init__(self, threshold, weight=1.0, enabled=True):
        super().__init__("slippage_buy", weight, enabled)
        self.threshold = threshold  # e.g., 0.05 for 5%

    def compute(self, data):
        return data.get("slippage_buy", float("inf"))

    def evaluate(self, value):
        return value <= self.threshold if value is not None else False

class SlippageSellMetric(Metric):
    """Evaluates sell slippage against a threshold."""
    def __init__(self, threshold, weight=1.0, enabled=True):
        super().__init__("slippage_sell", weight, enabled)
        self.threshold = threshold  # e.g., -0.05 for -5%

    def compute(self, data):
        return data.get("slippage_sell", float("inf"))

    def evaluate(self, value):
        return abs(value) <= abs(self.threshold) if value is not None else False

class TotalScoreMetric(Metric):
    """Evaluates total score from order book analysis."""
    def __init__(self, threshold, weight=1.0, enabled=True):
        super().__init__("total_score", weight, enabled)
        self.threshold = threshold  # e.g., 0.7

    def compute(self, data):
        return data.get("total_score", 0.0)

    def evaluate(self, value):
        return value >= self.threshold if value is not None else False

class MomentumMetric(Metric):
    """Evaluates momentum based on recent price change."""
    def __init__(self, threshold, confirm_minutes, weight=1.0, enabled=True):
        super().__init__("momentum", weight, enabled)
        self.threshold = threshold  # e.g., -0.25
        self.confirm_minutes = confirm_minutes  # e.g., 3

    def compute(self, data):
        symbol = data.get("symbol")
        negative_count = negative_momentum_counts.get(symbol, 0)
        momentum = data.get("momentum", 0.0)
        if momentum < self.threshold:
            negative_momentum_counts[symbol] = negative_count + 1
        else:
            negative_momentum_counts[symbol] = 0
        return negative_momentum_counts[symbol]

    def evaluate(self, value):
        return value < self.confirm_minutes

class ProfitTargetMetric(Metric):
    """Evaluates unrealized profit against a dynamic profit target."""
    def __init__(self, base_target, multiplier, weight=1.0, enabled=True):
        super().__init__("profit_target", weight, enabled)
        self.base_target = base_target  # e.g., 0.05
        self.multiplier = multiplier  # e.g., 2.0

    def compute(self, data):
        current_price = data.get("current_price", 0)
        purchase_price = data.get("purchase_price", 0)
        atr = data.get("atr", 0)
        holding_minutes = data.get("holding_minutes", 0)
        profit_target = (
            max(0.015, min(self.base_target, 1.2 * atr / purchase_price))
            if atr > 0 and purchase_price > 0
            else self.base_target
        )
        if data.get("near_threshold", False):
            profit_target = min(profit_target, config.config.ADJUSTED_PROFIT_TARGET)
        unrealized_profit = (
            (current_price - purchase_price) / purchase_price
            if purchase_price > 0 else 0
        )
        return {
            "unrealized_profit": unrealized_profit,
            "profit_target": profit_target,
            "multiplied_target": self.multiplier * profit_target
        }

    def evaluate(self, value):
        unrealized_profit = value["unrealized_profit"]
        profit_target = value["profit_target"]
        multiplied_target = value["multiplied_target"]
        return unrealized_profit >= profit_target or unrealized_profit >= multiplied_target

class TrailingStopMetric(Metric):
    """Evaluates trailing stop loss based on ATR and holding time."""
    def __init__(self, factor, early_factor, weight=1.0, enabled=True):
        super().__init__("trailing_stop", weight, enabled)
        self.factor = factor  # e.g., 1.0
        self.early_factor = early_factor  # e.g., 1.5

    def compute(self, data):
        current_price = data.get("current_price", 0)
        highest_price = data.get("highest_price", 0)
        purchase_price = data.get("purchase_price", 0)
        atr = data.get("atr", 0)
        holding_minutes = data.get("holding_minutes", 0)
        trailing_stop = (
            (self.early_factor if holding_minutes < 15 else self.factor) * atr / purchase_price
            if atr > 0 and purchase_price > 0
            else 0.05
        )
        trailing_loss = (
            (highest_price - current_price) / highest_price
            if highest_price > purchase_price else 0
        )
        return {"trailing_loss": trailing_loss, "trailing_stop": trailing_stop}

    def evaluate(self, value):
        return value["trailing_loss"] >= value["trailing_stop"]

class CatastrophicLossMetric(Metric):
    """Evaluates catastrophic loss based on unrealized profit and ATR."""
    def __init__(self, threshold, weight=1.0, enabled=True):
        super().__init__("catastrophic_loss", weight, enabled)
        self.threshold = threshold  # e.g., -0.08

    def compute(self, data):
        current_price = data.get("current_price", 0)
        purchase_price = data.get("purchase_price", 0)
        atr = data.get("atr", 0)
        unrealized_profit = (
            (current_price - purchase_price) / purchase_price
            if purchase_price > 0 else 0
        )
        return {
            "unrealized_profit": unrealized_profit,
            "catastrophic": unrealized_profit <= self.threshold and abs(unrealized_profit) > 2 * atr / purchase_price
            if atr > 0 and purchase_price > 0 else False
        }

    def evaluate(self, value):
        return value["catastrophic"]

class TimeStopMetric(Metric):
    """Evaluates time stop based on holding time and unrealized profit."""
    def __init__(self, time_stop_minutes, weight=1.0, enabled=True):
        super().__init__("time_stop", weight, enabled)
        self.time_stop_minutes = time_stop_minutes  # e.g., 90

    def compute(self, data):
        holding_minutes = data.get("holding_minutes", 0)
        current_price = data.get("current_price", 0)
        purchase_price = data.get("purchase_price", 0)
        unrealized_profit = (
            (current_price - purchase_price) / purchase_price
            if purchase_price > 0 else 0
        )
        return {
            "holding_minutes": holding_minutes,
            "unrealized_profit": unrealized_profit
        }

    def evaluate(self, value):
        return value["holding_minutes"] >= self.time_stop_minutes and value["unrealized_profit"] < 0

# --- Trading Strategy Class ---

class TradingStrategy:
    """Manages buy and sell decisions based on configurable metrics."""
    def __init__(self, config_dict=None):
        self.metrics = []
        self.load_config(config_dict or self.default_config())

    def default_config(self):
        """Default strategy configuration."""
        return {
            "buy_metrics": [
                {"name": "slippage_buy", "threshold": config.config.MAX_SLIPPAGE_BUY, "weight": 0.4, "enabled": True},
                {"name": "total_score", "threshold": config.config.MIN_TOTAL_SCORE, "weight": 0.6, "enabled": True},
            ],
            "sell_metrics": [
                {"name": "slippage_sell", "threshold": config.config.MAX_SLIPPAGE_SELL, "weight": 0.2, "enabled": True},
                {"name": "profit_target", "base_target": config.config.PROFIT_TARGET, "multiplier": config.config.PROFIT_TARGET_MULTIPLIER, "weight": 0.3, "enabled": True},
                {"name": "trailing_stop", "factor": config.config.TRAILING_STOP_FACTOR, "early_factor": config.config.TRAILING_STOP_FACTOR_EARLY, "weight": 0.2, "enabled": True},
                {"name": "catastrophic_loss", "threshold": config.config.CAT_LOSS_THRESHOLD, "weight": 0.2, "enabled": True},
                {"name": "time_stop", "time_stop_minutes": config.config.TIME_STOP_MINUTES, "weight": 0.2, "enabled": True},
                {"name": "momentum", "threshold": config.config.MOMENTUM_THRESHOLD, "confirm_minutes": config.config.MOMENTUM_CONFIRM_MINUTES, "weight": 0.1, "enabled": True},
            ]
        }

    def load_config(self, config_dict):
        """Loads metrics from configuration."""
        metric_classes = {
            "slippage_buy": SlippageBuyMetric,
            "slippage_sell": SlippageSellMetric,
            "total_score": TotalScoreMetric,
            "momentum": MomentumMetric,
            "profit_target": ProfitTargetMetric,
            "trailing_stop": TrailingStopMetric,
            "catastrophic_loss": CatastrophicLossMetric,
            "time_stop": TimeStopMetric,
        }
        self.metrics = {"buy": [], "sell": []}
        for metric_config in config_dict.get("buy_metrics", []):
            if metric_config.get("enabled", True):
                metric_class = metric_classes.get(metric_config["name"])
                if metric_class:
                    self.metrics["buy"].append(metric_class(**{k: v for k, v in metric_config.items() if k != "name"}))
        for metric_config in config_dict.get("sell_metrics", []):
            if metric_config.get("enabled", True):
                metric_class = metric_classes.get(metric_config["name"])
                if metric_class:
                    self.metrics["sell"].append(metric_class(**{k: v for k, v in metric_config.items() if k != "name"}))

    def evaluate_buy(self, data):
        """Evaluates buy conditions based on configured metrics."""
        results = []
        total_weight = sum(metric.weight for metric in self.metrics["buy"] if metric.enabled)
        if total_weight == 0:
            logger.warning("No enabled buy metrics configured.")
            return False, []

        for metric in self.metrics["buy"]:
            if not metric.enabled:
                continue
            value = metric.compute(data)
            passes = metric.evaluate(value)
            results.append({
                "metric": metric.name,
                "value": value,
                "passes": passes,
                "weight": metric.weight
            })
            logger.debug(f"Buy metric {metric.name}: value={value}, passes={passes}")

        # Require all metrics to pass (can be modified to use weighted scoring)
        all_pass = all(result["passes"] for result in results)
        return all_pass, results

    def evaluate_sell(self, data):
        """Evaluates sell conditions based on configured metrics."""
        results = []
        total_weight = sum(metric.weight for metric in self.metrics["sell"] if metric.enabled)
        if total_weight == 0:
            logger.warning("No enabled sell metrics configured.")
            return False, [], "No metrics"

        for metric in self.metrics["sell"]:
            if not metric.enabled:
                continue
            value = metric.compute(data)
            passes = metric.evaluate(value)
            results.append({
                "metric": metric.name,
                "value": value,
                "passes": passes,
                "weight": metric.weight
            })
            logger.debug(f"Sell metric {metric.name}: value={value}, passes={passes}")

        # Any passing metric triggers a sell
        for result in results:
            if result["passes"]:
                reason = (
                    f"{result['metric'].replace('_', ' ').title()} "
                    f"(value={result['value']})"
                )
                return True, results, reason
        return False, results, "No sell conditions met"

# --- Modified Functions ---

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.RequestException, requests.HTTPError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying fetch_ticker_price after {retry_state.attempt_number} attempts"
    ),
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
    before_sleep=lambda retry_state: logger.info(
        f"Retrying fetch_trade_details after {retry_state.attempt_number} attempts"
    ),
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
        # Input validation
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"Invalid symbol: {symbol}")
        if not isinstance(asset, dict) or not all(key in asset for key in ["quantity", "purchase_price", "purchase_time"]):
            raise ValueError(f"Invalid asset data for {symbol}")
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError(f"Invalid sell price {current_price} for {symbol}")
        if not isinstance(sell_slippage, (int, float)):
            raise ValueError(f"Invalid sell_slippage {sell_slippage} for {symbol}")

        if not portfolio_lock.acquire(timeout=5):
            logger.error(f"Timeout acquiring portfolio lock for {symbol}")
            send_alert("Portfolio Lock Failure", f"Timeout acquiring portfolio lock for {symbol}")
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
            logger.debug(f"Created finished trade record for {symbol}")
            logger.info(
                f"Sold {asset['quantity']:.10f} {symbol} at {current_price * (1 - abs(sell_slippage)):.8f} EUR "
                f"(after {sell_slippage:.2f}% slippage and {sell_fee:.2f} fee) for {net_sale_value:.2f} € . Reason: {reason}"
            )
            del portfolio["assets"][symbol]
            low_volatility_assets.discard(symbol)
            negative_momentum_counts.pop(symbol, None)
            logger.debug(f"Updated portfolio state for {symbol}")
        finally:
            portfolio_lock.release()

        try:
            if price_monitor_manager:
                price_monitor_manager.stop(symbol)
            append_to_finished_trades_csv(finished_trade)
            telegram_notifier.notify_sell_trade(finished_trade)
        except Exception as e:
            logger.error(f"Failed to process post-sale actions for {symbol}: {e}", exc_info=True)
            send_alert("Post-Sale Action Failure", f"Failed to process post-sale actions for {symbol}: {e}")
        return finished_trade
    except ValueError as e:
        logger.error(f"Validation error in sell_asset for {symbol}: {e}", exc_info=True)
        send_alert("Sell Asset Failure", f"Validation error in sell_asset for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in sell_asset for {symbol}: {e}", exc_info=True)
        send_alert("Sell Asset Failure", f"Unexpected error in sell_asset for {symbol}: {e}")
        return None

def sell_most_profitable_asset(
    portfolio,
    portfolio_lock,
    percent_changes,
    finished_trades,
    price_monitor_manager=None,
    sell_slippages=None,
):
    """
    Sells the most profitable asset to free up a portfolio slot.
    """
    try:
        if not isinstance(portfolio, dict) or "assets" not in portfolio:
            raise ValueError("Invalid portfolio structure")
        if not isinstance(percent_changes, pd.DataFrame) or "symbol" not in percent_changes.columns:
            raise ValueError("Invalid percent_changes DataFrame")
        if not isinstance(finished_trades, list):
            raise ValueError("finished_trades must be a list")
        if sell_slippages is not None and not isinstance(sell_slippages, dict):
            raise ValueError("sell_slippages must be a dictionary")

        if not portfolio_lock.acquire(timeout=5):
            logger.error("Timeout acquiring portfolio lock")
            send_alert("Portfolio Lock Failure", "Timeout acquiring portfolio lock")
            return None

        try:
            current_time = datetime.utcnow()
            profitable_assets = [
                (symbol, asset)
                for symbol, asset in portfolio["assets"].items()
                if isinstance(asset.get("purchase_time"), datetime)
                and (current_time - asset["purchase_time"]).total_seconds() / 60
                >= config.config.MIN_HOLDING_MINUTES
            ]
            if not profitable_assets:
                logger.info("No profitable assets eligible for sale to free up slot.")
                return None

            max_profit = -float("inf")
            asset_to_sell = None
            for symbol, asset in profitable_assets:
                current_price_series = percent_changes[percent_changes["symbol"] == symbol]["close_price"]
                current_price = None
                if current_price_series.empty:
                    logger.warning(f"No price in percent_changes for {symbol}. Fetching ticker price.")
                    try:
                        current_price = fetch_ticker_price_with_retry(symbol)
                    except Exception as e:
                        logger.error(f"Failed to fetch price for {symbol}: {e}", exc_info=True)
                        continue
                else:
                    current_price = float(current_price_series.iloc[0])
                unrealized_profit = (
                    (current_price - asset["purchase_price"]) / asset["purchase_price"]
                    if asset["purchase_price"] > 0 else 0
                )
                sell_slippage = sell_slippages.get(symbol, config.config.MAX_SLIPPAGE_SELL + 0.1) if sell_slippages else (config.config.MAX_SLIPPAGE_SELL + 0.1)
                if unrealized_profit >= 0.01 and unrealized_profit > max_profit and abs(sell_slippage) <= config.config.MAX_SLIPPAGE_SELL:
                    max_profit = unrealized_profit
                    asset_to_sell = (symbol, asset, current_price, sell_slippage)
            if asset_to_sell is None:
                logger.info("No assets with unrealized profit >= 1% or acceptable slippage to sell.")
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
    except ValueError as e:
        logger.error(f"Validation error in sell_most_profitable_asset: {e}", exc_info=True)
        send_alert("Sell Profitable Asset Failure", f"Validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in sell_most_profitable_asset: {e}", exc_info=True)
        send_alert("Sell Profitable Asset Failure", f"Unexpected error: {e}")
        return None

def save_portfolio():
    """
    Saves the current portfolio state to a JSON file and maintains only the 3 latest backup files atomically.
    """
    try:
        if not isinstance(portfolio, dict) or "cash" not in portfolio or "assets" not in portfolio:
            raise ValueError("Invalid portfolio structure")

        if not portfolio_lock.acquire(timeout=5):
            logger.error("Timeout acquiring portfolio lock")
            send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
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
            if not file_path or not os.path.basename(file_path):
                raise ValueError(f"Invalid PORTFOLIO_FILE path: {file_path}")

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
                    logger.warning(f"Error deleting old backup file {old_file}: {e}", exc_info=True)
        finally:
            portfolio_lock.release()
    except ValueError as e:
        logger.error(f"Validation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}", exc_info=True)
        send_alert("Portfolio Save Failure", f"Validation error: {e}")
    except OSError as e:
        logger.error(f"File operation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}", exc_info=True)
        send_alert("Portfolio Save Failure", f"File operation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}", exc_info=True)
        send_alert("Portfolio Save Failure", f"Unexpected error: {e}")

def manage_portfolio(
    above_threshold_data,
    percent_changes,
    price_monitor_manager,
    order_book_metrics_list=None,
    sell_slippages=None,
):
    """
    Manages the portfolio by processing sell signals, updating assets, and buying new assets.
    """
    try:
        # Input validation
        if not isinstance(above_threshold_data, list):
            raise ValueError("above_threshold_data must be a list")
        if not isinstance(percent_changes, pd.DataFrame) or not {"symbol", "close_price"}.issubset(percent_changes.columns):
            raise ValueError("Invalid percent_changes DataFrame")
        if not price_monitor_manager:
            raise ValueError("price_monitor_manager cannot be None")
        if order_book_metrics_list is None:
            order_book_metrics_list = []
        elif not isinstance(order_book_metrics_list, list):
            raise ValueError("order_book_metrics_list must be a list")
        if sell_slippages is not None and not isinstance(sell_slippages, dict):
            raise ValueError("sell_slippages must be a dictionary")

        # Initialize trading strategy
        strategy = TradingStrategy()

        current_time = datetime.utcnow()
        five_min_ago = current_time - timedelta(minutes=5)
        finished_trades = []
        total_asset_value = 0.0
        skipped_assets = []

        # Calculate sell slippage for all held assets
        sell_slippages = sell_slippages or {}
        for symbol in portfolio.get("assets", {}):
            if symbol not in sell_slippages:
                try:
                    amount_quote = portfolio["assets"][symbol]["quantity"] * portfolio["assets"][symbol]["current_price"]
                    metrics = calculate_order_book_metrics(symbol.replace("/", "-"), amount_quote=amount_quote)
                    if "error" not in metrics and metrics.get("slippage_sell") is not None:
                        sell_slippages[symbol] = metrics["slippage_sell"]
                    else:
                        logger.warning(f"Could not calculate sell slippage for {symbol}. Using default value.")
                        sell_slippages[symbol] = -(config.config.MAX_SLIPPAGE_SELL + 0.1)
                except Exception as e:
                    logger.error(f"Error calculating sell slippage for {symbol}: {e}", exc_info=True)
                    sell_slippages[symbol] = -(config.config.MAX_SLIPPAGE_SELL + 0.1)

        if not portfolio_lock.acquire(timeout=5):
            logger.error("Timeout acquiring portfolio lock")
            send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
            return

        try:
            logger.debug("Acquired portfolio_lock for portfolio management")
            active_monitors = set(price_monitor_manager.running.keys()) if price_monitor_manager.running else set()
            active_assets = set(portfolio.get("assets", {}).keys())
            orphaned_monitors = active_monitors - active_assets
            for symbol in orphaned_monitors:
                logger.warning(f"Stopping orphaned monitor for {symbol} not in portfolio.")
                price_monitor_manager.stop(symbol)

            # Adjust profit targets if portfolio is near threshold
            if (
                len(portfolio.get("assets", {})) >= config.config.ASSET_THRESHOLD
                and above_threshold_data
                and portfolio.get("cash", 0) >= config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
            ):
                profitable_assets = [
                    symbol
                    for symbol, asset in portfolio.get("assets", {}).items()
                    if asset.get("current_price", 0) > asset.get("purchase_price", 0) * 1.01
                    and isinstance(asset.get("purchase_time"), datetime)
                    and (datetime.utcnow() - asset["purchase_time"]).total_seconds() / 60
                    >= config.config.MIN_HOLDING_MINUTES
                ]
                if profitable_assets:
                    for symbol in profitable_assets:
                        portfolio["assets"][symbol]["profit_target"] = min(
                            portfolio["assets"][symbol].get("profit_target", config.config.PROFIT_TARGET),
                            config.config.ADJUSTED_PROFIT_TARGET,
                        )
                        logger.info(
                            f"Adjusted profit target for {symbol} to {config.config.ADJUSTED_PROFIT_TARGET}"
                        )
                else:
                    trade = sell_most_profitable_asset(
                        portfolio,
                        portfolio_lock,
                        percent_changes,
                        finished_trades,
                        price_monitor_manager,
                        sell_slippages,
                    )
                    if trade:
                        finished_trades.append(trade)

            # Process existing assets for sell decisions
            for symbol in list(portfolio.get("assets", {}).keys()):
                asset = portfolio["assets"][symbol]
                if (
                    symbol not in low_volatility_assets
                    and symbol not in price_monitor_manager.running
                ):
                    price_monitor_manager.start(
                        symbol, portfolio, portfolio_lock, percent_changes
                    )

                current_price_series = percent_changes[percent_changes["symbol"] == symbol]["close_price"]
                current_price = None
                if current_price_series.empty:
                    logger.warning(f"No price in percent_changes for {symbol}. Fetching ticker price.")
                    try:
                        current_price = fetch_ticker_price_with_retry(symbol)
                    except Exception as e:
                        logger.error(f"Failed to fetch price for {symbol}: {e}", exc_info=True)
                        skipped_assets.append(symbol)
                        continue
                else:
                    current_price = float(current_price_series.iloc[0])

                asset["current_price"] = current_price
                price_monitor_manager.last_prices[symbol] = current_price
                purchase_price = asset.get("purchase_price", 0)
                if current_price <= 0 or purchase_price <= 0:
                    logger.warning(
                        f"Invalid price for {symbol}: current_price={current_price}, purchase_price={purchase_price}. Using purchase_price."
                    )
                    asset_value = asset["quantity"] * purchase_price
                    total_asset_value += asset_value
                    skipped_assets.append(symbol)
                    continue

                asset_value = asset["quantity"] * current_price
                total_asset_value += asset_value
                highest_price = max(asset.get("highest_price", purchase_price), current_price)
                asset["highest_price"] = highest_price
                holding_minutes = (
                    (current_time - asset["purchase_time"]).total_seconds() / 60
                    if isinstance(asset.get("purchase_time"), datetime)
                    else 0
                )
                symbol_candles = percent_changes[percent_changes["symbol"] == symbol].tail(5)
                atr = (
                    np.mean(symbol_candles["high"] - symbol_candles["low"])
                    if len(symbol_candles) >= 5
                    and "high" in symbol_candles.columns
                    and "low" in symbol_candles.columns
                    else 0
                )
                momentum = (
                    percent_changes[percent_changes["symbol"] == symbol]["percent_change"].iloc[0]
                    if not percent_changes[percent_changes["symbol"] == symbol].empty
                    and "percent_change" in percent_changes.columns
                    else 0
                )
                sell_data = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "purchase_price": purchase_price,
                    "highest_price": highest_price,
                    "holding_minutes": holding_minutes,
                    "atr": atr,
                    "momentum": momentum,
                    "near_threshold": len(portfolio.get("assets", {})) >= config.config.ASSET_THRESHOLD
                }
                sell_signal, sell_results, reason = strategy.evaluate_sell(sell_data)
                sell_slippage = sell_slippages.get(symbol, config.config.MAX_SLIPPAGE_SELL + 0.1)
                if sell_signal:
                    logger.info(
                        f"Evaluation Decision for {symbol}: Selling due to {reason}. "
                        f"Current: {current_price:.4f}, Highest: {highest_price:.4f}, "
                        f"Holding: {holding_minutes:.2f} min, Sell Slippage: {sell_slippage:.2f}%"
                    )
                    trade = sell_asset(
                        symbol,
                        asset,
                        current_price,
                        portfolio,
                        portfolio_lock,
                        finished_trades,
                        reason,
                        price_monitor_manager,
                        sell_slippage,
                    )
                    if trade:
                        finished_trades.append(trade)
                else:
                    logger.info(
                        f"Asset: {symbol}, Current: {current_price:.4f}, "
                        f"Purchase: {purchase_price:.4f}, Quantity: {asset['quantity']:.4f}, "
                        f"Holding: {holding_minutes:.2f} min"
                    )

            # Buy new assets
            for record in above_threshold_data:
                if not isinstance(record, dict) or "symbol" not in record or "close_price" not in record:
                    logger.warning(f"Invalid record in above_threshold_data: {record}")
                    continue
                symbol = record["symbol"]
                metrics = next((m for m in order_book_metrics_list if m.get("market") == symbol.replace("/", "-")), {})
                if not metrics:
                    logger.warning(f"No matching metrics found for {symbol}")
                    continue
                buy_data = {
                    "symbol": symbol,
                    "slippage_buy": metrics.get("slippage_buy", float("inf")),
                    "total_score": metrics.get("total_score", 0.0)
                }
                can_buy, buy_results = strategy.evaluate_buy(buy_data)
                logger.debug(f"Evaluating buy for {symbol}: can_buy={can_buy}, results={buy_results}")
                if (
                    can_buy
                    and symbol not in portfolio.get("assets", {})
                    and portfolio.get("cash", 0) >= config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
                    and len(portfolio.get("assets", {})) < config.config.MAX_ACTIVE_ASSETS
                    and symbol not in low_volatility_assets
                ):
                    close_price = record["close_price"]
                    slippage_buy = metrics.get("slippage_buy", 0.0)
                    purchase_price = close_price * (1 + slippage_buy / 100)
                    allocation = config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
                    buy_fee = allocation * config.config.BUY_FEE
                    net_allocation = allocation - buy_fee
                    quantity = net_allocation / purchase_price if purchase_price > 0 else 0
                    if quantity <= 0:
                        logger.warning(f"Cannot buy {symbol}: Invalid quantity {quantity}")
                        continue
                    actual_cost = quantity * purchase_price
                    try:
                        trade_count, largest_trade_volume_eur = fetch_trade_details_with_retry(
                            symbol, five_min_ago, current_time
                        )
                    except Exception as e:
                        logger.error(f"Failed to fetch trade details for {symbol}: {e}", exc_info=True)
                        continue
                    buy_trade_data = {
                        "Symbol": symbol,
                        "Buy Quantity": f"{quantity:.10f}",
                        "Buy Price": f"{purchase_price:.10f}",
                        "Buy Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Buy Fee": f"{buy_fee:.2f}",
                        "Buy Slippage": f"{slippage_buy:.2f}%",
                        "Allocation": f"{allocation:.2f}",
                        "Actual Cost": f"{actual_cost:.2f}",
                        "Trade Count": trade_count,
                        "Largest Trade Volume EUR": f"{largest_trade_volume_eur:.2f}",
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
                        "profit_target": config.config.PROFIT_TARGET,
                        "original_profit_target": config.config.PROFIT_TARGET,
                        "sell_price": purchase_price * (1 + config.config.PROFIT_TARGET),
                    }
                    portfolio["cash"] -= allocation
                    total_asset_value += (quantity * close_price)
                    logger.info(
                        f"Bought {quantity:.4f} {symbol} at {purchase_price:.4f} EUR (close {close_price:.4f}) "
                        f"for {actual_cost:.2f} EUR (after {slippage_buy:.2f}% slippage and {buy_fee:.2f} fee)"
                    )
                    for metrics in order_book_metrics_list:
                        if metrics.get("market") == symbol.replace("/", "-"):
                            metrics["bought"] = True
                    price_monitor_manager.start(
                        symbol, portfolio, portfolio_lock, percent_changes
                    )
                else:
                    logger.info(
                        f"Cannot buy {symbol}: Conditions not met - "
                        f"can_buy={can_buy}, "
                        f"in_portfolio={symbol in portfolio.get('assets', {})}, "
                        f"sufficient_cash={portfolio.get('cash', 0) >= config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE}, "
                        f"asset_limit={len(portfolio.get('assets', {})) < config.config.MAX_ACTIVE_ASSETS}, "
                        f"low_volatility={symbol in low_volatility_assets}"
                    )
                    if len(portfolio.get("assets", {})) >= config.config.MAX_ACTIVE_ASSETS:
                        trade = sell_most_profitable_asset(
                            portfolio,
                            portfolio_lock,
                            percent_changes,
                            finished_trades,
                            price_monitor_manager,
                            sell_slippages,
                        )
                        if trade:
                            finished_trades.append(trade)
        finally:
            portfolio_lock.release()

        if order_book_metrics_list:
            append_to_order_book_metrics_csv(order_book_metrics_list)

        total_portfolio_value = portfolio.get("cash", 0) + total_asset_value
        portfolio_values.append({"timestamp": datetime.utcnow().isoformat(), "portfolio_value": total_portfolio_value})
        if skipped_assets:
            logger.warning(
                f"Portfolio value may be inaccurate due to missing prices for: {', '.join(skipped_assets)}"
            )
    except ValueError as e:
        logger.error(f"Validation error in portfolio management: {e}", exc_info=True)
        send_alert("Portfolio Management Failure", f"Validation error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in portfolio management: {e}", exc_info=True)
        send_alert("Portfolio Management Failure", f"Unexpected error: {e}")

def send_alert(subject, message):
    telegram_notifier.notify_error(subject, message)