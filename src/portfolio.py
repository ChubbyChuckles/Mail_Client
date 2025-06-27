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
from .bitvavo_order_metrics import calculate_order_book_metrics, fetch_order_book_with_retry  # Import for slippage calculation

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
    sell_slippage=0.0,  # Added parameter for sell slippage
):
    """
    Sells a specified asset and updates the portfolio.

    Args:
        symbol (str): Asset symbol.
        asset (dict): Asset details.
        current_price (float): Current price of the asset.
        portfolio (dict): Portfolio state.
        portfolio_lock (Lock): Thread lock for portfolio updates.
        finished_trades (list): List to append finished trade record.
        reason (str): Reason for selling.
        price_monitor_manager: Instance of PriceMonitorManager.
        sell_slippage (float): Slippage percentage for selling (default: 0.0).

    Returns:
        dict or None: Finished trade details if sold, else None.

    Raises:
        ValueError: If inputs are invalid.
    """
    try:
        # Input validation
        if not isinstance(symbol, str) or not symbol:
            raise ValueError(f"Invalid symbol: {symbol}")
        if not isinstance(asset, dict) or not all(key in asset for key in ["quantity", "purchase_price", "purchase_time"]):
            raise ValueError(f"Invalid asset data for {symbol}")
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            raise ValueError(f"Invalid sell price {current_price} for {symbol}")
        if not isinstance(sell_slippage, (int, float)) or sell_slippage < 0:
            raise ValueError(f"Invalid sell_slippage {sell_slippage} for {symbol}")

        if not portfolio_lock.acquire(timeout=5):
            logger.error(f"Timeout acquiring portfolio lock for {symbol}")
            send_alert("Portfolio Lock Failure", f"Timeout acquiring portfolio lock for {symbol}")
            return None

        try:
            logger.debug(f"Starting sell process for {symbol}: {reason}")
            sale_value = asset["quantity"] * current_price * (1 - sell_slippage)  # Adjust for slippage
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
                "Sell Price": f"{current_price * (1 - sell_slippage):.10f}",  # Adjust for slippage
                "Sell Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Sell Fee": f"{sell_fee:.2f}",
                "Sell Slippage": f"{sell_slippage*100:.2f}%",  # Record slippage percentage
                "Profit/Loss": f"{profit_loss:.2f}",
                "Reason": reason,
            }
            portfolio["cash"] += net_sale_value
            finished_trades.append(finished_trade)
            logger.debug(f"Created finished trade record for {symbol}")
            logger.info(
                f"Sold {asset['quantity']:.10f} {symbol} at {current_price * (1 - sell_slippage):.8f} EUR "
                f"(after {sell_slippage*100:.2f}% slippage and {sell_fee:.2f} fee). Reason: {reason}"
            )
            del portfolio["assets"][symbol]
            low_volatility_assets.discard(symbol)
            negative_momentum_counts.pop(symbol, None)
            logger.debug(f"Updated portfolio state for {symbol}")
        finally:
            portfolio_lock.release()

        # Post-sale actions
        try:
            if price_monitor_manager:
                price_monitor_manager.stop(symbol)
            else:
                logger.warning(f"Price monitor manager is None for {symbol}. Cannot stop monitoring.")
            append_to_finished_trades_csv(finished_trade)
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
    sell_slippages=None,  # Added parameter for sell slippages
):
    """
    Sells the most profitable asset to free up a portfolio slot.

    Args:
        portfolio (dict): Portfolio dictionary.
        portfolio_lock (Lock): Thread lock for portfolio updates.
        percent_changes (pandas.DataFrame): DataFrame with price changes.
        finished_trades (list): List to append finished trade record.
        price_monitor_manager: Instance of PriceMonitorManager.
        sell_slippages (dict): Dictionary of sell slippages for each asset (default: None).

    Returns:
        dict or None: Finished trade details if sold, else None.

    Raises:
        ValueError: If inputs are invalid.
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
                # Check slippage for selling decision
                sell_slippage = sell_slippages.get(symbol, abs(config.config.MAX_SLIPPAGE_SELL) + 0.1) if sell_slippages else (abs(config.config.MAX_SLIPPAGE_SELL) + 0.1)
                if unrealized_profit >= 0.01 and unrealized_profit > max_profit and sell_slippage <= abs(config.config.MAX_SLIPPAGE_SELL):
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

    Raises:
        ValueError: If portfolio data is invalid.
        OSError: For file operation errors.
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
            logger.info(f"Saved portfolio to {file_path}")

            # Save to backup file
            backup_file = f"{file_path}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(portfolio_copy, temp_file, indent=4)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            shutil.move(temp_file.name, backup_file)
            logger.info(f"Saved portfolio backup to {backup_file}")

            # Manage backup files: keep only the 3 latest
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
    sell_slippages=None,  # Added parameter for sell slippages
):
    """
    Manages the portfolio by processing sell signals, updating assets, and buying new assets.

    Args:
        above_threshold_data (list): List of assets meeting price/volume thresholds.
        percent_changes (pandas.DataFrame): DataFrame with price changes and OHLCV data.
        price_monitor_manager: Instance of PriceMonitorManager.
        order_book_metrics_list (list): List of order book metrics to update with buy decisions.
        sell_slippages (dict): Dictionary of sell slippages for each asset (default: None).

    Raises:
        ValueError: If inputs are invalid.
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

        current_time = datetime.utcnow()
        five_min_ago = current_time - timedelta(minutes=5)
        finished_trades = []
        total_asset_value = 0.0
        skipped_assets = []

        # Calculate sell slippage for all held assets
        sell_slippages = sell_slippages or {}
        for symbol in portfolio.get("assets", {}):
            if symbol not in sell_slippages:  # Only calculate if not provided (e.g., not backtesting)
                try:
                    amount_quote = portfolio["assets"][symbol]["quantity"] * portfolio["assets"][symbol]["current_price"]
                    metrics = calculate_order_book_metrics(symbol.replace("/", "-"), amount_quote=amount_quote)
                    if "error" not in metrics and metrics.get("slippage_sell") is not None:
                        sell_slippages[symbol] = metrics["slippage_sell"] / 100  # Convert percentage to decimal
                    else:
                        logger.warning(f"Could not calculate sell slippage for {symbol}. Using default value.")
                        sell_slippages[symbol] = abs(config.config.MAX_SLIPPAGE_SELL) + 0.1  # Prevent selling
                except Exception as e:
                    logger.error(f"Error calculating sell slippage for {symbol}: {e}", exc_info=True)
                    sell_slippages[symbol] = abs(config.config.MAX_SLIPPAGE_SELL) + 0.1  # Prevent selling

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
                        sell_slippages,  # Pass sell slippages
                    )
                    if trade:
                        finished_trades.append(trade)

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
                trailing_stop = (
                    (
                        config.config.TRAILING_STOP_FACTOR_EARLY
                        if holding_minutes < 15
                        else config.config.TRAILING_STOP_FACTOR
                    )
                    * atr / purchase_price
                    if atr > 0 and purchase_price > 0
                    else 0.05
                )
                profit_target = (
                    max(0.015, min(0.05, 1.2 * atr / purchase_price))
                    if atr > 0 and purchase_price > 0
                    else asset.get("profit_target", 0.015)
                )
                if (
                    len(portfolio.get("assets", {})) >= config.config.ASSET_THRESHOLD
                    and current_price > purchase_price * 1.01
                ):
                    profit_target = min(profit_target, config.config.ADJUSTED_PROFIT_TARGET)
                asset["profit_target"] = profit_target
                ema_period = calculate_dynamic_ema_period(
                    holding_minutes,
                    config.config.TIME_STOP_MINUTES,
                    len(portfolio.get("assets", {})),
                    config.config.ASSET_THRESHOLD,
                )
                ema_dynamic = (
                    calculate_ema(symbol_candles["close"].values, ema_period)
                    if len(symbol_candles) >= ema_period
                    else current_price
                )
                momentum = (
                    percent_changes[percent_changes["symbol"] == symbol]["percent_change"].iloc[0]
                    if not percent_changes[percent_changes["symbol"] == symbol].empty
                    and "percent_change" in percent_changes.columns
                    else 0
                )
                if momentum < config.config.MOMENTUM_THRESHOLD:
                    negative_momentum_counts[symbol] = negative_momentum_counts.get(symbol, 0) + 1
                else:
                    negative_momentum_counts[symbol] = 0
                unrealized_profit = (
                    (current_price - purchase_price) / purchase_price
                    if purchase_price > 0
                    else 0
                )
                trailing_loss = (
                    (highest_price - current_price) / highest_price
                    if highest_price > purchase_price
                    else 0
                )
                sell_prices = []
                profit_target_price = purchase_price * (1 + profit_target)
                sell_prices.append(profit_target_price)
                if highest_price > purchase_price:
                    trailing_stop_price = highest_price * (1 - trailing_stop)
                    if trailing_stop_price > purchase_price:
                        sell_prices.append(trailing_stop_price)
                asset["sell_price"] = min(sell_prices) if sell_prices else profit_target_price
                catastrophic_loss = (
                    unrealized_profit <= config.config.CAT_LOSS_THRESHOLD
                    and abs(unrealized_profit) > 2 * atr / purchase_price
                    if atr > 0 and purchase_price > 0
                    else False
                )
                time_stop = (
                    holding_minutes >= config.config.TIME_STOP_MINUTES and unrealized_profit < 0
                )
                multiplied_profit_target = (
                    unrealized_profit >= config.config.PROFIT_TARGET_MULTIPLIER * profit_target
                )
                regular_sell_signal = (
                    holding_minutes >= config.config.MIN_HOLDING_MINUTES
                    and (
                        trailing_loss >= trailing_stop
                        or unrealized_profit >= profit_target
                        or negative_momentum_counts.get(symbol, 0)
                        >= config.config.MOMENTUM_CONFIRM_MINUTES
                    )
                )
                sell_slippage = sell_slippages.get(symbol, abs(config.config.MAX_SLIPPAGE_SELL) + 0.1)
                slippage_ok = sell_slippage <= abs(config.config.MAX_SLIPPAGE_SELL)
                sell_signal = (
                    (multiplied_profit_target or regular_sell_signal or catastrophic_loss or time_stop)
                    and slippage_ok
                )
                if sell_signal:
                    reason = (
                        "Catastrophic loss" if catastrophic_loss else
                        "Time stop" if time_stop else
                        f"Multiplied profit target ({config.config.PROFIT_TARGET_MULTIPLIER}x = {(config.config.PROFIT_TARGET_MULTIPLIER * profit_target)*100:.1f}%)"
                        if multiplied_profit_target else
                        "Trailing stop" if trailing_loss >= trailing_stop else
                        f"Dynamic profit target ({profit_target*100:.1f}%)" if unrealized_profit >= profit_target else
                        "Negative momentum"
                    )
                    logger.info(
                        f"Evaluation Decision for {symbol}: Selling due to {reason}. "
                        f"Current: {current_price:.4f}, Highest: {highest_price:.4f}, "
                        f"Trailing Loss: {trailing_loss:.4f}, ATR Stop: {trailing_stop:.4f}, "
                        f"Profit/Loss: {unrealized_profit:.4f}, Profit Target: {profit_target:.4f}, "
                        f"EMA_{ema_period}: {ema_dynamic:.2f}, "
                        f"Holding: {holding_minutes:.2f} min, Neg Momentum Count: {negative_momentum_counts.get(symbol, 0)}, "
                        f"Sell Slippage: {sell_slippage*100:.2f}%"
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
                    if not slippage_ok:
                        logger.info(
                            f"Delaying sell for {symbol}: Sell slippage {sell_slippage*100:.2f}% exceeds threshold {abs(config.config.MAX_SLIPPAGE_SELL)*100:.2f}%"
                        )
                    # Log asset statistics including sell slippage
                    logger.info(
                        f"Asset: {symbol}, "
                        f"Current: {current_price:.4f}, "
                        f"Purchase: {purchase_price:.4f}, "
                        f"Quantity: {asset['quantity']:.4f}, "
                        f"Unrealized P/L: {unrealized_profit*100:.2f}%, "
                        f"Sell Slippage: {sell_slippage*100:.2f}%, "
                        f"Holding: {holding_minutes:.2f} min"
                    )

            # Buy new assets and update order book metrics
            for record in above_threshold_data:
                if not isinstance(record, dict) or "symbol" not in record or "close_price" not in record:
                    logger.warning(f"Invalid record in above_threshold_data: {record}")
                    continue
                symbol = record["symbol"]
                total_score = 0.0
                slippage_buy = 0.0
                for metrics in order_book_metrics_list:
                    if metrics.get("market") == symbol.replace("/", "-"):
                        total_score = metrics.get("total_score", 0)
                        slippage_buy = metrics.get("slippage_buy", float("inf"))
                if (
                    symbol not in portfolio.get("assets", {})
                    and portfolio.get("cash", 0) >= config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
                    and len(portfolio.get("assets", {})) < config.config.MAX_ACTIVE_ASSETS
                    and symbol not in low_volatility_assets
                    and total_score >= config.config.MIN_TOTAL_SCORE
                    and slippage_buy <= config.config.MAX_SLIPPAGE_BUY
                ):
                    purchase_price = record["close_price"]
                    allocation = config.config.PORTFOLIO_VALUE * config.config.ALLOCATION_PER_TRADE
                    buy_fee = allocation * config.config.BUY_FEE
                    net_allocation = allocation - buy_fee
                    quantity = net_allocation / purchase_price if purchase_price > 0 else 0
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
                        "Allocation": f"{allocation:.2f}",
                        "Trade Count": trade_count,
                        "Largest Trade Volume EUR": f"{largest_trade_volume_eur:.2f}",
                    }
                    append_to_buy_trades_csv(buy_trade_data)
                    portfolio["assets"][symbol] = {
                        "quantity": quantity,
                        "purchase_price": purchase_price,
                        "purchase_time": current_time,
                        "highest_price": purchase_price,
                        "current_price": purchase_price,
                        "profit_target": config.config.PROFIT_TARGET,
                        "original_profit_target": config.config.PROFIT_TARGET,
                        "sell_price": purchase_price * (1 + config.config.PROFIT_TARGET),
                    }
                    portfolio["cash"] -= allocation
                    total_asset_value += net_allocation
                    logger.info(
                        f"Bought {quantity:.4f} {symbol} at {purchase_price:.4f} EUR for {net_allocation:.2f} EUR "
                        f"(after {buy_fee:.2f} fee), Trade Count: {trade_count}, Largest Trade Volume: €{largest_trade_volume_eur:.2f}"
                    )
                    for metrics in order_book_metrics_list:
                        if metrics.get("market") == symbol.replace("/", "-"):
                            metrics["bought"] = True
                    price_monitor_manager.start(
                        symbol, portfolio, portfolio_lock, percent_changes
                    )
                elif len(portfolio.get("assets", {})) >= config.config.MAX_ACTIVE_ASSETS:
                    logger.warning(
                        f"Cannot buy {symbol}: Maximum active assets ({config.config.MAX_ACTIVE_ASSETS}) reached."
                    )
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
                elif slippage_buy >= config.config.MAX_SLIPPAGE_BUY:
                    logger.info(
                        f"Cannot buy {symbol}: Slippage Buy {slippage_buy:.2f}% is above threshold ({config.config.MAX_SLIPPAGE_BUY:.2f}%)."
                    )
                elif total_score <= config.config.MIN_TOTAL_SCORE:
                    logger.info(
                        f"Cannot buy {symbol}: Total Score {total_score:.2f} is below threshold {config.config.MIN_TOTAL_SCORE:.2f}."
                    )
                elif symbol in portfolio.get("assets", {}):
                    logger.debug(f"Cannot buy {symbol}: Already owned.")
                elif symbol in low_volatility_assets:
                    logger.debug(f"Cannot buy {symbol}: Marked as low volatility.")
                else:
                    logger.warning(
                        f"Cannot buy {symbol}: Insufficient cash ({portfolio.get('cash', 0):.2f} EUR)."
                    )
        finally:
            portfolio_lock.release()
    except ValueError as e:
        logger.error(f"Validation error in portfolio management: {e}", exc_info=True)
        send_alert("Portfolio Management Failure", f"Validation error: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error in portfolio management: {e}", exc_info=True)
        send_alert("Portfolio Management Failure", f"Unexpected error: {e}")
        return

    # Save order book metrics to CSV
    try:
        if order_book_metrics_list:
            append_to_order_book_metrics_csv(order_book_metrics_list)
    except Exception as e:
        logger.error(f"Error saving order book metrics: {e}", exc_info=True)
        send_alert("Order Book Metrics Save Failure", f"Error saving order book metrics: {e}")

    total_portfolio_value = portfolio.get("cash", 0) + total_asset_value
    if skipped_assets:
        logger.warning(
            f"Portfolio value may be inaccurate due to missing prices for: {', '.join(skipped_assets)}"
        )
    logger.info(
        f"Portfolio: Cash: {portfolio.get('cash', 0):.2f} EUR, Assets: {len(portfolio.get('assets', {}))}, Total Value: {total_portfolio_value:.2f} EUR"
    )

def send_alert(subject, message):
    """
    Sends an alert for critical errors (placeholder).

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(f"ALERT: {subject} - {message}")