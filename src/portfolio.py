# trading_bot/src/portfolio.py
import glob
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import (ADJUSTED_PROFIT_TARGET, ALLOCATION_PER_TRADE,
                     ASSET_THRESHOLD, BUY_FEE, CAT_LOSS_THRESHOLD,
                     FINISHED_TRADES_CSV, MAX_ACTIVE_ASSETS,
                     MIN_HOLDING_MINUTES, MOMENTUM_CONFIRM_MINUTES,
                     MOMENTUM_THRESHOLD, PORTFOLIO_FILE, PORTFOLIO_VALUE,
                     PROFIT_TARGET, PROFIT_TARGET_MULTIPLIER, SELL_FEE,
                     TIME_STOP_MINUTES, TRAILING_STOP_FACTOR,
                     TRAILING_STOP_FACTOR_EARLY, logger)
from .exchange import fetch_ticker_price, fetch_trade_details
from .state import (low_volatility_assets, negative_momentum_counts, portfolio,
                    portfolio_lock)
from .utils import (append_to_buy_trades_csv, append_to_finished_trades_csv,
                    append_to_order_book_metrics_csv,
                    calculate_dynamic_ema_period, calculate_ema)


def sell_asset(
    symbol,
    asset,
    current_price,
    portfolio,
    portfolio_lock,
    finished_trades,
    reason,
    price_monitor_manager,
):
    """
    Sells a specified asset and updates the portfolio.
    """
    try:
        logger.debug(f"Starting sell process for {symbol}: {reason}")
        finished_trade = None
        with portfolio_lock:
            logger.debug(f"Selling {symbol}: {reason}")
            if current_price <= 0:
                logger.error(
                    f"Invalid sell price {current_price} for {symbol}. Cannot sell."
                )
                return None
            sale_value = asset["quantity"] * current_price
            sell_fee = sale_value * SELL_FEE
            net_sale_value = sale_value - sell_fee
            buy_value = asset["quantity"] * asset["purchase_price"]
            buy_fee = buy_value * BUY_FEE
            profit_loss = net_sale_value - (buy_value + buy_fee)
            portfolio["cash"] += net_sale_value
            finished_trade = {
                "Symbol": symbol,
                "Buy Quantity": f"{asset['quantity']:.10f}",
                "Buy Price": f"{asset['purchase_price']:.10f}",
                "Buy Time": asset["purchase_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "Buy Fee": f"{buy_fee:.2f}",
                "Sell Quantity": f"{asset['quantity']:.10f}",
                "Sell Price": f"{current_price:.10f}",
                "Sell Time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Sell Fee": f"{sell_fee:.2f}",
                "Profit/Loss": f"{profit_loss:.2f}",
                "Reason": reason,
            }
            finished_trades.append(finished_trade)
            logger.debug(f"Created finished trade record for {symbol}")
            logger.info(
                f"Sold {asset['quantity']:.10f} {symbol} at {current_price:.8f} EUR for {net_sale_value:.2f} EUR "
                f"(after {sell_fee:.2f} fee). Reason: {reason}"
            )
            del portfolio["assets"][symbol]
            low_volatility_assets.discard(symbol)
            negative_momentum_counts.pop(symbol, None)
            logger.debug(f"Updated portfolio state for {symbol}")

        # Move post-sale actions outside the lock
        try:
            if price_monitor_manager:
                price_monitor_manager.stop(symbol)
            else:
                logger.warning(
                    f"Price monitor manager is None for {symbol}. Cannot stop monitoring."
                )
            append_to_finished_trades_csv(finished_trade)
        except Exception as e:
            logger.error(f"Failed to process post-sale actions for {symbol}: {e}")
        return finished_trade
    except Exception as e:
        logger.error(f"Critical error in sell_asset for {symbol}: {e}")
        raise


def sell_most_profitable_asset(
    portfolio,
    portfolio_lock,
    percent_changes,
    finished_trades,
    price_monitor_manager=None,
):
    """
    Sells the most profitable asset to free up a portfolio slot.

    Args:
        portfolio (dict): Portfolio dictionary.
        portfolio_lock (Lock): Thread lock for portfolio updates.
        percent_changes (pandas.DataFrame): DataFrame with price changes.
        finished_trades (list): List to append finished trade record.

    Returns:
        dict or None: Finished trade details if sold, else None.
    """
    with portfolio_lock:
        current_time = datetime.utcnow()
        profitable_assets = [
            (symbol, asset)
            for symbol, asset in portfolio["assets"].items()
            if (current_time - asset["purchase_time"]).total_seconds() / 60
            >= MIN_HOLDING_MINUTES
        ]
        if not profitable_assets:
            logger.info("No profitable assets eligible for sale to free up slot.")
            return None
        max_profit = -float("inf")
        asset_to_sell = None
        for symbol, asset in profitable_assets:
            current_price = percent_changes[percent_changes["symbol"] == symbol][
                "close_price"
            ]
            if current_price.empty:
                current_price = fetch_ticker_price(symbol)
                if current_price is None:
                    continue
                current_price = float(current_price)
            else:
                current_price = float(current_price.iloc[0])
            unrealized_profit = (current_price - asset["purchase_price"]) / asset[
                "purchase_price"
            ]
            if unrealized_profit >= 0.01 and unrealized_profit > max_profit:
                max_profit = unrealized_profit
                asset_to_sell = (symbol, asset, current_price)
        if asset_to_sell is None:
            logger.info("No assets with unrealized profit >= 1% to sell.")
            return None
        symbol, asset, current_price = asset_to_sell
        return sell_asset(
            symbol,
            asset,
            current_price,
            portfolio,
            portfolio_lock,
            finished_trades,
            "Sold to free up slot for new buy",
            price_monitor_manager=price_monitor_manager,
        )


def save_portfolio():
    """
    Saves the current portfolio state to a JSON file and maintains only the 3 latest backup files.
    """
    try:
        with portfolio_lock:
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
        # Save to primary file
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio_copy, f, indent=4)

        # Save to backup file
        backup_file = (
            f"{PORTFOLIO_FILE}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        with open(backup_file, "w") as f:
            json.dump(portfolio_copy, f, indent=4)
        logger.info(f"Saved portfolio to {PORTFOLIO_FILE} and backup {backup_file}")

        # Manage backup files: keep only the 3 latest
        backup_files = glob.glob(f"{PORTFOLIO_FILE}.backup_*")
        backup_files.sort(key=lambda x: x.split("backup_")[-1], reverse=True)
        for old_file in backup_files[3:]:
            try:
                os.remove(old_file)
                logger.debug(f"Deleted old backup file: {old_file}")
            except Exception as e:
                logger.error(
                    f"Error deleting old backup file {old_file}: {e}", exc_info=True
                )
    except Exception as e:
        logger.error(f"Error saving portfolio to {PORTFOLIO_FILE}: {e}", exc_info=True)


def manage_portfolio(
    above_threshold_data,
    percent_changes,
    price_monitor_manager,
    order_book_metrics_list=None,
):
    """
    Manages the portfolio by processing sell signals, updating assets, and buying new assets.

    Args:
        above_threshold_data (list): List of assets meeting price/volume thresholds.
        percent_changes (pandas.DataFrame): DataFrame with price changes and OHLCV data.
        price_monitor_manager: Instance of PriceMonitorManager.
        order_book_metrics_list (list): List of order book metrics to update with buy decisions.
    """
    if order_book_metrics_list is None:
        order_book_metrics_list = []

    current_time = datetime.utcnow()
    five_min_ago = current_time - timedelta(minutes=5)
    logger.info(
        f"Portfolio before update: Cash = {portfolio['cash']:.2f} EUR, Assets = {len(portfolio['assets'])}"
    )
    finished_trades = []
    total_asset_value = 0.0
    skipped_assets = []

    # Process existing assets
    try:
        with portfolio_lock:
            logger.debug("Acquired portfolio_lock for portfolio management")
            active_monitors = set(price_monitor_manager.running.keys())
            active_assets = set(portfolio["assets"].keys())
            orphaned_monitors = active_monitors - active_assets
            for symbol in orphaned_monitors:
                logger.warning(
                    f"Stopping orphaned monitor for {symbol} not in portfolio."
                )
                price_monitor_manager.stop(symbol)

            # Adjust profit targets if portfolio is near threshold
            if (
                len(portfolio["assets"]) >= ASSET_THRESHOLD
                and above_threshold_data
                and portfolio["cash"] >= PORTFOLIO_VALUE * ALLOCATION_PER_TRADE
            ):
                profitable_assets = [
                    symbol
                    for symbol, asset in portfolio["assets"].items()
                    if asset["current_price"] > asset["purchase_price"] * 1.01
                    and (datetime.utcnow() - asset["purchase_time"]).total_seconds()
                    / 60
                    >= MIN_HOLDING_MINUTES
                ]
                if profitable_assets:
                    for symbol in profitable_assets:
                        portfolio["assets"][symbol]["profit_target"] = min(
                            portfolio["assets"][symbol]["profit_target"],
                            ADJUSTED_PROFIT_TARGET,
                        )
                        logger.info(
                            f"Adjusted profit target for {symbol} to {ADJUSTED_PROFIT_TARGET}"
                        )
                else:
                    finished_trades.append(
                        sell_most_profitable_asset(
                            portfolio,
                            portfolio_lock,
                            percent_changes,
                            finished_trades,
                            price_monitor_manager=price_monitor_manager,
                        )
                    )

            for symbol in list(portfolio["assets"].keys()):
                if (
                    symbol not in low_volatility_assets
                    and symbol not in price_monitor_manager.running
                ):
                    price_monitor_manager.start(
                        symbol, portfolio, portfolio_lock, percent_changes
                    )
                asset = portfolio["assets"][symbol]
                current_price = percent_changes[percent_changes["symbol"] == symbol][
                    "close_price"
                ]
                if current_price.empty:
                    logger.warning(
                        f"No current price in percent_changes for {symbol}. Fetching ticker price."
                    )
                    current_price = fetch_ticker_price(symbol)
                    if current_price is None:
                        logger.error(
                            f"Failed to fetch price for {symbol}. Using purchase_price for portfolio value."
                        )
                        skipped_assets.append(symbol)
                        continue
                    current_price = float(current_price)
                else:
                    current_price = float(current_price.iloc[0])
                asset["current_price"] = current_price
                price_monitor_manager.last_prices[symbol] = current_price
                if current_price > 0:
                    asset_value = asset["quantity"] * current_price
                    total_asset_value += asset_value
                else:
                    logger.warning(
                        f"Invalid current_price {current_price} for {symbol}. Using purchase_price."
                    )
                    asset_value = asset["quantity"] * asset["purchase_price"]
                    total_asset_value += asset_value
                purchase_price = asset["purchase_price"]
                highest_price = max(asset["highest_price"], current_price)
                asset["highest_price"] = highest_price
                holding_minutes = (
                    current_time - asset["purchase_time"]
                ).total_seconds() / 60
                symbol_candles = percent_changes[
                    percent_changes["symbol"] == symbol
                ].tail(5)
                atr = (
                    np.mean(symbol_candles["high"] - symbol_candles["low"])
                    if len(symbol_candles) >= 5
                    and "high" in symbol_candles
                    and "low" in symbol_candles
                    else 0
                )
                trailing_stop = (
                    (
                        TRAILING_STOP_FACTOR_EARLY
                        if holding_minutes < 15
                        else TRAILING_STOP_FACTOR
                    )
                    * atr
                    / purchase_price
                    if atr > 0
                    else 0.05
                )
                profit_target = (
                    max(0.015, min(0.05, 1.2 * atr / purchase_price))
                    if atr > 0
                    else asset.get("profit_target", 0.015)
                )
                if (
                    len(portfolio["assets"]) >= ASSET_THRESHOLD
                    and current_price > purchase_price * 1.01
                ):
                    profit_target = min(profit_target, ADJUSTED_PROFIT_TARGET)
                asset["profit_target"] = profit_target
                ema_period = calculate_dynamic_ema_period(
                    holding_minutes,
                    TIME_STOP_MINUTES,
                    len(portfolio["assets"]),
                    ASSET_THRESHOLD,
                )
                ema_dynamic = (
                    calculate_ema(symbol_candles["close"].values, ema_period)
                    if len(symbol_candles) >= ema_period
                    else current_price
                )
                momentum = (
                    percent_changes[percent_changes["symbol"] == symbol][
                        "percent_change"
                    ].iloc[0]
                    if not percent_changes[percent_changes["symbol"] == symbol].empty
                    else 0
                )
                if momentum < MOMENTUM_THRESHOLD:
                    negative_momentum_counts[symbol] = (
                        negative_momentum_counts.get(symbol, 0) + 1
                    )
                else:
                    negative_momentum_counts[symbol] = 0
                unrealized_profit = 0
                if purchase_price > 0:
                    unrealized_profit = (
                        current_price - purchase_price
                    ) / purchase_price
                else:
                    logger.error(
                        f"Purchase price for {symbol} is zero, skipping unrealized profit calculation."
                    )
                    skipped_assets.append(symbol)
                    continue
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
                asset["sell_price"] = (
                    min(sell_prices) if sell_prices else profit_target_price
                )
                catastrophic_loss = (
                    unrealized_profit <= CAT_LOSS_THRESHOLD
                    and abs(unrealized_profit) > 2 * atr / purchase_price
                    if atr > 0
                    else False
                )
                time_stop = (
                    holding_minutes >= TIME_STOP_MINUTES and unrealized_profit < 0
                )
                multiplied_profit_target = (
                    unrealized_profit >= PROFIT_TARGET_MULTIPLIER * profit_target
                )
                regular_sell_signal = holding_minutes >= MIN_HOLDING_MINUTES and (
                    trailing_loss >= trailing_stop
                    or unrealized_profit >= profit_target
                    or negative_momentum_counts.get(symbol, 0)
                    >= MOMENTUM_CONFIRM_MINUTES
                )
                sell_signal = (
                    multiplied_profit_target
                    or regular_sell_signal
                    or catastrophic_loss
                    or time_stop
                )
                if sell_signal:
                    reason = (
                        "Catastrophic loss"
                        if catastrophic_loss
                        else (
                            "Time stop"
                            if time_stop
                            else (
                                f"Multiplied profit target ({PROFIT_TARGET_MULTIPLIER}x = {(PROFIT_TARGET_MULTIPLIER * profit_target)*100:.1f}%)"
                                if multiplied_profit_target
                                else (
                                    "Trailing stop"
                                    if trailing_loss >= trailing_stop
                                    else (
                                        f"Dynamic profit target ({profit_target*100:.1f}%)"
                                        if unrealized_profit >= profit_target
                                        else "Negative momentum"
                                    )
                                )
                            )
                        )
                    )
                    logger.info(
                        f"Evaluation Decision for {symbol}: Selling due to {reason}. "
                        f"Current: {current_price:.4f}, Highest: {highest_price:.4f}, "
                        f"Trailing Loss: {trailing_loss:.4f}, ATR Stop: {trailing_stop:.4f}, "
                        f"Profit/Loss: {unrealized_profit:.4f}, Profit Target: {profit_target:.4f}, "
                        f"EMA_{ema_period}: {ema_dynamic:.2f}, "
                        f"Holding: {holding_minutes:.2f} min, Neg Momentum Count: {negative_momentum_counts.get(symbol, 0)}"
                    )
                    finished_trades.append(
                        sell_asset(
                            symbol,
                            asset,
                            current_price,
                            portfolio,
                            portfolio_lock,
                            finished_trades,
                            reason,
                            price_monitor_manager=price_monitor_manager,
                        )
                    )

            # Buy new assets and update order book metrics
            for record in above_threshold_data:
                symbol = record["symbol"]
                if (
                    symbol not in portfolio["assets"]
                    and portfolio["cash"] >= PORTFOLIO_VALUE * ALLOCATION_PER_TRADE
                    and len(portfolio["assets"]) < MAX_ACTIVE_ASSETS
                    and symbol not in low_volatility_assets
                ):
                    purchase_price = record["close_price"]
                    allocation = PORTFOLIO_VALUE * ALLOCATION_PER_TRADE
                    buy_fee = allocation * BUY_FEE
                    net_allocation = allocation - buy_fee
                    quantity = net_allocation / purchase_price
                    trade_count, largest_trade_volume_eur = fetch_trade_details(
                        symbol, five_min_ago, current_time
                    )
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
                        "profit_target": PROFIT_TARGET,
                        "original_profit_target": PROFIT_TARGET,
                        "sell_price": purchase_price * (1 + PROFIT_TARGET),
                    }
                    portfolio["cash"] -= allocation
                    total_asset_value += net_allocation
                    logger.info(
                        f"Bought {quantity:.4f} {symbol} at {purchase_price:.4f} EUR for {net_allocation:.2f} EUR (after {buy_fee:.2f} fee), "
                        f"Trade Count: {trade_count}, Largest Trade Volume: €{largest_trade_volume_eur:.2f}"
                    )
                    # Mark as bought in order_book_metrics_list
                    for metrics in order_book_metrics_list:
                        if metrics.get("market") == symbol.replace("/", "-"):
                            metrics["bought"] = True
                    price_monitor_manager.start(
                        symbol, portfolio, portfolio_lock, percent_changes
                    )
                elif len(portfolio["assets"]) >= MAX_ACTIVE_ASSETS:
                    logger.warning(
                        f"Cannot buy {symbol}: Maximum active assets ({MAX_ACTIVE_ASSETS}) reached."
                    )
                    finished_trades.append(
                        sell_most_profitable_asset(
                            portfolio,
                            portfolio_lock,
                            percent_changes,
                            finished_trades,
                            price_monitor_manager=price_monitor_manager,
                        )
                    )
                elif symbol in portfolio["assets"]:
                    logger.debug(f"Cannot buy {symbol}: Already owned.")
                elif symbol in low_volatility_assets:
                    logger.debug(f"Cannot buy {symbol}: Marked as low volatility.")
                else:
                    logger.warning(
                        f"Cannot buy {symbol}: Insufficient cash ({portfolio['cash']:.2f} EUR)."
                    )

    except Exception as e:
        logger.error(f"Error in portfolio management: {e}", exc_info=True)

    # Save order book metrics to CSV
    if order_book_metrics_list:
        append_to_order_book_metrics_csv(order_book_metrics_list)

    total_portfolio_value = portfolio["cash"] + total_asset_value
    if skipped_assets:
        logger.info(
            f"Portfolio value may be inaccurate due to missing prices for: {', '.join(skipped_assets)}"
        )
    logger.info(
        f"Portfolio: Cash: {portfolio['cash']:.2f} EUR, Assets: {len(portfolio['assets'])}, Total Value: {total_portfolio_value:.2f} EUR"
    )
