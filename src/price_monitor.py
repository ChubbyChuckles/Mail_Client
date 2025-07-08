# trading_bot/src/price_monitor.py
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
from ccxt.base.errors import PermissionDenied

from . import config
from .config import IS_GITHUB_ACTIONS, logger
from .exchange import (bitvavo, check_rate_limit, handle_ban_error, semaphore,
                       wait_until_ban_lifted)
from .portfolio import sell_asset
from .state import (ban_expiry_time, is_banned, low_volatility_assets,
                    negative_momentum_counts, weight_used)
from .utils import calculate_dynamic_ema_period, calculate_ema
from .bitvavo_order_metrics import calculate_order_book_metrics  # Added for slippage

class PriceMonitorManager:
    def __init__(self):
        """Initializes the price monitor manager with thread and state tracking."""
        self.threads = {}
        self.running = {}
        self.last_update = {}
        self.exchange = bitvavo
        self.last_prices = {}
        self.ticker_errors = {}

    def adjust_concurrency(self, max_threads):
        """
        Adjusts the number of active monitoring threads to the specified limit.

        Args:
            max_threads (int): Maximum number of concurrent threads allowed.
        """
        with threading.Lock():
            # Use environment-specific concurrency limit
            effective_max_threads = (
                min(max_threads, config.config.CONCURRENT_REQUESTS)
                if IS_GITHUB_ACTIONS
                else max_threads
            )
            config.config.CONCURRENT_REQUESTS = effective_max_threads
            current_threads = len(self.threads)
            if current_threads > effective_max_threads:
                threads_to_stop = sorted(
                    self.threads.items(), key=lambda x: self.last_update.get(x[0], 0)
                )[: current_threads - effective_max_threads]
                for symbol, _ in threads_to_stop:
                    self.stop(symbol)
                logger.info(
                    f"Reduced active monitoring threads to {effective_max_threads}"
                )

    def handle_ticker(self, symbol, portfolio, portfolio_lock, candles_df):
        """
        Monitors ticker prices for a symbol and updates portfolio state.

        Args:
            symbol (str): Asset symbol (e.g., 'BTC/EUR').
            portfolio (dict): Portfolio dictionary.
            portfolio_lock (Lock): Thread lock for portfolio updates.
            candles_df (pandas.DataFrame): DataFrame with OHLCV data.
        """
        global is_banned, ban_expiry_time
        candles = []
        last_candle_time = None
        self.last_update[symbol] = time.time()
        try:
            logger.info(f"Started price monitoring for {symbol}")
            while self.running.get(symbol, False):
                if not self.running.get(symbol, False):  # Check before proceeding
                    break
                if is_banned and time.time() < ban_expiry_time:
                    logger.warning(
                        f"API banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Pausing {symbol}."
                    )
                    wait_until_ban_lifted(ban_expiry_time)
                    continue
                try:
                    check_rate_limit(1)
                    with semaphore:
                        ticker = self.exchange.fetch_ticker(symbol)
                        if not isinstance(ticker, dict) or "last" not in ticker:
                            logger.error(
                                f"Invalid ticker response for {symbol}: {ticker}"
                            )
                            self.ticker_errors[symbol] = (
                                self.ticker_errors.get(symbol, 0) + 1
                            )
                            if self.ticker_errors[symbol] >= 3:
                                logger.warning(
                                    f"{symbol} has {self.ticker_errors[symbol]} ticker errors. Marking as low volatility."
                                )
                                with portfolio_lock:
                                    low_volatility_assets.add(symbol)
                                self.stop(symbol)
                                break
                            time.sleep(0.1)
                            continue
                        price = float(ticker["last"])
                        self.last_prices[symbol] = price
                        self.ticker_errors[symbol] = 0
                        current_time = datetime.utcnow()
                        current_second = current_time.replace(microsecond=0)
                        self.last_update[symbol] = time.time()
                        # Minimize lock scope
                        if self.running.get(symbol, False):
                            with portfolio_lock:
                                if symbol in portfolio["assets"]:
                                    portfolio["assets"][symbol]["current_price"] = price
                                    portfolio["assets"][symbol]["highest_price"] = max(
                                        portfolio["assets"][symbol]["highest_price"],
                                        price,
                                    )
                        if (
                            last_candle_time is None
                            or current_second > last_candle_time
                        ):
                            if candles:
                                self.evaluate_candle(
                                    candles[-1],
                                    symbol,
                                    portfolio,
                                    portfolio_lock,
                                    candles,
                                    candles_df,
                                )
                            candles.append(
                                {
                                    "timestamp": current_second,
                                    "open": price,
                                    "high": price,
                                    "low": price,
                                    "close": price,
                                    "volume": 0,
                                }
                            )
                            last_candle_time = current_second
                        else:
                            candles[-1]["high"] = max(candles[-1]["high"], price)
                            candles[-1]["low"] = min(candles[-1]["low"], price)
                            candles[-1]["close"] = price
                        candles = [
                            c
                            for c in candles
                            if (current_time - c["timestamp"]).total_seconds() <= 5
                        ]
                        # Check running state before accessing last_update
                        if self.running.get(symbol, False):
                            last_update_time = self.last_update.get(symbol, time.time())
                            if time.time() - last_update_time > config.config.INACTIVITY_TIMEOUT:
                                logger.info(
                                    f"{symbol} inactive for {config.config.INACTIVITY_TIMEOUT} seconds. Marking as low volatility."
                                )
                                with portfolio_lock:
                                    low_volatility_assets.add(symbol)
                                self.stop(symbol)
                                break
                        time.sleep(2 if IS_GITHUB_ACTIONS else 1)
                except PermissionDenied as e:
                    ban_expiry = handle_ban_error(e)
                    if ban_expiry:
                        wait_until_ban_lifted(ban_expiry)
                        continue
                    logger.error(f"Permission denied for {symbol}: {e}")
                    self.ticker_errors[symbol] = self.ticker_errors.get(symbol, 0) + 1
                    if self.ticker_errors[symbol] >= 3:
                        with portfolio_lock:
                            low_volatility_assets.add(symbol)
                        self.stop(symbol)
                        break
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Ticker error for {symbol}: {e}", exc_info=True)
                    self.ticker_errors[symbol] = self.ticker_errors.get(symbol, 0) + 1
                    if self.ticker_errors[symbol] >= 3:
                        with portfolio_lock:
                            low_volatility_assets.add(symbol)
                        self.stop(symbol)
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Price monitoring error for {symbol}: {e}", exc_info=True)
        finally:
            logger.info(f"Stopped price monitoring for {symbol}")

    def evaluate_candle(
        self, candle, symbol, portfolio, portfolio_lock, candles, prices_df
    ):
        with portfolio_lock:
            if symbol not in portfolio["assets"]:
                return
            asset = portfolio["assets"][symbol]
            current_price = (
                candle["close"] if candle["close"] else self.last_prices.get(symbol)
            )
            if current_price is None:
                logger.warning(f"No valid price for {symbol}. Skipping evaluation.")
                return
            asset["current_price"] = current_price
            purchase_price = asset["purchase_price"]
            highest_price = max(asset["highest_price"], current_price)
            asset["highest_price"] = highest_price
            holding_minutes = (
                (datetime.utcnow() - asset["purchase_time"]).total_seconds() / 60
            )
            symbol_candles = prices_df[prices_df["symbol"] == symbol].tail(5)
            atr = (
                np.mean(symbol_candles["high"] - symbol_candles["low"])
                if len(symbol_candles) >= 5
                else 0
            )
            trailing_stop = (
                (
                    config.config.TRAILING_STOP_FACTOR_EARLY
                    if holding_minutes < 15
                    else config.config.TRAILING_STOP_FACTOR
                )
                * atr
                / purchase_price
                if atr > 0 and purchase_price > 0
                else 0.05
            )
            profit_target = (
                max(0.015, min(0.05, 1.2 * atr / purchase_price))
                if atr > 0 and purchase_price > 0
                else asset.get("profit_target", 0.015)
            )
            if (
                len(portfolio["assets"]) >= config.config.ASSET_THRESHOLD
                and current_price > purchase_price * (1 + config.config.MIN_PROFIT_PERCENT / 100)
            ):
                profit_target = min(profit_target, config.config.ADJUSTED_PROFIT_TARGET)
            asset["profit_target"] = profit_target
            ema_period = calculate_dynamic_ema_period(
                holding_minutes,
                config.config.TIME_STOP_MINUTES,
                len(portfolio["assets"]),
                config.config.ASSET_THRESHOLD,
            )
            ema_dynamic = (
                calculate_ema(symbol_candles["close"].values, ema_period)
                if len(symbol_candles) >= ema_period
                else current_price
            )
            symbol_data = prices_df[prices_df["symbol"] == symbol].tail(1)
            momentum = 0
            if (
                not symbol_data.empty
                and "open" in symbol_data
                and not symbol_data["open"].isnull().iloc[0]
            ):
                momentum = (
                    (current_price - symbol_data["open"].iloc[0])
                    / symbol_data["open"].iloc[0]
                ) * 100
            if momentum < config.config.MOMENTUM_THRESHOLD:
                negative_momentum_counts[symbol] = (
                    negative_momentum_counts.get(symbol, 0) + 1
                )
            else:
                negative_momentum_counts[symbol] = 0
            buy_fee = asset.get("buy_fee", 0)
            total_cost = purchase_price * asset["quantity"] + buy_fee
            # Calculate net profit/loss
            try:
                amount_quote = asset["quantity"] * current_price
                metrics = calculate_order_book_metrics(
                    symbol.replace("/", "-"), amount_quote=amount_quote
                )
                sell_slippage = (
                    metrics["slippage_sell"] / 100  # Convert percentage to decimal
                    if "error" not in metrics and metrics.get("slippage_sell") is not None
                    else float('inf')
                )
            except Exception as e:
                logger.error(f"Error calculating sell slippage for {symbol}: {e}", exc_info=True)
                sell_slippage = float('inf')
            sale_value = asset["quantity"] * current_price * (1 - abs(sell_slippage))
            sell_fee = sale_value * config.config.SELL_FEE
            net_sale_value = sale_value - sell_fee
            net_profit_loss = (
                (net_sale_value - total_cost) / total_cost * 100
                if total_cost > 0
                else 0
            )
            unrealized_profit = (
                ((current_price * asset["quantity"]) - total_cost) / total_cost * 100
                if total_cost > 0
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
            asset["sell_price"] = (
                min(sell_prices) if sell_prices else profit_target_price
            )
            catastrophic_loss = (
                net_profit_loss <= config.config.CAT_LOSS_THRESHOLD * 100
                and abs(net_profit_loss) > 2 * atr / purchase_price * 100
                if atr > 0 and purchase_price > 0
                else False
            )
            time_stop = (
                holding_minutes >= config.config.TIME_STOP_MINUTES
                and net_profit_loss < 0
            )
            multiplied_profit_target = (
                net_profit_loss
                >= config.config.PROFIT_TARGET_MULTIPLIER * profit_target * 100
            )
            if "reached_min_profit" not in asset:
                asset["reached_min_profit"] = False
            if net_profit_loss >= config.config.MIN_PROFIT_PERCENT:
                asset["reached_min_profit"] = True
            loss_exceeded = net_profit_loss <= config.config.MAX_UNREALIZED_LOSS_PERCENT
            dip_below_profit = (
                asset["reached_min_profit"]
                and net_profit_loss < config.config.MIN_PROFIT_PERCENT
            )
            advanced_sell_signal = (
                net_profit_loss >= config.config.MIN_PROFIT_PERCENT
                and (
                    trailing_loss >= trailing_stop
                    or net_profit_loss >= profit_target * 100
                    or negative_momentum_counts.get(symbol, 0)
                    >= config.config.MOMENTUM_CONFIRM_MINUTES
                    or multiplied_profit_target
                    or catastrophic_loss
                    or time_stop
                )
            )
            slippage_ok = abs(sell_slippage) <= abs(config.config.MAX_SLIPPAGE_SELL)
            sell_signal = (loss_exceeded or dip_below_profit or advanced_sell_signal) and slippage_ok
            if sell_signal:
                reason = (
                    "Unrealized loss exceeded"
                    if loss_exceeded
                    else (
                        "Dip below minimum profit"
                        if dip_below_profit
                        else (
                            "Catastrophic loss"
                            if catastrophic_loss
                            else (
                                "Time stop"
                                if time_stop
                                else (
                                    f"Multiplied profit target ({config.config.PROFIT_TARGET_MULTIPLIER}x = {(config.config.PROFIT_TARGET_MULTIPLIER * profit_target)*100:.1f}%)"
                                    if multiplied_profit_target
                                    else (
                                        "Trailing stop"
                                        if trailing_loss >= trailing_stop
                                        else (
                                            f"Dynamic profit target ({profit_target*100:.1f}%)"
                                            if net_profit_loss >= profit_target * 100
                                            else "Negative momentum"
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                logger.info(
                    f"Evaluation Decision for {symbol}: Selling due to {reason}. "
                    f"Current: {current_price:.4f}, Highest: {highest_price:.4f}, "
                    f"Trailing Loss: {trailing_loss:.4f}, ATR Stop: {trailing_stop:.4f}, "
                    f"Unrealized P/L: {unrealized_profit:.2f}%, Net P/L: {net_profit_loss:.2f}%, "
                    f"Profit Target: {profit_target:.4f}, EMA_{ema_period}: {ema_dynamic:.2f}, "
                    f"Holding: {holding_minutes:.2f} min, Neg Momentum Count: {negative_momentum_counts.get(symbol, 0)}, "
                    f"Sell Slippage: {sell_slippage*100:.2f}%, Reached Min Profit: {asset['reached_min_profit']}"
                )
                finished_trade = sell_asset(
                    symbol,
                    asset,
                    current_price,
                    portfolio,
                    portfolio_lock,
                    [],
                    reason,
                    price_monitor_manager=self,
                    sell_slippage=sell_slippage,
                )
                if finished_trade:
                    from .utils import append_to_finished_trades_csv
                    append_to_finished_trades_csv(finished_trade)
            else:
                if not slippage_ok:
                    logger.info(
                        f"Delaying sell for {symbol}: Sell slippage {sell_slippage*100:.2f}% exceeds threshold {config.config.MAX_SLIPPAGE_SELL*100:.2f}%"
                    )
                # logger.info(
                #     f"Asset: {symbol}, "
                #     f"Current: {current_price:.4f}, "
                #     f"Purchase: {purchase_price:.4f}, "
                #     f"Quantity: {asset['quantity']:.4f}, "
                #     f"Unrealized P/L: {unrealized_profit:.2f}%, "
                #     f"Net P/L: {net_profit_loss:.2f}%, "
                #     f"Sell Slippage: {sell_slippage*100:.2f}%, "
                #     f"Holding: {holding_minutes:.2f} min, "
                #     f"Reached Min Profit: {asset['reached_min_profit']}"
                # )
    
    def start(self, symbol, portfolio, portfolio_lock, candles_df):
        """
        Starts a price monitoring thread for a symbol.

        Args:
            symbol (str): Asset symbol.
            portfolio (dict): Portfolio dictionary.
            portfolio_lock (Lock): Thread lock for portfolio updates.
            candles_df (pandas.DataFrame): DataFrame with OHLCV data.
        """
        if symbol not in self.threads and symbol not in low_volatility_assets:
            with threading.Lock():
                global weight_used
                max_threads = config.config.CONCURRENT_REQUESTS
                if len(self.threads) >= max_threads:
                    logger.warning(
                        f"Max threads ({max_threads}) reached. Cannot start monitoring for {symbol}."
                    )
                    return
                if weight_used + 2 > config.config.RATE_LIMIT_WEIGHT * 0.8:
                    logger.warning(
                        f"Approaching rate limit ({weight_used}). Delaying monitoring for {symbol}."
                    )
                    time.sleep(5)
                    return
            self.running[symbol] = True
            thread = threading.Thread(
                target=self.handle_ticker,
                args=(symbol, portfolio, portfolio_lock, candles_df),
            )
            self.threads[symbol] = thread
            thread.start()
            logger.info(f"Started price monitoring thread for {symbol}")

    def stop(self, symbol):
        """
        Stops the price monitoring thread for a symbol.

        Args:
            symbol (str): Asset symbol.
        """
        if symbol in self.running:
            try:
                logger.debug(f"Attempting to stop price monitor for {symbol}")
                self.running[symbol] = False
                if symbol in self.threads:
                    thread = self.threads[symbol]
                    if thread != threading.current_thread():
                        thread.join(timeout=1)
                        if thread.is_alive():
                            logger.warning(
                                f"Thread for {symbol} did not terminate within 1 second."
                            )
                        else:
                            logger.debug(
                                f"Thread for {symbol} terminated successfully."
                            )
                    del self.threads[symbol]
                del self.running[symbol]
                self.last_update.pop(symbol, None)
                self.last_prices.pop(symbol, None)
                logger.info(f"Stopped price monitoring thread for {symbol}")
            except Exception as e:
                logger.error(
                    f"Error stopping price monitor for {symbol}: {e}", exc_info=True
                )
            finally:
                self.threads.pop(symbol, None)
                self.running.pop(symbol, None)

    def stop_all(self):
        """Stops all active price monitoring threads."""
        for symbol in list(self.running.keys()):
            self.stop(symbol)

    def active_monitors(self):
        """
        Returns the number of active monitoring threads.

        Returns:
            int: Number of active monitors.
        """
        return len(self.running)