# trading_bot/src/data_processor.py
from datetime import datetime, timedelta
import os
import ccxt
import pandas as pd
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)
import mplfinance as mpf
from mplfinance.plotting import make_addplot
import numpy as np

from . import config
from .bitvavo_order_metrics import calculate_order_book_metrics
from .config import logger
from .exchange import check_rate_limit, semaphore
from .state import low_volatility_assets, portfolio, portfolio_lock

# Define ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"


def colorize_value(value, column):
    try:
        if column == "slippage_buy":
            if abs(value) < config.config.MAX_SLIPPAGE_BUY:
                return f"{GREEN}{value:>7.3f}%{RESET}"
            else:
                return f"{RED}{value:>7.3f}%{RESET}"
        elif column == "slippage_sell":
            if value > config.config.MAX_SLIPPAGE_SELL:
                return f"{GREEN}{value:>7.3f}%{RESET}"
            else:
                return f"{RED}{value:>7.3f}%{RESET}"
        elif column == "percent_change":
            if config.config.MIN_PERCENT_CHANGE <= value <= config.config.MAX_PERCENT_CHANGE:
                return f"{GREEN}{value:>6.3f}%{RESET}"
            else:
                return f"{RED}{value:>6.3f}%{RESET}"
        elif column == "volume_eur":
            if value > config.config.MIN_VOLUME_EUR:
                return f"{GREEN}{value:>10.2f}€{RESET}"
            else:
                return f"{RED}{value:>10.2f}€{RESET}"
        elif column == "recommendation":
            if value == "Strong Buy":
                return f"{GREEN}{value:<10}{RESET}"
            elif value == "No Buy":
                return f"{RED}{value:<10}{RESET}"
            return f"{value:<10}"
        elif column == "symbol":
            return f"{YELLOW}{value:<12}{RESET}"
        elif column == "peak_drop_percent":
            return f"{RED}{value:>6.3f}%{RESET}" if value < 0 else f"{GREEN}{value:>6.3f}%{RESET}"
        elif column == "minutes_since_peak":
            return f"{value:>6.1f}"
        return value
    except (TypeError, ValueError) as e:
        logger.error(
            f"Error formatting value {value} for column {column}: {e}", exc_info=True
        )
        return str(value)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying {retry_state.fn.__name__} after {retry_state.attempt_number} attempts"
    ),
    reraise=True,
)
def calculate_order_book_metrics_with_retry(market, amount_quote, price_range_percent):
    return calculate_order_book_metrics(
        market=market,
        amount_quote=amount_quote,
        price_range_percent=price_range_percent,
    )


def save_candlestick_plot(symbol, data, current_time):
    """
    Generate and save a candlestick plot with a closing price line and peak marker for the given symbol.

    Args:
        symbol (str): The coin symbol (e.g., 'BTC/EUR').
        data (pandas.DataFrame): DataFrame with OHLCV data for the symbol.
        current_time (datetime): Current UTC timestamp for naming the file.

    Returns:
        dict: Contains peak metrics (minutes_since_peak, peak_drop_percent).
    """
    try:
        # Initialize peak metrics
        peak_metrics = {
            "minutes_since_peak": np.nan,
            "peak_drop_percent": np.nan
        }

        # Format DataFrame for mplfinance
        plot_data = data[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        plot_data["timestamp"] = pd.to_datetime(plot_data["timestamp"])
        plot_data.set_index("timestamp", inplace=True)
        plot_data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            },
            inplace=True
        )

        # Find the most recent local or global maximum in close price
        if len(plot_data) < 3:
            peak_idx = plot_data["Close"].idxmax()
        else:
            closes = plot_data["Close"]
            is_peak = (closes > closes.shift(1)) & (closes > closes.shift(-1))
            peaks = closes[is_peak]
            peak_idx = peaks.index[-1] if not peaks.empty else closes.idxmax()

        peak_close = plot_data.loc[peak_idx, "Close"]
        latest_close = plot_data["Close"].iloc[-1]
        peak_timestamp = peak_idx

        # Calculate peak metrics
        minutes_since_peak = (plot_data.index[-1] - peak_timestamp).total_seconds() / 60.0
        peak_drop_percent = ((latest_close - peak_close) / peak_close) * 100
        peak_metrics = {
            "minutes_since_peak": minutes_since_peak,
            "peak_drop_percent": peak_drop_percent
        }

        # # Generate plot only if enabled
        # if config.config.ENABLE_CANDLESTICK_PLOTS:
        #     # Create marker for the peak
        #     peak_marker = np.full(len(plot_data), np.nan)
        #     peak_marker[plot_data.index.get_loc(peak_idx)] = peak_close
        #     peak_plot = mpf.make_addplot(
        #         peak_marker,
        #         type="scatter",
        #         markersize=100,
        #         marker="^",
        #         color="red",
        #         label="Recent Peak"
        #     )

        #     # Create line plot for closing prices
        #     close_line = mpf.make_addplot(plot_data["Close"], color="blue", width=1.5, label="Close Price")

        #     # Generate filename with symbol and timestamp
        #     timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        #     filename = f"plots/candlesticks/{symbol.replace('/', '-')}_{timestamp_str}.png"

        #     # Create candlestick plot with closing price line and peak marker
        #     os.makedirs("plots/candlesticks", exist_ok=True)
        #     mpf.plot(
        #         plot_data,
        #         type="candle",
        #         title=f"{symbol} Candlestick Chart",
        #         style="yahoo",
        #         ylabel="Price (EUR)",
        #         addplot=[close_line, peak_plot],
        #         savefig=filename,
        #         figsize=(10, 6)
        #     )
        #     logger.info(f"Saved candlestick plot for {symbol} to {filename}")

        return peak_metrics
    except Exception as e:
        logger.error(f"Error processing candlestick plot for {symbol}: {e}", exc_info=True)
        return {
            "minutes_since_peak": np.nan,
            "peak_drop_percent": np.nan
        }


def verify_and_analyze_data(df, price_monitor_manager):
    """
    Verifies and analyzes OHLCV data to identify assets for buying the dip.

    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and 'symbol' column.
        price_monitor_manager: Instance of PriceMonitorManager for managing price monitoring.

    Returns:
        tuple: (above_threshold_data, percent_changes, order_book_metrics_list)
            - above_threshold_data (list): List of dictionaries for assets meeting dip criteria.
            - percent_changes (pandas.DataFrame): DataFrame with price changes and OHLCV data.
            - order_book_metrics_list (list): List of order book metrics for filtered coins.

    Raises:
        ValueError: If input data is invalid or missing required columns.
        Exception: For unexpected errors during processing.
    """
    try:
        if df.empty:
            logger.warning("Input DataFrame is empty.")
            return [], pd.DataFrame(), []

        required_columns = {"timestamp", "open", "close", "volume", "symbol"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame missing required columns: {required_columns - set(df.columns)}"
            )

        current_time = datetime.utcnow()
        ten_min_ago = current_time - timedelta(minutes=10)
        five_min_ago = current_time - timedelta(minutes=5)

        if df["timestamp"].max() < ten_min_ago:
            logger.warning("Data contains no candles from within the last 10 minutes.")
            return [], pd.DataFrame(), []

        recent_data = df[df["timestamp"] >= five_min_ago]
        if recent_data.empty:
            logger.warning("No recent data within the last 5 minutes.")
            return [], pd.DataFrame(), []

        try:
            grouped = recent_data.groupby("symbol")
            symbols = grouped.first()
            latest = grouped.last()
            percent_changes = pd.DataFrame(
                {
                    "symbol": latest.index,
                    "percent_change": (
                        (latest["close"] - symbols["open"]) / symbols["open"] * 100
                    ).where(symbols["open"] > 0),
                    "open_price": symbols["open"],
                    "close_price": latest["close"],
                    "latest_timestamp": latest["timestamp"],
                    "volume_eur": (latest["volume"] * latest["close"]).where(
                        latest["volume"] > 0
                    ),
                }
            ).dropna()
        except (pd.errors.InvalidIndexError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error computing percent changes: {e}", exc_info=True)
            return [], pd.DataFrame(), []

        # Filter coins within MIN_PERCENT_CHANGE and MAX_PERCENT_CHANGE
        filtered_coins = percent_changes[
            (percent_changes["percent_change"] >= config.config.MIN_PERCENT_CHANGE) &
            (percent_changes["percent_change"] <= config.config.MAX_PERCENT_CHANGE)
        ]

        above_threshold_data = []
        order_book_metrics_list = []

        # Process coins for dip buying criteria
        for _, row in filtered_coins.iterrows():
            symbol = row["symbol"]
            try:
                # Get peak metrics
                symbol_data = df[df["symbol"] == symbol][[
                    "timestamp", "open", "high", "low", "close", "volume"
                ]]
                if symbol_data.empty:
                    logger.warning(f"No data for {symbol} to analyze")
                    continue

                peak_metrics = save_candlestick_plot(symbol, symbol_data, current_time)

                # Check dip buying criteria
                if (
                    not np.isnan(peak_metrics["minutes_since_peak"]) and
                    not np.isnan(peak_metrics["peak_drop_percent"]) and
                    peak_metrics["minutes_since_peak"] <= config.config.MAX_MINUTES_SINCE_PEAK and
                    peak_metrics["peak_drop_percent"] <= config.config.MIN_PEAK_DROP_PERCENT
                ):
                    # Calculate order book metrics
                    if not semaphore.acquire(timeout=5):
                        logger.error(f"Timeout acquiring semaphore for {symbol}")
                        send_alert(
                            "Semaphore Failure", f"Failed to acquire semaphore for {symbol}"
                        )
                        continue
                    try:
                        check_rate_limit(1)
                        metrics = calculate_order_book_metrics_with_retry(
                            market=symbol.replace("/", "-"),
                            amount_quote=portfolio["cash"] * config.config.ALLOCATION_PER_TRADE,
                            price_range_percent=config.config.PRICE_RANGE_PERCENT,
                        )
                        metrics["bought"] = False
                        order_book_metrics_list.append(metrics)

                        # Add to above_threshold_data
                        coin_data = row.to_dict()
                        coin_data.update(peak_metrics)
                        above_threshold_data.append(coin_data)

                        # Log coin data
                        logger.info(
                            f"Symbol: {colorize_value(symbol, 'symbol')}  "
                            f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                            f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                            f"Open: {row['open_price']:>15.8f}  "
                            f"Close: {row['close_price']:>15.8f}  "
                            f"Slippage Buy: {colorize_value(metrics.get('slippage_buy', 0), 'slippage_buy')}  "
                            f"Slippage Sell: {colorize_value(metrics.get('slippage_sell', 0), 'slippage_sell')}  "
                            f"Total Score: {metrics.get('total_score', 0):>4.2f}  "
                            f"Recommendation: {colorize_value(metrics.get('recommendation', 'N/A'), 'recommendation')}  "
                            f"Minutes Since Peak: {colorize_value(peak_metrics['minutes_since_peak'], 'minutes_since_peak')} min  "
                            f"Peak Drop: {colorize_value(peak_metrics['peak_drop_percent'], 'peak_drop_percent')}  "
                            f"Latest Timestamp: {row['latest_timestamp']}"
                        )
                    finally:
                        semaphore.release()
                else:
                    logger.info(
                        f"Symbol: {colorize_value(symbol, 'symbol')} skipped - "
                        f"Minutes Since Peak: {colorize_value(peak_metrics['minutes_since_peak'], 'minutes_since_peak')} min "
                        f"(> {config.config.MAX_MINUTES_SINCE_PEAK}), "
                        f"Peak Drop: {colorize_value(peak_metrics['peak_drop_percent'], 'peak_drop_percent')} "
                        f"(> {config.config.MIN_PEAK_DROP_PERCENT}%)"
                    )
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logger.error(
                    f"Network error for {symbol}: {e}", exc_info=True
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error for {symbol}: {e}", exc_info=True
                )

        # Log if no coins meet criteria
        if not above_threshold_data:
            logger.info(
                f"{BRIGHT_BLUE}DATA GATHERING: No coins with price increase between {config.config.MIN_PERCENT_CHANGE:.2f}% "
                f"and {config.config.MAX_PERCENT_CHANGE:.2f}% meeting dip criteria{RESET}"
            )

        # Update low volatility assets
        try:
            if not portfolio_lock.acquire(timeout=5):
                logger.error("Timeout acquiring portfolio lock")
                send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
                return (
                    above_threshold_data,
                    percent_changes,
                    order_book_metrics_list,
                )
            try:
                for symbol in list(low_volatility_assets):
                    if (
                        symbol in portfolio["assets"]
                        and symbol in percent_changes.index
                    ):
                        try:
                            recent_change = percent_changes.loc[
                                percent_changes["symbol"] == symbol, "percent_change"
                            ].iloc[0]
                            if (
                                abs(recent_change)
                                >= config.config.PRICE_INCREASE_THRESHOLD / 2
                            ):
                                logger.info(
                                    f"{symbol} regained volatility (change: {recent_change:.2f}%). Resuming monitoring."
                                )
                                low_volatility_assets.discard(symbol)
                                price_monitor_manager.start(
                                    symbol, portfolio, portfolio_lock, df
                                )
                        except (KeyError, IndexError) as e:
                            logger.error(
                                f"Error checking volatility for {symbol}: {e}",
                                exc_info=True,
                            )
            finally:
                portfolio_lock.release()
        except Exception as e:
            logger.error(f"Error managing low volatility assets: {e}", exc_info=True)

        return (
            above_threshold_data,
            percent_changes,
            order_book_metrics_list,
        )

    except Exception as e:
        logger.error(f"Critical error in verify_and_analyze_data: {e}", exc_info=True)
        send_alert("Data Processing Failure", f"Critical error in data processing: {e}")
        return [], pd.DataFrame(), []


def send_alert(subject, message):
    logger.error(f"ALERT: {subject} - {message}")