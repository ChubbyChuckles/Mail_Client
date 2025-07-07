# trading_bot/src/data_processor.py
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

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


def verify_and_analyze_data(df, price_monitor_manager):
    """
    Verifies and analyzes OHLCV data to identify assets within the specified percentage change range.

    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and 'symbol' column.
        price_monitor_manager: Instance of PriceMonitorManager for managing price monitoring.

    Returns:
        tuple: (above_threshold_data, percent_changes, order_book_metrics_list)
            - above_threshold_data (list): List of dictionaries for assets within the percentage change range.
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

        order_book_metrics_list = []

        # Calculate order book metrics for filtered coins
        for _, row in filtered_coins.iterrows():
            symbol = row["symbol"]
            try:
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
                finally:
                    semaphore.release()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logger.error(
                    f"Network error calculating order book metrics for {symbol}: {e}",
                    exc_info=True,
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error calculating order book metrics for {symbol}: {e}",
                    exc_info=True,
                )

        # Log filtered coins
        if not filtered_coins.empty:
            logger.info(
                f"{BRIGHT_BLUE}DATA GATHERING: Coins with price increase between {config.config.MIN_PERCENT_CHANGE:.2f}% "
                f"and {config.config.MAX_PERCENT_CHANGE:.2f}%{RESET}"
            )
            logger.info(
                f"{YELLOW}(only Buy if Total Score > {config.config.MIN_TOTAL_SCORE} and Slippage Buy < {config.config.MAX_SLIPPAGE_BUY}%){RESET}"
            )
            for _, row in filtered_coins.iterrows():
                try:
                    symbol = row["symbol"]
                    metrics = next(
                        (
                            m
                            for m in order_book_metrics_list
                            if m["market"] == symbol.replace("/", "-")
                        ),
                        {},
                    )
                    if not metrics:
                        logger.warning(f"No order book metrics for {symbol}")
                        continue
                    logger.info(
                        f"Symbol: {colorize_value(row['symbol'], 'symbol')}  "
                        f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                        f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                        f"Open: {row['open_price']:>15.8f}  "
                        f"Close: {row['close_price']:>15.8f}  "
                        f"Slippage Buy: {colorize_value(metrics.get('slippage_buy', 0), 'slippage_buy')}  "
                        f"Slippage Sell: {colorize_value(metrics.get('slippage_sell', 0), 'slippage_sell')}  "
                        f"Total Score: {metrics.get('total_score', 0):>4.2f}  "
                        f"Recommendation: {colorize_value(metrics.get('recommendation', 'N/A'), 'recommendation')}  "
                        f"Latest Timestamp: {row['latest_timestamp']}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error logging data for {row['symbol']}: {e}", exc_info=True
                    )
        else:
            logger.info(
                f"{BRIGHT_BLUE}DATA GATHERING: No coins with price increase between {config.config.MIN_PERCENT_CHANGE:.2f}% "
                f"and {config.config.MAX_PERCENT_CHANGE:.2f}%{RESET}"
            )

        # Update low volatility assets
        try:
            if not portfolio_lock.acquire(timeout=5):
                logger.error("Timeout acquiring portfolio lock")
                send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
                return (
                    filtered_coins.to_dict("records"),
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
            filtered_coins.to_dict("records"),
            percent_changes,
            order_book_metrics_list,
        )

    except Exception as e:
        logger.error(f"Critical error in verify_and_analyze_data: {e}", exc_info=True)
        send_alert("Data Processing Failure", f"Critical error in data processing: {e}")
        return [], pd.DataFrame(), []


def send_alert(subject, message):
    logger.error(f"ALERT: {subject} - {message}")