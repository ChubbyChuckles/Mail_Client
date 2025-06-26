# trading_bot/src/data_processor.py
from datetime import datetime, timedelta

import pandas as pd

from .bitvavo_order_metrics import calculate_order_book_metrics
from .config import (MIN_VOLUME_EUR, PRICE_INCREASE_THRESHOLD, MAX_SLIPPAGE_BUY, MAX_SLIPPAGE_SELL, ALLOCATION_PER_TRADE, MIN_TOTAL_SCORE,
                     PRICE_RANGE_PERCENT, logger)
from .exchange import check_rate_limit, semaphore
from .state import low_volatility_assets, portfolio, portfolio_lock

# Define ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"  # Resets color to default

def colorize_value(value, column):
    if column == "slippage_buy":
        if abs(value) < MAX_SLIPPAGE_BUY:
            return f"{GREEN}{value:>7.3f}%{RESET}"
        else:
            return f"{RED}{value:>7.3f}%{RESET}"
    elif column == "slippage_sell":
        if value > MAX_SLIPPAGE_SELL:
            return f"{GREEN}{value:>7.3f}%{RESET}"
        else:
            return f"{RED}{value:>7.3f}%{RESET}"
    elif column == "percent_change":
        if value > PRICE_INCREASE_THRESHOLD:
            return f"{GREEN}{value:>6.3f}%{RESET}"
        else:
            return f"{RED}{value:>6.3f}%{RESET}"
    elif column == "volume_eur":
        if value > MIN_VOLUME_EUR:
            return f"{GREEN}{value:>10.2f}€{RESET}"
        else:
            return f"{RED}{value:>10.2f}€{RESET}"
    elif column == "recommendation":
        if value == "Strong Buy":
            return f"{GREEN}{value:<10}{RESET}"
        elif value == "No Buy":
            return f"{RED}{value:<10}{RESET}"
        return f"{value:<10}"  # Default for other recommendations
    elif column == "symbol":
        return f"{YELLOW}{value:<12}{RESET}"
    
    return value  # Default case for uncolored columns

def verify_and_analyze_data(df, price_monitor_manager):
    """
    Verifies and analyzes OHLCV data to identify assets with significant price increases and volume.

    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and 'symbol' column.

    Returns:
        tuple: (above_threshold_data, percent_changes, order_book_metrics_list)
            - above_threshold_data (list): List of dictionaries for assets meeting thresholds.
            - percent_changes (pandas.DataFrame): DataFrame with price changes and OHLCV data.
            - order_book_metrics_list (list): List of order book metrics for relevant coins.
    """
    
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return [], pd.DataFrame(), []

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

    above_threshold = percent_changes[
        (percent_changes["percent_change"] >= PRICE_INCREASE_THRESHOLD)
        & (percent_changes["volume_eur"] >= MIN_VOLUME_EUR)
    ]

    order_book_metrics_list = []

    # Calculate order book metrics for coins above threshold
    for _, row in above_threshold.iterrows():
        symbol = row["symbol"]
        # logger.info(f"Calculating order book metrics for {symbol} (above threshold)")
        with semaphore:
            check_rate_limit(1)  # Assume order book fetch has weight of 2
            metrics = calculate_order_book_metrics(
                market=symbol.replace("/", "-"),
                amount_quote=portfolio['cash'] * ALLOCATION_PER_TRADE,
                price_range_percent=PRICE_RANGE_PERCENT,
            )
            metrics["bought"] = False  # Will be updated in portfolio.py if bought
            order_book_metrics_list.append(metrics)

    if not above_threshold.empty:
        logger.info(
            f"{BRIGHT_BLUE}DATA GATHERING: Coins with price increase >= {PRICE_INCREASE_THRESHOLD}% and volume >= €{MIN_VOLUME_EUR}:   (only Buy if Total Score > {MIN_TOTAL_SCORE} and Slippage Buy < {MAX_SLIPPAGE_BUY}%){RESET}"
        )
        for _, row in above_threshold.iterrows():

            logger.info(
                f"Symbol: {colorize_value(row['symbol'], 'symbol')}  "
                f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                f"Open: {row['open_price']:>15.8f}  "
                f"Close: {row['close_price']:>15.8f}  "
                f"Slippage Buy: {colorize_value(metrics['slippage_buy'], 'slippage_buy')}  "
                f"Slippage Sell: {colorize_value(metrics['slippage_sell'], 'slippage_sell')}  "
                f"Total Score: {metrics['total_score']:>4.2f}  "
                f"Recommendation: {colorize_value(metrics['recommendation'], 'recommendation')}  "
                f"Latest Timestamp: {row['latest_timestamp']}"
            )

    else:
        logger.info(
            f"{BRIGHT_BLUE}DATA GATHERING: No coins with price increase >= {PRICE_INCREASE_THRESHOLD}% and trade volume >= {MIN_VOLUME_EUR} €   (only Buy if Total Score > {MIN_TOTAL_SCORE} and Slippage Buy < {MAX_SLIPPAGE_BUY}%){RESET}{RESET}"
        )

    below_threshold = percent_changes[
        percent_changes["percent_change"] < PRICE_INCREASE_THRESHOLD
    ]
    if not below_threshold.empty:
        top_5_below = below_threshold.sort_values(
            by="percent_change", ascending=False
        ).head(5)
        logger.info(f"{BRIGHT_BLUE}DATA GATHERING: Top 5 coins with price increase < {PRICE_INCREASE_THRESHOLD}% or trade volume < {MIN_VOLUME_EUR} €:{RESET}")
        for _, row in top_5_below.iterrows():

            # Calculate order book metrics for top 5 below threshold
            symbol = row["symbol"]
            # logger.info(f"Calculating order book metrics for {symbol} (below threshold)")
            with semaphore:
                check_rate_limit(1)
                metrics = calculate_order_book_metrics(
                    market=symbol.replace("/", "-"),
                    amount_quote=portfolio['cash'] * ALLOCATION_PER_TRADE,
                    price_range_percent=PRICE_RANGE_PERCENT,
                )
                metrics["bought"] = False
                order_book_metrics_list.append(metrics)

            logger.info(
                f"Symbol: {colorize_value(row['symbol'], 'symbol')}  "
                f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                f"Open: {row['open_price']:>15.8f}  "
                f"Close: {row['close_price']:>15.8f}  "
                f"Slippage Buy: {colorize_value(metrics['slippage_buy'], 'slippage_buy')}  "
                f"Slippage Sell: {colorize_value(metrics['slippage_sell'], 'slippage_sell')}  "
                f"Total Score: {metrics['total_score']:>4.2f}  "
                f"Recommendation: {colorize_value(metrics['recommendation'], 'recommendation')}  "
                f"Latest Timestamp: {row['latest_timestamp']}"
            )
    else:
        logger.info(f"No coins with price increase < {PRICE_INCREASE_THRESHOLD}%")

    with portfolio_lock:
        for symbol in list(low_volatility_assets):
            if symbol in portfolio["assets"] and symbol in percent_changes.index:
                recent_change = percent_changes.loc[
                    percent_changes["symbol"] == symbol, "percent_change"
                ].iloc[0]
                if abs(recent_change) >= PRICE_INCREASE_THRESHOLD / 2:
                    logger.info(
                        f"{symbol} regained volatility (change: {recent_change:.2f}%). Resuming monitoring."
                    )
                    low_volatility_assets.discard(symbol)
                    price_monitor_manager.start(symbol, portfolio, portfolio_lock, df)

    return above_threshold.to_dict("records"), percent_changes, order_book_metrics_list
