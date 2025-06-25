# trading_bot/src/data_processor.py
from datetime import datetime, timedelta

import pandas as pd

from .bitvavo_order_metrics import calculate_order_book_metrics
from .config import (AMOUNT_QUOTE, MIN_VOLUME_EUR, PRICE_INCREASE_THRESHOLD,
                     PRICE_RANGE_PERCENT, logger)
from .exchange import check_rate_limit, semaphore
from .state import low_volatility_assets, portfolio, portfolio_lock


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
                amount_quote=AMOUNT_QUOTE,
                price_range_percent=PRICE_RANGE_PERCENT,
            )
            metrics["bought"] = False  # Will be updated in portfolio.py if bought
            order_book_metrics_list.append(metrics)

    if not above_threshold.empty:
        logger.info(
            f"\nCoins with price increase >= {PRICE_INCREASE_THRESHOLD}% and volume >= €{MIN_VOLUME_EUR}:"
        )
        for _, row in above_threshold.iterrows():

            logger.info(
                f"Symbol: {row['symbol']:<12}  "
                f"Change: {row['percent_change']:>6.3f}%  "
                f"Volume: {row['volume_eur']:>10.2f}€  "
                f"Open: {row['open_price']:>12.8f}  "
                f"Close: {row['close_price']:>12.8f}  "
                f"Slippage Buy: {metrics['slippage_buy']:>7.3f}%  "
                f"Slippage Sell: {metrics['slippage_sell']:>7.3f}%  "
                f"Total Score: {metrics['total_score']:>4.2f}  "
                f"Recommendation: {metrics['recommendation']:<10}  "
                f"Latest Timestamp: {row['latest_timestamp']}"
            )

    else:
        logger.info(
            f"No coins with price increase >= {PRICE_INCREASE_THRESHOLD}% and volume >= €{MIN_VOLUME_EUR}"
        )

    below_threshold = percent_changes[
        percent_changes["percent_change"] < PRICE_INCREASE_THRESHOLD
    ]
    if not below_threshold.empty:
        top_5_below = below_threshold.sort_values(
            by="percent_change", ascending=False
        ).head(5)
        logger.info(f"\nTop 5 coins with price increase < {PRICE_INCREASE_THRESHOLD}%:")
        for _, row in top_5_below.iterrows():

            # Calculate order book metrics for top 5 below threshold
            symbol = row["symbol"]
            # logger.info(f"Calculating order book metrics for {symbol} (below threshold)")
            with semaphore:
                check_rate_limit(1)
                metrics = calculate_order_book_metrics(
                    market=symbol.replace("/", "-"),
                    amount_quote=AMOUNT_QUOTE,
                    price_range_percent=PRICE_RANGE_PERCENT,
                )
                metrics["bought"] = False
                order_book_metrics_list.append(metrics)

            logger.info(
                f"Symbol: {row['symbol']:<12}  "
                f"Change: {row['percent_change']:>6.3f}%  "
                f"Volume: {row['volume_eur']:>10.2f}€  "
                f"Open: {row['open_price']:>12.8f}  "
                f"Close: {row['close_price']:>12.8f}  "
                f"Slippage Buy: {metrics['slippage_buy']:>7.3f}%  "
                f"Slippage Sell: {metrics['slippage_sell']:>7.3f}%  "
                f"Total Score: {metrics['total_score']:>4.2f}  "
                f"Recommendation: {metrics['recommendation']:<10}  "
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
