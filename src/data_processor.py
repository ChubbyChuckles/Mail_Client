# trading_bot/src/data_processor.py
from datetime import datetime, timedelta

import pandas as pd

from .config import MIN_VOLUME_EUR, PRICE_INCREASE_THRESHOLD, logger
from .state import low_volatility_assets, portfolio, portfolio_lock


def verify_and_analyze_data(df, price_monitor_manager):
    """
    Verifies and analyzes OHLCV data to identify assets with significant price increases and volume.

    Args:
        df (pandas.DataFrame): DataFrame with OHLCV data and 'symbol' column.

    Returns:
        tuple: (above_threshold_data, percent_changes)
            - above_threshold_data (list): List of dictionaries for assets meeting thresholds.
            - percent_changes (pandas.DataFrame): DataFrame with price changes and OHLCV data.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return [], pd.DataFrame()

    current_time = datetime.utcnow()
    ten_min_ago = current_time - timedelta(minutes=10)
    five_min_ago = current_time - timedelta(minutes=5)

    if df["timestamp"].max() < ten_min_ago:
        logger.warning("Data contains no candles from within the last 10 minutes.")
        return [], pd.DataFrame()

    recent_data = df[df["timestamp"] >= five_min_ago]
    if recent_data.empty:
        logger.warning("No recent data within the last 5 minutes.")
        return [], pd.DataFrame()

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

    if not above_threshold.empty:
        logger.info(
            f"\nCoins with price increase >= {PRICE_INCREASE_THRESHOLD}% and volume >= €{MIN_VOLUME_EUR}:"
        )
        for _, row in above_threshold.iterrows():
            logger.info(
                f"Symbol: {row['symbol']}, Change: {row['percent_change']:.2f}%, "
                f"Open: {row['open_price']:.2f}, Close: {row['close_price']:.2f}, "
                f"Volume: {row['volume_eur']:.2f} EUR, Timestamp: {row['latest_timestamp']}"
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
            logger.info(
                f"Symbol: {row['symbol']}, Change: {row['percent_change']:.2f}%, "
                f"Open: {row['open_price']:.2f}, Close: {row['close_price']:.2f}, "
                f"Volume: {row['volume_eur']:.2f} EUR, Timestamp: {row['latest_timestamp']}"
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

    return above_threshold.to_dict("records"), percent_changes
