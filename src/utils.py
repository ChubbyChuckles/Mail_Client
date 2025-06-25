import csv
import os

import pandas as pd

from .config import BUY_TRADES_CSV, FINISHED_TRADES_CSV, logger

def calculate_ema(prices, period):
    """
    Calculates the Exponential Moving Average (EMA) for a given price series.

    Args:
        prices (list or pandas.Series): List of prices.
        period (int): EMA period.

    Returns:
        float: The EMA value, or None if insufficient data.
    """
    try:
        prices_series = pd.Series(prices)
        if len(prices_series) < period:
            logger.warning(
                f"Insufficient data for EMA calculation: {len(prices_series)} prices, need {period}"
            )
            return None
        ema = prices_series.ewm(span=period, adjust=False).mean().iloc[-1]
        logger.debug(f"Calculated EMA for period {period}: {ema:.4f}")
        return ema
    except Exception as e:
        logger.error(f"Error calculating EMA: {e}")
        return None

def calculate_dynamic_ema_period(
    holding_minutes, time_stop_minutes, active_assets, asset_threshold
):
    """
    Calculates a dynamic EMA period based on holding time and portfolio size.

    Args:
        holding_minutes (float): Minutes since asset purchase.
        time_stop_minutes (int): Time stop threshold in minutes.
        active_assets (int): Number of active assets in portfolio.
        asset_threshold (int): Portfolio size threshold for adjustments.

    Returns:
        int: The dynamic EMA period.
    """
    try:
        base_period = 5
        if active_assets >= asset_threshold:
            base_period = min(base_period + 2, 10)
        if holding_minutes >= time_stop_minutes * 0.5:
            base_period = min(base_period + 3, 12)
        period = max(2, base_period)
        logger.debug(
            f"Calculated dynamic EMA period: {period} (holding: {holding_minutes:.2f} min, "
            f"active assets: {active_assets}, threshold: {asset_threshold})"
        )
        return period
    except Exception as e:
        logger.error(f"Error calculating dynamic EMA period: {e}")
        return 5

def append_to_buy_trades_csv(trade_data):
    """
    Appends buy trade data to a CSV file.

    Args:
        trade_data (dict): Dictionary containing trade details.
    """
    try:
        file_exists = os.path.exists(BUY_TRADES_CSV)
        with open(BUY_TRADES_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Symbol",
                    "Buy Quantity",
                    "Buy Price",
                    "Buy Time",
                    "Buy Fee",
                    "Allocation",
                    "Trade Count",
                    "Largest Trade Volume EUR",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
        logger.info(
            f"Appended buy trade for {trade_data['Symbol']} to {BUY_TRADES_CSV}"
        )
    except Exception as e:
        logger.error(f"Error appending to {BUY_TRADES_CSV}: {e}")

def append_to_finished_trades_csv(trade_data):
    """
    Appends finished trade data to a CSV file.

    Args:
        trade_data (dict): Dictionary containing finished trade details.
    """
    try:
        file_exists = os.path.exists(FINISHED_TRADES_CSV)
        with open(FINISHED_TRADES_CSV, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Symbol",
                    "Buy Quantity",
                    "Buy Price",
                    "Buy Time",
                    "Buy Fee",
                    "Sell Quantity",
                    "Sell Price",
                    "Sell Time",
                    "Sell Fee",
                    "Profit/Loss",
                    "Reason",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade_data)
        logger.info(
            f"Appended finished trade for {trade_data['Symbol']} to {FINISHED_TRADES_CSV}"
        )
    except Exception as e:
        logger.error(f"Error appending to {FINISHED_TRADES_CSV}: {e}")