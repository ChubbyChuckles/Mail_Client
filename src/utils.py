import csv
import os

import pandas as pd

from .config import ORDER_BOOK_METRICS_CSV, BUY_TRADES_CSV, FINISHED_TRADES_CSV, logger
from datetime import datetime

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

def append_to_order_book_metrics_csv(metrics_list):
    """
    Appends order book metrics to a CSV file.

    Args:
        metrics_list (list): List of dictionaries containing order book metrics and buy status.
    """
    try:
        file_exists = os.path.exists(ORDER_BOOK_METRICS_CSV)
        fieldnames = [
            "Timestamp",
            "Market",
            "Nonce",
            "Best_Bid",
            "Best_Ask",
            "Spread",
            "Spread_Percentage",
            "Mid_Price",
            "Buy_Depth",
            "Sell_Depth",
            "Total_Depth",
            "Bid_Volume",
            "Ask_Volume",
            "Bid_Value",
            "Ask_Value",
            "Order_Book_Imbalance",
            "Bid_Levels_Count",
            "Ask_Levels_Count",
            "Avg_Bid_Price",
            "Avg_Ask_Price",
            "VWAP_Bid",
            "VWAP_Ask",
            "Slippage_Buy",
            "Predicted_Price_Buy",
            "Slippage_Sell",
            "Predicted_Price_Sell",
            "Bought",
            "Error",
        ]
        with open(ORDER_BOOK_METRICS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for metrics in metrics_list:
                row = {
                    "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "Market": metrics.get("market"),
                    "Nonce": metrics.get("nonce"),
                    "Best_Bid": (
                        f"{metrics['best_bid']:.2f}" if metrics.get("best_bid") else None
                    ),
                    "Best_Ask": (
                        f"{metrics['best_ask']:.2f}" if metrics.get("best_ask") else None
                    ),
                    "Spread": (
                        f"{metrics['spread']:.2f}" if metrics.get("spread") else None
                    ),
                    "Spread_Percentage": (
                        f"{metrics['spread_percentage']:.2f}"
                        if metrics.get("spread_percentage")
                        else None
                    ),
                    "Mid_Price": (
                        f"{metrics['mid_price']:.2f}" if metrics.get("mid_price") else None
                    ),
                    "Buy_Depth": (
                        f"{metrics['buy_depth']:.2f}" if metrics.get("buy_depth") else None
                    ),
                    "Sell_Depth": (
                        f"{metrics['sell_depth']:.2f}" if metrics.get("sell_depth") else None
                    ),
                    "Total_Depth": (
                        f"{metrics['total_depth']:.2f}" if metrics.get("total_depth") else None
                    ),
                    "Bid_Volume": (
                        f"{metrics['bid_volume']:.8f}" if metrics.get("bid_volume") else None
                    ),
                    "Ask_Volume": (
                        f"{metrics['ask_volume']:.8f}" if metrics.get("ask_volume") else None
                    ),
                    "Bid_Value": (
                        f"{metrics['bid_value']:.2f}" if metrics.get("bid_value") else None
                    ),
                    "Ask_Value": (
                        f"{metrics['ask_value']:.2f}" if metrics.get("ask_value") else None
                    ),
                    "Order_Book_Imbalance": (
                        f"{metrics['order_book_imbalance']:.2f}"
                        if metrics.get("order_book_imbalance")
                        else None
                    ),
                    "Bid_Levels_Count": metrics.get("bid_levels_count"),
                    "Ask_Levels_Count": metrics.get("ask_levels_count"),
                    "Avg_Bid_Price": (
                        f"{metrics['avg_bid_price']:.2f}"
                        if metrics.get("avg_bid_price")
                        else None
                    ),
                    "Avg_Ask_Price": (
                        f"{metrics['avg_ask_price']:.2f}"
                        if metrics.get("avg_ask_price")
                        else None
                    ),
                    "VWAP_Bid": (
                        f"{metrics['vwap_bid']:.2f}" if metrics.get("vwap_bid") else None
                    ),
                    "VWAP_Ask": (
                        f"{metrics['vwap_ask']:.2f}" if metrics.get("vwap_ask") else None
                    ),
                    "Slippage_Buy": (
                        f"{metrics['slippage_buy']:.2f}"
                        if metrics.get("slippage_buy")
                        else None
                    ),
                    "Predicted_Price_Buy": (
                        f"{metrics['predicted_price_buy']:.2f}"
                        if metrics.get("predicted_price_buy")
                        else None
                    ),
                    "Slippage_Sell": (
                        f"{metrics['slippage_sell']:.2f}"
                        if metrics.get("slippage_sell")
                        else None
                    ),
                    "Predicted_Price_Sell": (
                        f"{metrics['predicted_price_sell']:.2f}"
                        if metrics.get("predicted_price_sell")
                        else None
                    ),
                    "Bought": metrics.get("bought", False),
                    "Error": metrics.get("error"),
                }
                writer.writerow(row)
        logger.info(
            f"Appended {len(metrics_list)} order book metrics to {ORDER_BOOK_METRICS_CSV}"
        )
    except Exception as e:
        logger.error(f"Error appending to {ORDER_BOOK_METRICS_CSV}: {e}")