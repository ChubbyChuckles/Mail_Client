# trading_bot/src/utils.py
import csv
import os
from datetime import datetime
import pandas as pd
import tempfile
import shutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from . import config
from .config import logger

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(OSError),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying file operation after {retry_state.attempt_number} attempts"
    ),
    reraise=True
)
def move_file_with_retry(src, dst):
    """Helper function to move a file with retries for OSError."""
    shutil.move(src, dst)

def calculate_ema(prices, period):
    """
    Calculates the Exponential Moving Average (EMA) for a given price series.

    Args:
        prices (list or pandas.Series): List of prices.
        period (int): EMA period.

    Returns:
        float: The EMA value, or None if insufficient data or an error occurs.

    Raises:
        ValueError: If prices is empty or period is invalid.
        TypeError: If prices contains non-numeric values.
    """
    try:
        if not prices:
            raise ValueError("Prices list is empty")
        if not isinstance(period, int) or period <= 0:
            raise ValueError(f"Invalid EMA period: {period}")
        
        prices_series = pd.Series(prices)
        if len(prices_series) < period:
            logger.warning(
                f"Insufficient data for EMA calculation: {len(prices_series)} prices, need {period}"
            )
            return None
        
        if not pd.api.types.is_numeric_dtype(prices_series):
            raise TypeError("Prices contain non-numeric values")
        
        ema = prices_series.ewm(span=period, adjust=False).mean().iloc[-1]
        logger.debug(f"Calculated EMA for period {period}: {ema:.4f}")
        return ema
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating EMA: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error calculating EMA: {e}", exc_info=True)
        send_alert("EMA Calculation Failure", f"Unexpected error calculating EMA: {e}")
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
        int: The dynamic EMA period, or default (5) if an error occurs.

    Raises:
        ValueError: If inputs are invalid (negative or non-numeric).
        TypeError: If inputs are of incorrect type.
    """
    try:
        if not all(isinstance(x, (int, float)) for x in [holding_minutes, time_stop_minutes, active_assets, asset_threshold]):
            raise TypeError("All inputs must be numeric")
        if any(x < 0 for x in [holding_minutes, time_stop_minutes, active_assets, asset_threshold]):
            raise ValueError("Inputs cannot be negative")
        
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
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating dynamic EMA period: {e}", exc_info=True)
        return 5
    except Exception as e:
        logger.error(f"Unexpected error calculating dynamic EMA period: {e}", exc_info=True)
        send_alert("Dynamic EMA Period Failure", f"Unexpected error calculating dynamic EMA period: {e}")
        return 5

def append_to_buy_trades_csv(trade_data):
    """
    Appends buy trade data to a CSV file without overwriting existing data.

    Args:
        trade_data (dict): Dictionary containing trade details with required fields.

    Raises:
        ValueError: If trade_data is missing required fields.
        OSError: For file operation errors.
    """
    try:
        required_fields = [
            "Symbol", "Buy Quantity", "Buy Price", "Buy Time", "Buy Fee", 'Buy Slippage', 'Actual Cost',
            "Allocation", "Trade Count", "Largest Trade Volume EUR"
        ]
        if not all(field in trade_data for field in required_fields):
            raise ValueError(f"trade_data missing required fields: {set(required_fields) - set(trade_data.keys())}")

        file_path = config.config.BUY_TRADES_CSV
        if not file_path or not os.path.basename(file_path):
            raise ValueError(f"Invalid BUY_TRADES_CSV path: {file_path}")
        
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        # Check if file exists and read existing data
        existing_data = []
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            with open(file_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                if set(reader.fieldnames) != set(required_fields):
                    logger.warning(f"Fieldnames mismatch in {file_path}. Existing: {reader.fieldnames}")
                existing_data = [row for row in reader]
        
        # Append new trade data
        existing_data.append(trade_data)
        
        # Write all data back to a temporary file and move it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.tmp') as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=required_fields)
            writer.writeheader()
            writer.writerows(existing_data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        move_file_with_retry(temp_file.name, file_path)
        logger.info(f"Appended buy trade for {trade_data['Symbol']} to {file_path}")
    except (ValueError, OSError) as e:
        logger.error(f"Error appending to {config.config.BUY_TRADES_CSV}: {e}", exc_info=True)
        send_alert("Buy Trades CSV Failure", f"Error appending to buy trades CSV: {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to {config.config.BUY_TRADES_CSV}: {e}", exc_info=True)
        send_alert("Buy Trades CSV Failure", f"Unexpected error appending to buy trades CSV: {e}")

def append_to_finished_trades_csv(trade_data):
    """
    Appends finished trade data to a CSV file without overwriting existing data.

    Args:
        trade_data (dict): Dictionary containing finished trade details.

    Raises:
        ValueError: If trade_data is missing required fields.
        OSError: For file operation errors.
    """
    try:
        required_fields = [
            "Symbol", "Buy Quantity", "Buy Price", "Buy Time", "Buy Fee",
            "Sell Quantity", "Sell Price", "Sell Time", "Sell Fee", "Sell Slippage", "Profit/Loss", "Reason"
        ]
        if not all(field in trade_data for field in required_fields):
            raise ValueError(f"trade_data missing required fields: {set(required_fields) - set(trade_data.keys())}")

        file_path = config.config.FINISHED_TRADES_CSV
        if not file_path or not os.path.basename(file_path):
            raise ValueError(f"Invalid FINISHED_TRADES_CSV path: {file_path}")
        
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        # Check if file exists and read existing data
        existing_data = []
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            with open(file_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                if set(reader.fieldnames) != set(required_fields):
                    logger.warning(f"Fieldnames mismatch in {file_path}. Existing: {reader.fieldnames}")
                existing_data = [row for row in reader]
        
        # Append new trade data
        existing_data.append(trade_data)
        
        # Write all data back to a temporary file and move it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.tmp') as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=required_fields)
            writer.writeheader()
            writer.writerows(existing_data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        move_file_with_retry(temp_file.name, file_path)
        logger.info(f"Appended finished trade for {trade_data['Symbol']} to {file_path}")
    except (ValueError, OSError) as e:
        logger.error(f"Error appending to {config.config.FINISHED_TRADES_CSV}: {e}", exc_info=True)
        send_alert("Finished Trades CSV Failure", f"Error appending to finished trades CSV: {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to {config.config.FINISHED_TRADES_CSV}: {e}", exc_info=True)
        send_alert("Finished Trades CSV Failure", f"Unexpected error appending to finished trades CSV: {e}")

def append_to_order_book_metrics_csv(metrics_list):
    """
    Appends order book metrics to a CSV file without overwriting existing data.

    Args:
        metrics_list (list): List of dictionaries containing order book metrics and buy status.

    Raises:
        ValueError: If metrics_list is empty or contains invalid data.
        OSError: For file operation errors.
    """
    try:
        if not metrics_list:
            raise ValueError("metrics_list is empty")

        file_path = config.config.ORDER_BOOK_METRICS_CSV
        if not file_path or not os.path.basename(file_path):
            raise ValueError(f"Invalid ORDER_BOOK_METRICS_CSV path: {file_path}")

        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        
        fieldnames = [
            "Timestamp", "Market", "Nonce", "Best_Bid", "Best_Ask", "Spread",
            "Spread_Percentage", "Mid_Price", "Buy_Depth", "Sell_Depth", "Total_Depth",
            "Bid_Volume", "Ask_Volume", "Bid_Value", "Ask_Value", "Order_Book_Imbalance",
            "Bid_Levels_Count", "Ask_Levels_Count", "Avg_Bid_Price", "Avg_Ask_Price",
            "VWAP_Bid", "VWAP_Ask", "Slippage_Buy", "Predicted_Price_Buy",
            "Slippage_Sell", "Predicted_Price_Sell", "Bought", "Error"
        ]
        
        # Check if file exists and read existing data
        existing_data = []
        file_exists = os.path.exists(file_path)
        
        if file_exists:
            with open(file_path, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                if set(reader.fieldnames) != set(fieldnames):
                    logger.warning(f"Fieldnames mismatch in {file_path}. Existing: {reader.fieldnames}")
                existing_data = [row for row in reader]
        
        # Prepare new rows
        new_rows = []
        for metrics in metrics_list:
            if not isinstance(metrics, dict):
                logger.warning(f"Skipping invalid metrics entry: {metrics}")
                continue
            row = {
                "Timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "Market": metrics.get("market"),
                "Nonce": metrics.get("nonce"),
                "Best_Bid": (
                    f"{metrics['best_bid']:.2f}"
                    if metrics.get("best_bid") is not None else None
                ),
                "Best_Ask": (
                    f"{metrics['best_ask']:.2f}"
                    if metrics.get("best_ask") is not None else None
                ),
                "Spread": (
                    f"{metrics['spread']:.2f}"
                    if metrics.get("spread") is not None else None
                ),
                "Spread_Percentage": (
                    f"{metrics['spread_percentage']:.2f}"
                    if metrics.get("spread_percentage") is not None else None
                ),
                "Mid_Price": (
                    f"{metrics['mid_price']:.2f}"
                    if metrics.get("mid_price") is not None else None
                ),
                "Buy_Depth": (
                    f"{metrics['buy_depth']:.2f}"
                    if metrics.get("buy_depth") is not None else None
                ),
                "Sell_Depth": (
                    f"{metrics['sell_depth']:.2f}"
                    if metrics.get("sell_depth") is not None else None
                ),
                "Total_Depth": (
                    f"{metrics['total_depth']:.2f}"
                    if metrics.get("total_depth") is not None else None
                ),
                "Bid_Volume": (
                    f"{metrics['bid_volume']:.8f}"
                    if metrics.get("bid_volume") is not None else None
                ),
                "Ask_Volume": (
                    f"{metrics['ask_volume']:.8f}"
                    if metrics.get("ask_volume") is not None else None
                ),
                "Bid_Value": (
                    f"{metrics['bid_value']:.2f}"
                    if metrics.get("bid_value") is not None else None
                ),
                "Ask_Value": (
                    f"{metrics['ask_value']:.2f}"
                    if metrics.get("ask_value") is not None else None
                ),
                "Order_Book_Imbalance": (
                    f"{metrics['order_book_imbalance']:.2f}"
                    if metrics.get("order_book_imbalance") is not None else None
                ),
                "Bid_Levels_Count": metrics.get("bid_levels_count"),
                "Ask_Levels_Count": metrics.get("ask_levels_count"),
                "Avg_Bid_Price": (
                    f"{metrics['avg_bid_price']:.2f}"
                    if metrics.get("avg_bid_price") is not None else None
                ),
                "Avg_Ask_Price": (
                    f"{metrics['avg_ask_price']:.2f}"
                    if metrics.get("avg_ask_price") is not None else None
                ),
                "VWAP_Bid": (
                    f"{metrics['vwap_bid']:.2f}"
                    if metrics.get("vwap_bid") is not None else None
                ),
                "VWAP_Ask": (
                    f"{metrics['vwap_ask']:.2f}"
                    if metrics.get("vwap_ask") is not None else None
                ),
                "Slippage_Buy": (
                    f"{metrics['slippage_buy']:.2f}"
                    if metrics.get("slippage_buy") is not None else None
                ),
                "Predicted_Price_Buy": (
                    f"{metrics['predicted_price_buy']:.2f}"
                    if metrics.get("predicted_price_buy") is not None else None
                ),
                "Slippage_Sell": (
                    f"{metrics['slippage_sell']:.2f}"
                    if metrics.get("slippage_sell") is not None else None
                ),
                "Predicted_Price_Sell": (
                    f"{metrics['predicted_price_sell']:.2f}"
                    if metrics.get("predicted_price_sell") is not None else None
                ),
                "Bought": metrics.get("bought", False),
                "Error": metrics.get("error")
            }
            new_rows.append(row)
        
        # Combine existing and new data
        existing_data.extend(new_rows)
        
        # Write all data back to a temporary file and move it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='', suffix='.tmp') as temp_file:
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_data)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        move_file_with_retry(temp_file.name, file_path)
        logger.info(f"Appended {len(new_rows)} order book metrics to {file_path}")
    except (ValueError, OSError) as e:
        logger.error(f"Error appending to {config.config.ORDER_BOOK_METRICS_CSV}: {e}", exc_info=True)
        send_alert("Order Book Metrics CSV Failure", f"Error appending to order book metrics CSV: {e}")
    except Exception as e:
        logger.error(f"Unexpected error appending to {config.config.ORDER_BOOK_METRICS_CSV}: {e}", exc_info=True)
        send_alert("Order Book Metrics CSV Failure", f"Unexpected error appending to order book metrics CSV: {e}")

def send_alert(subject, message):
    """
    Sends an alert for critical errors (placeholder).

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(f"ALERT: {subject} - {message}")