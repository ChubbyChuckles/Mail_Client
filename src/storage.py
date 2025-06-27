# trading_bot/src/storage.py
import json
import os
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import shutil
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from . import config
from .config import logger
from .state import portfolio, portfolio_lock

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

def save_to_local(df, output_path):
    """
    Saves a DataFrame to a local Parquet file, appending to existing data atomically.

    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path to the Parquet file.

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
        OSError: For file operation errors.
        pyarrow.lib.ArrowException: For Parquet-related errors.
    """
    try:
        if df.empty:
            logger.warning(f"Attempted to save empty DataFrame to {output_path}. Skipping.")
            return
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_columns = {"timestamp", "symbol"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame missing required columns: {required_columns - set(df.columns)}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        temp_file.close()  # Explicitly close the file to release the handle
        try:
            if os.path.exists(output_path):
                try:
                    existing_df = pq.read_table(output_path).to_pandas()
                    combined_df = pd.concat([existing_df, df]).drop_duplicates(
                        subset=["timestamp", "symbol"], keep="last"
                    )
                except (pa.lib.ArrowException, pd.errors.EmptyDataError) as e:
                    logger.warning(f"Error reading existing Parquet file {output_path}: {e}. Using new DataFrame.")
                    combined_df = df
            else:
                combined_df = df

            table = pa.Table.from_pandas(combined_df, preserve_index=False)
            pq.write_table(table, temp_file.name)
            move_file_with_retry(temp_file.name, output_path)
            # logger.info(f"Saved {len(df)} records to {output_path}")
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError as e:
                    logger.warning(f"Error cleaning up temporary file {temp_file.name}: {e}", exc_info=True)
    except ValueError as e:
        logger.error(f"Validation error saving to {output_path}: {e}", exc_info=True)
        send_alert("Parquet Save Failure", f"Validation error saving DataFrame to {output_path}: {e}")
    except OSError as e:
        logger.error(f"File operation error saving to {output_path}: {e}", exc_info=True)
        send_alert("Parquet Save Failure", f"File operation error saving DataFrame to {output_path}: {e}")
    except pa.lib.ArrowException as e:
        logger.error(f"Parquet error saving to {output_path}: {e}", exc_info=True)
        send_alert("Parquet Save Failure", f"Parquet error saving DataFrame to {output_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving to {output_path}: {e}", exc_info=True)
        send_alert("Parquet Save Failure", f"Unexpected error saving DataFrame to {output_path}: {e}")

def save_portfolio():
    """
    Saves the current portfolio state to a JSON file atomically.

    Raises:
        ValueError: If portfolio data is invalid.
        OSError: For file operation errors.
    """
    try:
        if not isinstance(portfolio, dict) or "cash" not in portfolio or "assets" not in portfolio:
            raise ValueError("Invalid portfolio structure")

        if not portfolio_lock.acquire(timeout=5):
            logger.error("Timeout acquiring portfolio lock")
            send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
            return

        try:
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
            file_path = config.config.PORTFOLIO_FILE
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
                json.dump(portfolio_copy, temp_file, indent=4)
                temp_file.flush()  # Ensure data is written to disk
                os.fsync(temp_file.fileno())  # Ensure data is flushed to disk on Windows
            
            move_file_with_retry(temp_file.name, file_path)
            logger.info(f"Saved portfolio to {file_path}")
        finally:
            portfolio_lock.release()
    except (ValueError, OSError, TypeError) as e:
        logger.error(f"Error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}", exc_info=True)
        send_alert("Portfolio Save Failure", f"Error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}", exc_info=True)
        send_alert("Portfolio Save Failure", f"Unexpected error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}")

def send_alert(subject, message):
    """
    Sends an alert for critical errors (placeholder).

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(f"ALERT: {subject} - {message}")