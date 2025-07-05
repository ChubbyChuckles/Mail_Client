# trading_bot/src/storage.py
import glob
import json
import os
import shutil
import tempfile
import time
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from . import config
from .config import IS_GITHUB_ACTIONS, logger
from .state import portfolio, portfolio_lock


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(OSError),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying file operation after {retry_state.attempt_number} attempts"
    ),
    reraise=True,
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
            logger.warning(
                f"Attempted to save empty DataFrame to {output_path}. Skipping."
            )
            return
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        required_columns = {"timestamp", "symbol"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"DataFrame missing required columns: {required_columns - set(df.columns)}"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        temp_file.close()  # Explicitly close the file to release the handle
        try:
            if os.path.exists(output_path):
                try:
                    existing_df = pq.read_table(output_path).to_pandas()
                    combined_df = pd.concat([existing_df, df]).drop_duplicates(
                        subset=["timestamp", "symbol"], keep="last"
                    )
                except (pa.lib.ArrowException, pd.errors.EmptyDataError) as e:
                    logger.warning(
                        f"Error reading existing Parquet file {output_path}: {e}. Using new DataFrame."
                    )
                    combined_df = df
            else:
                combined_df = df

            table = pa.Table.from_pandas(combined_df, preserve_index=False)
            pq.write_table(table, temp_file.name)
            move_file_with_retry(temp_file.name, output_path)
            logger.info(f"Saved {len(df)} records to {output_path}")

            # In GitHub Actions, log file creation for artifact tracking
            if IS_GITHUB_ACTIONS:
                logger.info(
                    f"Parquet file saved at {output_path} for artifact collection"
                )
        finally:
            if os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError as e:
                    logger.warning(
                        f"Error cleaning up temporary file {temp_file.name}: {e}",
                        exc_info=True,
                    )
    except ValueError as e:
        logger.error(f"Validation error saving to {output_path}: {e}", exc_info=True)
        send_alert(
            "Parquet Save Failure",
            f"Validation error saving DataFrame to {output_path}: {e}",
        )
    except OSError as e:
        logger.error(
            f"File operation error saving to {output_path}: {e}", exc_info=True
        )
        send_alert(
            "Parquet Save Failure",
            f"File operation error saving DataFrame to {output_path}: {e}",
        )
    except pa.lib.ArrowException as e:
        logger.error(f"Parquet error saving to {output_path}: {e}", exc_info=True)
        send_alert(
            "Parquet Save Failure",
            f"Parquet error saving DataFrame to {output_path}: {e}",
        )
    except Exception as e:
        logger.error(f"Unexpected error saving to {output_path}: {e}", exc_info=True)
        send_alert(
            "Parquet Save Failure",
            f"Unexpected error saving DataFrame to {output_path}: {e}",
        )


def save_portfolio():
    """
    Saves the current portfolio state to a JSON file and maintains only the 3 latest backup files atomically.

    Raises:
        ValueError: If portfolio data is invalid.
        OSError: For file operation errors.
    """
    try:
        if (
            not isinstance(portfolio, dict)
            or "cash" not in portfolio
            or "assets" not in portfolio
        ):
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
            if not file_path or not os.path.basename(file_path):
                raise ValueError(f"Invalid PORTFOLIO_FILE path: {file_path}")

            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as temp_file:
                json.dump(portfolio_copy, temp_file, indent=4)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            move_file_with_retry(temp_file.name, file_path)
            logger.info(f"Saved portfolio to {file_path}")

            # Save to backup file, but skip in GitHub Actions to reduce disk usage
            if not IS_GITHUB_ACTIONS:
                backup_file = (
                    f"{file_path}.backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                )
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json"
                ) as temp_file:
                    json.dump(portfolio_copy, temp_file, indent=4)
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                move_file_with_retry(temp_file.name, backup_file)
                logger.info(f"Saved portfolio backup to {backup_file}")

                # Manage backup files: keep only the 3 latest
                backup_files = glob.glob(f"{file_path}.backup_*")
                backup_files.sort(key=lambda x: x.split("backup_")[-1], reverse=True)
                for old_file in backup_files[3:]:
                    try:
                        os.remove(old_file)
                        logger.debug(f"Deleted old backup file: {old_file}")
                    except OSError as e:
                        logger.warning(
                            f"Error deleting old backup file {old_file}: {e}",
                            exc_info=True,
                        )

            # In GitHub Actions, log file creation for artifact tracking
            if IS_GITHUB_ACTIONS:
                logger.info(
                    f"Portfolio file saved at {file_path} for artifact collection"
                )
        finally:
            portfolio_lock.release()
    except ValueError as e:
        logger.error(
            f"Validation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
            exc_info=True,
        )
        send_alert(
            "Portfolio Save Failure",
            f"Validation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
        )
    except OSError as e:
        logger.error(
            f"File operation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
            exc_info=True,
        )
        send_alert(
            "Portfolio Save Failure",
            f"File operation error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
        )
    except Exception as e:
        logger.error(
            f"Unexpected error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
            exc_info=True,
        )
        send_alert(
            "Portfolio Save Failure",
            f"Unexpected error saving portfolio to {config.config.PORTFOLIO_FILE}: {e}",
        )


def send_alert(subject, message):
    """
    Sends an alert for critical errors using TelegramNotifier.

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(message)
