# trading_bot/src/storage.py
import os
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json

from .config import PORTFOLIO_FILE, RESULTS_FOLDER, logger
from .state import portfolio, portfolio_lock

def save_to_local(df, output_path):
    """
    Saves a DataFrame to a local Parquet file, appending to existing data.

    Args:
        df (pandas.DataFrame): DataFrame to save.
        output_path (str): Path to the Parquet file.
    """
    if df.empty:
        logger.warning(f"Attempted to save empty DataFrame to {output_path}. Skipping.")
        return
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            existing_df = pq.read_table(output_path).to_pandas()
            combined_df = pd.concat([existing_df, df]).drop_duplicates(
                subset=["timestamp", "symbol"], keep="last"
            )
        else:
            combined_df = df
        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, output_path)
        logger.info(f"Saved {len(df)} records to {output_path}")
    except Exception as e:
        logger.error(f"Error saving to {output_path}: {e}", exc_info=True)

def save_portfolio():
    """
    Saves the current portfolio state to a JSON file.
    """
    try:
        with portfolio_lock:
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
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio_copy, f, indent=4)
        logger.info(f"Saved portfolio to {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(f"Error saving portfolio to {PORTFOLIO_FILE}: {e}", exc_info=True)