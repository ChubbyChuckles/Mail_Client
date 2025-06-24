# trading_bot/src/storage.py
import os
import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from .config import (
    RESULTS_FOLDER,
    PORTFOLIO_FILE,
    GOOGLE_SHEETS_CREDENTIALS,
    SPREADSHEET_NAME,
    ACTIVE_ASSETS_SHEET,
    FINISHED_TRADES_SHEET,
    logger,
)
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
                subset=['timestamp', 'symbol'], keep='last'
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
                'cash': portfolio['cash'],
                'assets': {
                    symbol: {
                        key: value.isoformat() if isinstance(value, datetime) else value
                        for key, value in asset.items()
                    }
                    for symbol, asset in portfolio['assets'].items()
                }
            }
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio_copy, f, indent=4)
        logger.info(f"Saved portfolio to {PORTFOLIO_FILE}")
    except Exception as e:
        logger.error(f"Error saving portfolio to {PORTFOLIO_FILE}: {e}", exc_info=True)

def write_to_google_sheets(data, credentials_file, spreadsheet_name, sheet_name, is_active_assets=False, is_finished_trades=False):
    """
    Writes data to a Google Sheet.

    Args:
        data: List or dict of data to write.
        credentials_file (str): Path to Google Sheets credentials JSON.
        spreadsheet_name (str): Name of the Google Sheet.
        sheet_name (str): Name of the worksheet.
        is_active_assets (bool): If True, formats data as active assets.
        is_finished_trades (bool): If True, formats data as finished trades.
    """
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(sheet_name)
        if is_active_assets:
            if not data:
                worksheet.clear()
                worksheet.append_row([
                    'Symbol', 'Quantity', 'Buy Price', 'Buy Time', 'Current Price',
                    'Highest Price', 'Profit Target', 'Sell Price', 'Take Action'
                ])
                logger.info(f"Cleared and initialized {sheet_name} with headers")
                return
            records = [
                [
                    symbol,
                    f"{asset['quantity']:.8f}",
                    f"{asset['purchase_price']:.2f}",
                    asset['purchase_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    f"{asset['current_price']:.2f}",
                    f"{asset['highest_price']:.2f}",
                    f"{asset['profit_target']:.4f}",
                    f"{asset['sell_price']:.2f}",
                    asset.get('take_action', '')
                ]
                for symbol, asset in data.items()
            ]
            worksheet.clear()
            worksheet.append_row([
                'Symbol', 'Quantity', 'Buy Price', 'Buy Time', 'Current Price',
                'Highest Price', 'Profit Target', 'Sell Price', 'Take Action'
            ])
            worksheet.append_rows(records)
            logger.info(f"Wrote {len(records)} active assets to {sheet_name}")
        elif is_finished_trades:
            if not data:
                logger.debug(f"No finished trades to write to {sheet_name}")
                return
            records = [
                [
                    trade['Symbol'],
                    trade['Buy Quantity'],
                    trade['Buy Price'],
                    trade['Buy Time'],
                    trade['Buy Fee'],
                    trade['Sell Quantity'],
                    trade['Sell Price'],
                    trade['Sell Time'],
                    trade['Sell Fee'],
                    trade['Profit/Loss']
                ]
                for trade in data
            ]
            worksheet.append_rows(records)
            logger.info(f"Wrote {len(records)} finished trades to {sheet_name}")
        else:
            logger.error("Invalid write mode for Google Sheets. Specify is_active_assets or is_finished_trades.")
    except Exception as e:
        logger.error(f"Error writing to Google Sheet {sheet_name}: {e}", exc_info=True)

def load_active_assets(credentials_file, spreadsheet_name, sheet_name):
    """
    Loads active assets from a Google Sheet.

    Args:
        credentials_file (str): Path to Google Sheets credentials JSON.
        spreadsheet_name (str): Name of the Google Sheet.
        sheet_name (str): Name of the worksheet.

    Returns:
        dict: Dictionary of active assets with symbol as key.
    """
    try:
        scope = [
            'https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet(sheet_name)
        data = worksheet.get_all_values()
        if not data or len(data) < 2:
            logger.info(f"No active assets found in {sheet_name}")
            return {}
        headers = data[0]
        records = data[1:]
        assets = {}
        for row in records:
            try:
                symbol = row[headers.index('Symbol')]
                assets[symbol] = {
                    'quantity': float(row[headers.index('Quantity')]),
                    'purchase_price': float(row[headers.index('Buy Price')]),
                    'purchase_time': datetime.strptime(row[headers.index('Buy Time')], '%Y-%m-%d %H:%M:%S'),
                    'current_price': float(row[headers.index('Current Price')]),
                    'highest_price': float(row[headers.index('Highest Price')]),
                    'profit_target': float(row[headers.index('Profit Target')]),
                    'original_profit_target': float(row[headers.index('Profit Target')]),
                    'sell_price': float(row[headers.index('Sell Price')]),
                    'take_action': row[headers.index('Take Action')]
                }
            except (IndexError, ValueError) as e:
                logger.warning(f"Invalid row in {sheet_name}: {row}. Error: {e}")
                continue
        logger.info(f"Loaded {len(assets)} active assets from {sheet_name}")
        return assets
    except Exception as e:
        logger.error(f"Error loading active assets from {sheet_name}: {e}", exc_info=True)
        return {}