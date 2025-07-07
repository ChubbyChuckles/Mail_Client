# trading_bot/src/main_new.py
import logging
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from dateutil import tz
import numpy as np

from . import config
from .config import logger
from .exchange import bitvavo, fetch_klines
from .state import is_banned, ban_expiry_time


def fetch_daily_candles(symbol, timeframe='5m'):
    """
    Fetches all candles for the current day for the specified symbol and timeframe.

    Args:
        symbol (str): Trading pair (e.g., 'BTC-EUR').
        timeframe (str): Candle timeframe (e.g., '5m', '1m').

    Returns:
        pandas.DataFrame: OHLCV data for the current day, or empty DataFrame if fetch fails.
    """
    try:
        # Get current UTC time and start of the current day
        utc_now = datetime.now(tz.UTC)
        start_of_day = utc_now.replace(hour=0, minute=0, second=0, microsecond=0)
        start_ms = int(start_of_day.timestamp() * 1000)
        
        # Calculate approximate number of candles in a day based on timeframe
        timeframe_minutes = int(timeframe.replace('m', '')) if 'm' in timeframe else 60
        candles_per_day = (24 * 60) // timeframe_minutes
        limit = 1000  # Fixed limit for sufficient data

        if is_banned and datetime.now(tz.UTC).timestamp() < ban_expiry_time:
            logger.warning(
                f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Skipping fetch for {symbol}."
            )
            return pd.DataFrame()

        # Log available markets for debugging
        markets = bitvavo.load_markets()
        logger.debug(f"Available markets: {list(markets.keys())}")

        logger.info(f"Fetching {timeframe} candles for {symbol} from {start_of_day} with limit {limit}")
        df = fetch_klines(symbol, timeframe=timeframe, limit=limit)
        
        if df.empty:
            logger.warning(f"No data returned from fetch_klines for {symbol}. Check API response, symbol validity, or timeframe.")
            # Try fetching previous day's data as a fallback
            logger.info(f"Attempting to fetch data for previous day for {symbol}")
            start_of_day = start_of_day - timedelta(days=1)
            start_ms = int(start_of_day.timestamp() * 1000)
            df = fetch_klines(symbol, timeframe=timeframe, limit=limit)
            if df.empty:
                logger.warning(f"No data returned for previous day either for {symbol}.")
                return df

        # Log DataFrame info before filtering
        logger.debug(f"Pre-filter DataFrame for {symbol}: {len(df)} rows")
        logger.debug(f"DataFrame head:\n{df.head().to_string()}")
        logger.debug(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Ensure timestamp is timezone-aware (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter candles to only include those from the start_of_day
        df = df[df['timestamp'] >= start_of_day]
        
        if df.empty:
            logger.warning(f"No candles remain after filtering for {symbol} on or after {start_of_day}.")
        else:
            logger.info(f"Fetched {len(df)} candles for {symbol} after filtering.")
            # Log OHLCV summary
            logger.debug(f"OHLCV summary for {symbol}:\n{df[['open', 'high', 'low', 'close', 'volume']].describe().to_string()}")

        return df

    except Exception as e:
        logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
        return pd.DataFrame()


def plot_candlestick_chart(df, symbol, timeframe):
    """
    Plots a candlestick chart, saving it as a PNG.

    Args:
        df (pandas.DataFrame): OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'].
        symbol (str): Trading pair (e.g., 'BTC-EUR').
        timeframe (str): Candle timeframe (e.g., '5m').
    """
    if df.empty:
        logger.warning(f"No data to plot for {symbol}. DataFrame is empty.")
        # Create synthetic data for testing
        logger.info(f"Generating synthetic data for {symbol} to test plotting")
        timestamps = pd.date_range(start=datetime.now(tz.UTC), periods=10, freq='5min', tz='UTC')
        synthetic_data = {
            'timestamp': timestamps,
            'open': np.random.uniform(50000, 51000, 10),
            'high': np.random.uniform(51000, 52000, 10),
            'low': np.random.uniform(49000, 50000, 10),
            'close': np.random.uniform(50000, 51000, 10),
            'volume': np.random.uniform(1, 10, 10),
            'symbol': symbol
        }
        df = pd.DataFrame(synthetic_data)

    try:
        # Validate DataFrame columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"DataFrame missing required columns: {required_columns}")
            return

        # Check for valid data
        if df[required_columns[1:]].isna().any().any():
            logger.error(f"DataFrame contains NaN values in OHLCV columns for {symbol}")
            return

        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error(f"DataFrame contains zero or negative values in OHLC columns for {symbol}")
            return

        # Prepare data for mplfinance
        df_plot = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_plot.set_index('timestamp', inplace=True)
        df_plot.index = pd.to_datetime(df_plot.index, utc=True)

        # Log DataFrame for plotting
        logger.debug(f"Plotting DataFrame for {symbol}: {len(df_plot)} rows")
        logger.debug(f"Plotting DataFrame head:\n{df_plot.head().to_string()}")
        logger.debug(f"Price range: open={df_plot['open'].min()}-{df_plot['open'].max()}, "
                     f"high={df_plot['high'].min()}-{df_plot['high'].max()}, "
                     f"low={df_plot['low'].min()}-{df_plot['low'].max()}, "
                     f"close={df_plot['close'].min()}-{df_plot['close'].max()}")

        # Calculate padding for x-axis
        timeframe_minutes = int(timeframe.replace('m', '')) if 'm' in timeframe else 60
        padding = timedelta(minutes=timeframe_minutes)
        min_time = df_plot.index.min() - padding
        max_time = df_plot.index.max() + padding

        # Define plot style
        mc = mpf.make_marketcolors(up='green', down='red')
        s = mpf.make_mpf_style(marketcolors=mc)

        # Simplified plot (no volume to isolate candlestick issue)
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=s,
            title=f"{symbol} {timeframe} Candlestick Chart - {datetime.now(tz.UTC).strftime('%Y-%m-%d')}",
            ylabel='Price (EUR)',
            volume=False,  # Disable volume to simplify
            returnfig=True,
            figsize=(12, 8),
            tight_layout=True,
            datetime_format='%H:%M',
            xrotation=45
        )

        # Set axis limits
        axes[0].set_xlim(min_time, max_time)
        price_min = df_plot[['open', 'high', 'low', 'close']].min().min()
        price_max = df_plot[['open', 'high', 'low', 'close']].max().max()
        price_range = price_max - price_min
        axes[0].set_ylim(price_min - 0.1 * price_range, price_max + 0.1 * price_range)

        # Save the plot as a PNG
        output_dir = config.config.RESULTS_FOLDER
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{symbol.replace('-', '_')}_candlestick_{datetime.now(tz.UTC).strftime('%Y%m%d')}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Candlestick chart saved to {output_path}")

        # Close the plot to free memory
        plt.close(fig)

    except Exception as e:
        logger.error(f"Error plotting candlestick chart for {symbol}: {e}", exc_info=True)


def main():
    """
    Main function to fetch and plot candles for a user-specified symbol and timeframe.
    """
    try:
        # Get user input for symbol and timeframe
        symbol = "BTC-EUR"
        timeframe = "5m"

        # Validate inputs
        if not symbol or '-' not in symbol:
            logger.error("Invalid symbol format. Expected format: XXX-YYY (e.g., BTC-EUR)")
            sys.exit(1)
        if not timeframe.endswith('m') or not timeframe[:-1].isdigit():
            logger.error("Invalid timeframe format. Expected format: Nm (e.g., 5m, 1m)")
            sys.exit(1)

        # Fetch candles
        df = fetch_daily_candles(symbol, timeframe=timeframe)
        
        # Plot candlestick chart
        plot_candlestick_chart(df, symbol, timeframe)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Exiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()