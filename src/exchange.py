import json
import logging
import re
import time
from datetime import datetime
from threading import Lock, Semaphore

import ccxt
import pandas as pd
from ccxt.base.errors import PermissionDenied
from tenacity import retry, stop_after_attempt, wait_fixed

from .config import (API_KEY, API_SECRET, CANDLE_LIMIT, CANDLE_TIMEFRAME,
                     CONCURRENT_REQUESTS, RATE_LIMIT_WEIGHT, logger)
from .state import (ban_expiry_time, is_banned, last_reset_time,
                    rate_limit_lock, weight_used)

# Initialize rate limit tracking
semaphore = Semaphore(CONCURRENT_REQUESTS)

# Initialize Bitvavo client
bitvavo = ccxt.bitvavo(
    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    }
)


def handle_ban_error(exception):
    """
    Checks if an exception indicates an IP/API key ban and returns the ban expiration time.

    Args:
        exception: The exception caught during an API call.

    Returns:
        float or None: Unix timestamp (seconds) when the ban expires, or None if not a ban error.
    """
    global is_banned, ban_expiry_time
    try:
        error_message = str(exception)
        if "403" in error_message and 'errorCode":105' in error_message:
            match = re.search(r"The ban expires at (\d+)", error_message)
            if match:
                ban_expiry_ms = int(match.group(1))
                ban_expiry = ban_expiry_ms / 1000  # Convert to seconds
                expiry_datetime = datetime.utcfromtimestamp(ban_expiry).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )
                logger.warning(f"API ban detected. Ban expires at {expiry_datetime}")
                is_banned = True
                ban_expiry_time = ban_expiry
                return ban_expiry
        return None
    except Exception as e:
        logger.error(f"Error parsing ban response: {e}")
        return None


def wait_until_ban_lifted(ban_expiry):
    """
    Waits until the ban expiration time is reached.

    Args:
        ban_expiry (float): Unix timestamp (seconds) when the ban expires.
    """
    global is_banned
    current_time = time.time()
    wait_time = max(0, ban_expiry - current_time)
    if wait_time > 0:
        logger.info(
            f"Waiting {wait_time:.2f} seconds until ban lifts at {datetime.utcfromtimestamp(ban_expiry)}"
        )
        time.sleep(wait_time)
    is_banned = False
    logger.info("Ban has been lifted")


def check_rate_limit(request_weight):
    """
    Manages Bitvavo API rate limits by tracking request weight and sleeping if necessary.
    Updates global weight_used and last_reset_time.

    Args:
        request_weight (int): Weight of the specific API request.
    """
    global weight_used, last_reset_time, CONCURRENT_REQUESTS
    with rate_limit_lock:
        current_time = time.time()
        if current_time - last_reset_time >= 60:
            logger.info(f"Resetting rate limit: weight_used={weight_used} -> 0")
            weight_used = 0
            last_reset_time = current_time
        logger.debug(f"Checking rate limit: weight_used={weight_used}, request_weight={request_weight}, total={weight_used + request_weight}, limit={RATE_LIMIT_WEIGHT}")
        if (weight_used + request_weight) > RATE_LIMIT_WEIGHT * 0.5:
            sleep_time = 60 - (current_time - last_reset_time)
            if sleep_time > 0:
                logger.info(f"Approaching rate limit, sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                weight_used = 0
                last_reset_time = current_time
                new_concurrency = max(1, CONCURRENT_REQUESTS - 1)
                CONCURRENT_REQUESTS = new_concurrency
                from .price_monitor import PriceMonitorManager
                PriceMonitorManager().adjust_concurrency(new_concurrency)
                logger.info(f"Adjusted CONCURRENT_REQUESTS to {CONCURRENT_REQUESTS}")
        weight_used += request_weight
        logger.debug(f"Updated weight_used to {weight_used}")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_ticker_price(symbol):
    """
    Fetches the latest ticker price for a given symbol from Bitvavo.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/EUR').

    Returns:
        float: Latest price, or None if fetch fails or price is invalid.
    """
    with semaphore:
        acquired = semaphore.acquire(timeout=5)  # Add timeout
        if not acquired:
            logger.warning(f"Failed to acquire semaphore for {symbol}. Skipping fetch.")
            return None

        try:
            check_rate_limit(1)  # Weight for fetch_ticker is typically 1
            ticker = bitvavo.fetch_ticker(symbol)
            if not isinstance(ticker, dict) or "last" not in ticker:
                logger.error(f"Invalid ticker response for {symbol}: {ticker}")
                return None
            price = float(ticker["last"])
            if price <= 0:
                logger.error(
                    f"Invalid price {price} for {symbol}. Price must be positive."
                )
                return None
            logger.debug(f"Fetched ticker price {price:.8f} for {symbol}")
            return price
        except PermissionDenied as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return None
            logger.error(f"Permission denied for {symbol}: {e}")
            return None
        except ccxt.NetworkError as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return None
            logger.error(f"Network error fetching ticker price for {symbol}: {e}")
            return None
        except Exception as e:  # Catch all exceptions
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                raise  # Retry after ban lifts
            logger.error(
                f"Failed to fetch ticker price for {symbol}: {e}", exc_info=True
            )
            # Fallback to last known price if available
            from .price_monitor import PriceMonitorManager

            last_price = PriceMonitorManager().last_prices.get(symbol)
            if last_price:
                logger.info(f"Using last known price {last_price:.8f} for {symbol}")
                return last_price
            return None
        finally:
            if acquired:
                semaphore.release()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_klines(symbol, timeframe=CANDLE_TIMEFRAME, limit=CANDLE_LIMIT):
    """
    Fetches OHLCV data for a given symbol from Bitvavo.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/EUR').
        timeframe (str): Candle timeframe (default: '1m').
        limit (int): Number of candles to fetch (default: 10).

    Returns:
        pandas.DataFrame: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'].
    """
    global is_banned, ban_expiry_time
    if is_banned and time.time() < ban_expiry_time:
        logger.warning(
            f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Skipping fetch for {symbol}."
        )
        return pd.DataFrame()  # Return empty DataFrame to continue execution
    with semaphore:
        check_rate_limit(1)  # Weight for fetch_ohlcv is typically 1
        try:
            start_time = time.time()
            klines = bitvavo.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(
                klines, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol"] = symbol
            # Validate prices
            if (df[["open", "high", "low", "close"]] <= 0).any().any():
                logger.error(
                    f"Invalid prices (zero or negative) in OHLCV data for {symbol}. Discarding data."
                )
                return pd.DataFrame()
            logger.debug(
                f"Fetched {len(df)} candles for {symbol} in {time.time() - start_time:.2f} seconds"
            )
            return df
        except PermissionDenied as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return pd.DataFrame()  # Return empty DataFrame instead of raising
            logger.error(f"Permission denied for {symbol}: {e}")
            return pd.DataFrame()
        except ccxt.NetworkError as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return pd.DataFrame()  # Return empty DataFrame instead of raising
            logger.error(f"Network error fetching data for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(
                f"Unexpected error fetching data for {symbol}: {e}", exc_info=True
            )
            return pd.DataFrame()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def fetch_trade_details(symbol, start_time, end_time):
    """
    Fetches trade details for a symbol within a time range from Bitvavo.

    Args:
        symbol (str): Trading pair (e.g., 'BTC/EUR').
        start_time (datetime): Start of the time range.
        end_time (datetime): End of the time range.

    Returns:
        tuple: (trade_count, largest_trade_volume_eur)
    """
    global is_banned, ban_expiry_time
    if is_banned and time.time() < ban_expiry_time:
        logger.warning(
            f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Skipping fetch for {symbol}."
        )
        return 0, 0.0
    with semaphore:
        check_rate_limit(1)
        try:
            if not isinstance(start_time, datetime) or not isinstance(
                end_time, datetime
            ):
                raise ValueError(
                    f"Invalid timestamp types: start_time={type(start_time)}, end_time={type(end_time)}"
                )
            if end_time <= start_time:
                raise ValueError(
                    f"end_time ({end_time}) must be after start_time ({start_time})"
                )
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            trades = bitvavo.fetch_trades(
                symbol,
                since=start_ms,
                limit=1000,
                params={"start": start_ms, "end": end_ms},
            )
            if not trades:
                logger.warning(
                    f"No trades returned for {symbol} from {start_time} to {end_time}"
                )
                return 0, 0.0
            valid_trades = [
                trade for trade in trades if start_ms <= trade["timestamp"] <= end_ms
            ]
            trade_count = len(valid_trades)
            largest_trade_volume_eur = max(
                (trade["amount"] * trade["price"] for trade in trades), default=0.0
            )
            logger.debug(
                f"Fetched {trade_count} trades for {symbol}, Largest trade volume: €{largest_trade_volume_eur:.2f}"
            )
            return trade_count, largest_trade_volume_eur
        except PermissionDenied as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return 0, 0.0
            logger.error(f"Permission denied for {symbol}: {e}")
            return 0, 0.0
        except ccxt.NetworkError as e:
            ban_expiry = handle_ban_error(e)
            if ban_expiry:
                wait_until_ban_lifted(ban_expiry)
                return 0, 0.0
            logger.error(f"Network error fetching trade details for {symbol}: {e}")
            return 0, 0.0
        except Exception as e:
            logger.error(
                f"Unexpected error fetching trade details for {symbol}: {e}",
                exc_info=True,
            )
            return 0, 0.0
