# trading_bot/src/data_processor.py
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .bitvavo_order_metrics import calculate_order_book_metrics
from . import config
from .config import logger
from .exchange import check_rate_limit, semaphore
from .state import low_volatility_assets, portfolio, portfolio_lock

# Define ANSI color codes
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BRIGHT_BLUE = "\033[94m"
RESET = "\033[0m"

def colorize_value(value, column):
    try:
        if column == "slippage_buy":
            if abs(value) < config.config.MAX_SLIPPAGE_BUY:
                return f"{GREEN}{value:>7.3f}%{RESET}"
            else:
                return f"{RED}{value:>7.3f}%{RESET}"
        elif column == "slippage_sell":
            if value > config.config.MAX_SLIPPAGE_SELL:
                return f"{GREEN}{value:>7.3f}%{RESET}"
            else:
                return f"{RED}{value:>7.3f}%{RESET}"
        elif column == "percent_change":
            if value > config.config.PRICE_INCREASE_THRESHOLD:
                return f"{GREEN}{value:>6.3f}%{RESET}"
            else:
                return f"{RED}{value:>6.3f}%{RESET}"
        elif column == "volume_eur":
            if value > config.config.MIN_VOLUME_EUR:
                return f"{GREEN}{value:>10.2f}€{RESET}"
            else:
                return f"{RED}{value:>10.2f}€{RESET}"
        elif column == "recommendation":
            if value == "Strong Buy":
                return f"{GREEN}{value:<10}{RESET}"
            elif value == "No Buy":
                return f"{RED}{value:<10}{RESET}"
            return f"{value:<10}"
        elif column == "symbol":
            return f"{YELLOW}{value:<12}{RESET}"
        return value
    except (TypeError, ValueError) as e:
        logger.error(f"Error formatting value {value} for column {column}: {e}", exc_info=True)
        return str(value)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying {retry_state.fn.__name__} after {retry_state.attempt_number} attempts"
    ),
    reraise=True
)
def calculate_order_book_metrics_with_retry(market, amount_quote, price_range_percent):
    return calculate_order_book_metrics(
        market=market,
        amount_quote=amount_quote,
        price_range_percent=price_range_percent
    )

def preprocess_market_data(api_data):
    """Preprocesses API data to match expected column names and symbol format."""
    # Convert to DataFrame if input is a list of dictionaries
    if isinstance(api_data, list):
        logger.warning("api_data is a list of dictionaries; converting to DataFrame")
        api_data = pd.DataFrame(api_data)
    
    if not isinstance(api_data, pd.DataFrame):
        logger.error(f"api_data must be a DataFrame or list of dictionaries, got {type(api_data)}")
        raise TypeError("api_data must be a DataFrame or list of dictionaries")
    
    if api_data.empty:
        logger.warning("api_data is an empty DataFrame")
        return api_data
    
    logger.debug(f"API data columns: {api_data.columns.tolist()}, dtypes: {api_data.dtypes.to_dict()}")
    
    # Check for duplicate columns
    if api_data.columns.duplicated().any():
        logger.warning(f"Duplicate columns found: {api_data.columns[api_data.columns.duplicated()].tolist()}")
        api_data = api_data.loc[:, ~api_data.columns.duplicated()]
    
    # Rename columns to match TradingStrategy expectations
    column_mapping = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close_price",
        "Date": "timestamp",
        "Asset_name": "symbol",
        "open_price": "open",
        "close_price": "close_price",
        "latest_timestamp": "timestamp",
        "volume_eur": "volume_eur"
    }
    api_data = api_data.rename(columns={k: v for k, v in column_mapping.items() if k in api_data.columns})
    
    # Check for duplicate columns after renaming
    if api_data.columns.duplicated().any():
        logger.warning(f"Duplicate columns after renaming: {api_data.columns[api_data.columns.duplicated()].tolist()}")
        api_data = api_data.loc[:, ~api_data.columns.duplicated()]
    
    # Convert symbol format (e.g., ADA-USDT to ADA/EUR) if needed
    if "symbol" in api_data.columns and api_data["symbol"].str.contains("-USDT").any():
        api_data["symbol"] = api_data["symbol"].str.replace("-USDT", "/EUR")
    
    # Ensure numeric columns
    numeric_columns = ["open", "high", "low", "close_price", "volume", "volume_eur"]
    for col in numeric_columns:
        if col in api_data.columns:
            try:
                # Ensure api_data[col] is a Series
                if isinstance(api_data[col], pd.DataFrame):
                    logger.error(f"Column {col} is a DataFrame with columns: {api_data[col].columns.tolist()}")
                    api_data[col] = api_data[col].iloc[:, 0]
                    logger.warning(f"Selected first column for {col} to resolve DataFrame issue")
                # Debug sample values
                sample_values = api_data[col].head().tolist()
                logger.debug(f"Sample values for {col}: {sample_values}")
                # Handle lists in the column
                if api_data[col].apply(lambda x: isinstance(x, (list, tuple))).any():
                    logger.warning(f"Lists found in {col}: {api_data[col].head().tolist()}")
                    api_data[col] = api_data[col].apply(
                        lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else pd.NA
                    )
                # Check for non-scalar values
                non_scalar_mask = api_data[col].apply(lambda x: isinstance(x, (list, dict)))
                if non_scalar_mask.any():
                    logger.warning(f"Non-scalar values in {col}: {api_data[non_scalar_mask]['symbol'].tolist()}")
                    api_data[col] = api_data[col].apply(lambda x: pd.NA if isinstance(x, (list, dict)) else x)
                # Convert to numeric
                api_data[col] = pd.to_numeric(api_data[col].astype(str).str.strip(), errors="coerce")
                if api_data[col].isna().any():
                    logger.warning(f"Non-numeric values converted to NaN in {col} for symbols: {api_data[api_data[col].isna()]['symbol'].tolist()}")
            except Exception as e:
                sample_values = api_data[col].head().to_numpy().tolist() if isinstance(api_data[col], pd.DataFrame) else api_data[col].head().tolist()
                logger.error(f"Failed to process column {col}: {e}. Problematic values: {sample_values}", exc_info=True)
                continue
    
    # Calculate volume_eur if not present
    if "volume_eur" not in api_data.columns and "volume" in api_data.columns and "close_price" in api_data.columns:
        api_data["volume_eur"] = api_data["volume"] * api_data["close_price"]
    
    # Warn if high/low are missing
    if "high" not in api_data.columns or "low" not in api_data.columns:
        logger.warning("Missing 'high' or 'low' columns in API data. ATR-based features will use defaults.")
    
    # Drop rows with missing critical columns
    api_data = api_data.dropna(subset=["symbol", "close_price"])
    
    # Log volume for SEI/EUR
    if "symbol" in api_data.columns and "volume_eur" in api_data.columns:
        sei_volume = api_data[api_data["symbol"] == "SEI/EUR"]["volume_eur"].sum() if "SEI/EUR" in api_data["symbol"].values else 0
        logger.debug(f"SEI/EUR volume_eur after preprocessing: {sei_volume:.2f} EUR")
    
    logger.debug(f"Preprocessed market_data columns: {api_data.columns.tolist()}, dtypes: {api_data.dtypes.to_dict()}")
    return api_data


def verify_and_analyze_data(df, price_monitor_manager):
    """
    Verifies and analyzes OHLCV data to identify assets with significant price increases and volume.

    Args:
        df (pd.DataFrame): Input market data with OHLCV columns.
        price_monitor_manager: Price monitor manager instance.

    Returns:
        tuple: (above_threshold_data, percent_changes, order_book_metrics_list)
            - above_threshold_data: DataFrame of coins meeting criteria.
            - percent_changes: DataFrame of price changes.
            - order_book_metrics_list: List of order book metrics.
    """
    try:
        if df.empty:
            logger.warning("Input DataFrame is empty.")
            return pd.DataFrame(), pd.DataFrame(), []

        # Preprocess to ensure correct column names
        df = preprocess_market_data(df)

        required_columns = {"timestamp", "open", "high", "low", "close_price", "volume", "symbol"}
        if not required_columns.issubset(df.columns):
            logger.error(f"DataFrame missing required columns: {required_columns - set(df.columns)}")
            return pd.DataFrame(), pd.DataFrame(), []

        current_time = datetime.utcnow()
        ten_min_ago = current_time - timedelta(minutes=10)
        five_min_ago = current_time - timedelta(minutes=5)

        if df["timestamp"].max() < ten_min_ago:
            logger.warning("Data contains no candles from within the last 10 minutes.")
            return pd.DataFrame(), pd.DataFrame(), []

        recent_data = df[df["timestamp"] >= five_min_ago]
        if recent_data.empty:
            logger.warning("No recent data within the last 5 minutes.")
            return pd.DataFrame(), pd.DataFrame(), []

        try:
            grouped = recent_data.groupby("symbol")
            symbols = grouped.first()
            latest = grouped.last()
            # Ensure volume is numeric
            latest["volume"] = pd.to_numeric(latest["volume"], errors="coerce")
            logger.debug(f"Sample volume values in latest: {latest['volume'].head().values.tolist()}")
            if latest["volume"].isna().any():
                logger.warning(f"Non-numeric volume values in latest data for symbols: {latest[latest['volume'].isna()].index.tolist()}")
            percent_changes = pd.DataFrame({
                "symbol": latest.index,
                "percent_change": (
                    (latest["close_price"] - symbols["open"]) / symbols["open"] * 100
                ).where(symbols["open"] > 0),
                "open_price": symbols["open"],
                "close_price": latest["close_price"],
                "high": latest["high"],
                "low": latest["low"],
                "volume": latest["volume"],
                "timestamp": latest["timestamp"],
                "volume_eur": (latest["volume"] * latest["close_price"]).where(latest["volume"] > 0)
            }).dropna()
        except (pd.errors.InvalidIndexError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error computing percent changes: {e}", exc_info=True)
            return pd.DataFrame(), pd.DataFrame(), []

        above_threshold = percent_changes[
            (percent_changes["percent_change"] >= config.config.PRICE_INCREASE_THRESHOLD)
            & (percent_changes["volume_eur"] >= config.config.MIN_VOLUME_EUR)
        ]

        # Log SEI/EUR volume
        if "SEI/EUR" in above_threshold["symbol"].values:
            sei_volume = above_threshold[above_threshold["symbol"] == "SEI/EUR"]["volume_eur"].iloc[0]
            logger.debug(f"SEI/EUR volume_eur in above_threshold: {sei_volume:.2f} EUR")

        order_book_metrics_list = []

        # Calculate order book metrics for coins above threshold
        for _, row in above_threshold.iterrows():
            symbol = row["symbol"]
            try:
                if not semaphore.acquire(timeout=5):
                    logger.error(f"Timeout acquiring semaphore for {symbol}")
                    send_alert("Semaphore Failure", f"Failed to acquire semaphore for {symbol}")
                    continue
                try:
                    check_rate_limit(1)
                    metrics = calculate_order_book_metrics_with_retry(
                        market=symbol.replace("/", "-"),
                        amount_quote=portfolio['cash'] * config.config.ALLOCATION_PER_TRADE,
                        price_range_percent=config.config.PRICE_RANGE_PERCENT,
                    )
                    metrics["bought"] = False
                    order_book_metrics_list.append(metrics)
                finally:
                    semaphore.release()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logger.error(f"Network error calculating order book metrics for {symbol}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Unexpected error calculating order book metrics for {symbol}: {e}", exc_info=True)

        if not above_threshold.empty:
            logger.info(
                f"{BRIGHT_BLUE}DATA GATHERING: Coins with price increase >= {config.config.PRICE_INCREASE_THRESHOLD}% "
                f"and volume >= €{config.config.MIN_VOLUME_EUR}:   "
                f"(only Buy if Total Score > {config.config.MIN_TOTAL_SCORE} and Slippage Buy < {config.config.MAX_SLIPPAGE_BUY}%){RESET}"
            )
            for _, row in above_threshold.iterrows():
                try:
                    symbol = row["symbol"]
                    metrics = next((m for m in order_book_metrics_list if m["market"] == symbol.replace("/", "-")), {})
                    if not metrics:
                        logger.warning(f"No order book metrics for {symbol}")
                        continue
                    logger.info(
                        f"Symbol: {colorize_value(row['symbol'], 'symbol')}  "
                        f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                        f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                        f"Open: {row['open_price']:>15.8f}  "
                        f"Close: {row['close_price']:>15.8f}  "
                        f"High: {row['high']:>15.8f}  "
                        f"Low: {row['low']:>15.8f}  "
                        f"Slippage Buy: {colorize_value(metrics.get('slippage_buy', 0), 'slippage_buy')}  "
                        f"Slippage Sell: {colorize_value(metrics.get('slippage_sell', 0), 'slippage_sell')}  "
                        f"Total Score: {metrics.get('total_score', 0):>4.2f}  "
                        f"Recommendation: {colorize_value(metrics.get('recommendation', 'N/A'), 'recommendation')}  "
                        f"Latest Timestamp: {row['timestamp']}"
                    )
                except Exception as e:
                    logger.error(f"Error logging data for {row['symbol']}: {e}", exc_info=True)

        else:
            logger.info(
                f"{BRIGHT_BLUE}DATA GATHERING: No coins with price increase >= {config.config.PRICE_INCREASE_THRESHOLD}% "
                f"and trade volume >= {config.config.MIN_VOLUME_EUR} €   "
                f"(only Buy if Total Score > {config.config.MIN_TOTAL_SCORE} and Slippage Buy < {config.config.MAX_SLIPPAGE_BUY}%){RESET}"
            )

        below_threshold = percent_changes[
            percent_changes["percent_change"] < config.config.PRICE_INCREASE_THRESHOLD
        ]
        if not below_threshold.empty:
            try:
                top_5_below = below_threshold.sort_values(
                    by="percent_change", ascending=False
                ).head(5)
                logger.info(
                    f"{BRIGHT_BLUE}DATA GATHERING: Top 5 coins with price increase < {config.config.PRICE_INCREASE_THRESHOLD}% "
                    f"or trade volume < {config.config.MIN_VOLUME_EUR} €:{RESET}"
                )
                for _, row in top_5_below.iterrows():
                    symbol = row["symbol"]
                    try:
                        if not semaphore.acquire(timeout=5):
                            logger.error(f"Timeout acquiring semaphore for {symbol}")
                            send_alert("Semaphore Failure", f"Failed to acquire semaphore for {symbol}")
                            continue
                        try:
                            check_rate_limit(1)
                            metrics = calculate_order_book_metrics_with_retry(
                                market=symbol.replace("/", "-"),
                                amount_quote=portfolio['cash'] * config.config.ALLOCATION_PER_TRADE,
                                price_range_percent=config.config.PRICE_RANGE_PERCENT,
                            )
                            metrics["bought"] = False
                            order_book_metrics_list.append(metrics)
                        finally:
                            semaphore.release()
                        logger.info(
                            f"Symbol: {colorize_value(row['symbol'], 'symbol')}  "
                            f"Change: {colorize_value(row['percent_change'], 'percent_change')}  "
                            f"Volume: {colorize_value(row['volume_eur'], 'volume_eur')}  "
                            f"Open: {row['open_price']:>15.8f}  "
                            f"Close: {row['close_price']:>15.8f}  "
                            f"High: {row['high']:>15.8f}  "
                            f"Low: {row['low']:>15.8f}  "
                            f"Slippage Buy: {colorize_value(metrics.get('slippage_buy', 0), 'slippage_buy')}  "
                            f"Slippage Sell: {colorize_value(metrics.get('slippage_sell', 0), 'slippage_sell')}  "
                            f"Total Score: {metrics.get('total_score', 0):>4.2f}  "
                            f"Recommendation: {colorize_value(metrics.get('recommendation', 'N/A'), 'recommendation')}  "
                            f"Latest Timestamp: {row['timestamp']}"
                        )
                    except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                        logger.error(f"Network error calculating order book metrics for {symbol}: {e}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Unexpected error processing {symbol}: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"Error processing below-threshold coins: {e}", exc_info=True)
        else:
            logger.info(f"No coins with price increase < {config.config.PRICE_INCREASE_THRESHOLD}%")

        try:
            if not portfolio_lock.acquire(timeout=5):
                logger.error("Timeout acquiring portfolio lock")
                send_alert("Portfolio Lock Failure", "Failed to acquire portfolio lock")
                return above_threshold, percent_changes, order_book_metrics_list
            try:
                for symbol in list(low_volatility_assets):
                    if symbol in portfolio["assets"] and symbol in percent_changes.index:
                        try:
                            recent_change = percent_changes.loc[
                                percent_changes["symbol"] == symbol, "percent_change"
                            ].iloc[0]
                            if abs(recent_change) >= config.config.PRICE_INCREASE_THRESHOLD / 2:
                                logger.info(
                                    f"{symbol} regained volatility (change: {recent_change:.2f}%). Resuming monitoring."
                                )
                                low_volatility_assets.discard(symbol)
                                price_monitor_manager.start(symbol, portfolio, portfolio_lock, df)
                        except (KeyError, IndexError) as e:
                            logger.error(f"Error checking volatility for {symbol}: {e}", exc_info=True)
            finally:
                portfolio_lock.release()
        except Exception as e:
            logger.error(f"Error managing low volatility assets: {e}", exc_info=True)

        return above_threshold, percent_changes, order_book_metrics_list

    except Exception as e:
        logger.error(f"Critical error in verify_and_analyze_data: {e}", exc_info=True)
        send_alert("Data Processing Failure", f"Critical error in data processing: {e}")
        return pd.DataFrame(), pd.DataFrame(), []

def send_alert(subject, message):
    logger.error(f"ALERT: {subject} - {message}")