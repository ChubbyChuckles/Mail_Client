# trading_bot/src/main.py
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import ccxt
import pandas as pd
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from . import config
from .config import logger, IS_GITHUB_ACTIONS
from .clean_up import garbage_collection
from .data_processor import colorize_value, verify_and_analyze_data
from .exchange import bitvavo, check_rate_limit, fetch_klines
from .portfolio import manage_portfolio, save_portfolio
from .price_monitor import PriceMonitorManager
from .print_assets import print_portfolio
from .print_trade_variables import print_trade_variables
from .state import (ban_expiry_time, is_banned, low_volatility_assets,
                    portfolio, portfolio_lock)
from .storage import save_to_local
import asyncio
from .telegram_notifications import TelegramNotifier

# Initialize Telegram notifier
telegram_notifier = TelegramNotifier(
    bot_token=config.config.TELEGRAM_BOT_TOKEN, chat_id=config.config.TELEGRAM_CHAT_ID
)
asyncio.run_coroutine_threadsafe(telegram_notifier.start(), asyncio.get_event_loop())

last_cycle_time = time.time()
GREEN = "\033[32m"
BRIGHT_BLUE = "\033[94m"
YELLOW = "\033[33m"
RESET = "\033[0m"

# Define runtime limit for GitHub Actions (4 hours and 3 minutes = 14580 seconds)
RUNTIME_LIMIT_SECONDS = 14580 if IS_GITHUB_ACTIONS else float("inf")

def watchdog(price_monitor_manager):
    global last_cycle_time
    while True:
        try:
            current_time = time.time()
            if current_time - last_cycle_time > config.config.LOOP_INTERVAL_SECONDS * 2:
                logger.error("Main loop appears to be hung. Checking ban status...")
                if is_banned and current_time < ban_expiry_time:
                    logger.info(
                        f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Stopping monitors and waiting..."
                    )
                    try:
                        price_monitor_manager.stop_all()
                    except Exception as e:
                        logger.error(
                            f"Error stopping monitors in watchdog: {e}", exc_info=True
                        )
                    time.sleep(min(ban_expiry_time - current_time, 60))
                else:
                    logger.error("Main loop hung without ban. Attempting recovery...")
                    telegram_notifier.notify_error(
                        "Main Loop Hung", "Main loop appears to be hung without API ban."
                    )
            time.sleep(10)
        except Exception as e:
            logger.error(f"Watchdog error: {e}", exc_info=True)
            telegram_notifier.notify_error("Watchdog Error", f"Watchdog failed: {e}")
            time.sleep(10)

def center_text(text, total_width=256):
    try:
        clean_text = text.replace(GREEN, "").replace(RESET, "").replace(BRIGHT_BLUE, "").replace(YELLOW, "")
        text_length = len(clean_text)
        padding = (total_width - text_length) // 2
        left_padding = " " * padding
        right_padding = " " * (total_width - text_length - padding)
        return f"{left_padding}{text}{right_padding}"
    except TypeError as e:
        logger.error(f"Error in center_text: {e}", exc_info=True)
        return text

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying {retry_state.fn.__name__} after {retry_state.attempt_number} attempts"
    ),
    reraise=True,
)
def load_markets_with_retry():
    return bitvavo.load_markets()

def main():
    price_monitor_manager = PriceMonitorManager()
    start_time = time.time()
    try:
        watchdog_thread = threading.Thread(
            target=watchdog, args=(price_monitor_manager,), daemon=True
        )
        watchdog_thread.start()
    except Exception as e:
        logger.error(f"Error starting watchdog thread: {e}", exc_info=True)
        telegram_notifier.notify_error("Watchdog Failure", f"Failed to start watchdog: {e}")

    try:
        logger.info(f"{GREEN}{'=' * 256}{RESET}")
        logger.info(center_text(f"{GREEN}CringeTrader 1.0.4{RESET}", total_width=256))
        logger.info(f"{GREEN}{'=' * 256}{RESET}")

        try:
            print_trade_variables(vars_per_line=7, total_line_width=256)
        except Exception as e:
            logger.error(f"Error printing trade variables: {e}", exc_info=True)

        try:
            garbage_collection()
        except Exception as e:
            logger.error(f"Error in garbage collection: {e}", exc_info=True)

        try:
            markets = load_markets_with_retry()
            eur_pairs = [
                symbol
                for symbol in markets
                if symbol.endswith("/EUR") and markets[symbol].get("active", False)
            ]
            logger.info(
                f"Loaded {BRIGHT_BLUE}{len(markets)}{RESET} markets, "
                f"{BRIGHT_BLUE}{len(eur_pairs)}{RESET} active EUR pairs"
            )
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            logger.error(f"Error loading markets: {e}", exc_info=True)
            eur_pairs = []
            telegram_notifier.notify_error("Market Load Failure", f"Failed to load markets: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading markets: {e}", exc_info=True)
            eur_pairs = []
            telegram_notifier.notify_error("Market Load Failure", f"Unexpected error loading markets: {e}")

        while True:
            # Check runtime limit for GitHub Actions
            if IS_GITHUB_ACTIONS and time.time() - start_time >= RUNTIME_LIMIT_SECONDS:
                logger.info("Reached 4-hour 3-minute runtime limit in GitHub Actions. Initiating shutdown...")
                break

            try:
                config.reload_config()
            except Exception as e:
                logger.error(f"Error reloading configuration: {e}", exc_info=True)

            global last_cycle_time
            last_cycle_time = time.time()

            if is_banned and time.time() < ban_expiry_time:
                logger.warning(
                    f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Skipping data fetch."
                )
                time.sleep(min(ban_expiry_time - time.time(), 60))
                continue

            try:
                check_rate_limit(1)
                tickers = bitvavo.fetch_tickers(eur_pairs)
                top_volume = sorted(
                    [
                        (symbol, ticker["quoteVolume"])
                        for symbol, ticker in tickers.items()
                        if ticker.get("quoteVolume")
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:300]
                symbols = [symbol for symbol, _ in top_volume]
            except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                logger.error(f"Error fetching tickers: {e}", exc_info=True)
                symbols = eur_pairs[:300]
                telegram_notifier.notify_error("Ticker Fetch Failure", f"Failed to fetch tickers: {e}")
            except Exception as e:
                logger.error(f"Unexpected error fetching tickers: {e}", exc_info=True)
                symbols = eur_pairs[:300]
                telegram_notifier.notify_error(
                    "Ticker Fetch Failure", f"Unexpected error fetching tickers: {e}"
                )

            logger.info(f"Processing {GREEN}{len(symbols)}{RESET} EUR symbols")

            active_monitors = price_monitor_manager.active_monitors()
            adjusted_concurrency = max(
                1,
                min(
                    config.config.CONCURRENT_REQUESTS_GITHUB if IS_GITHUB_ACTIONS else config.config.CONCURRENT_REQUESTS,
                    20 - active_monitors
                )
            )
            logger.info(
                f"{YELLOW}ACTIVE MONITORS:{RESET} {active_monitors}     |     "
                f"{YELLOW}ADJUSTED CONCURRENCY:{RESET} {adjusted_concurrency}"
            )

            all_data = []
            try:
                with ThreadPoolExecutor(max_workers=adjusted_concurrency) as executor:
                    logger.debug(
                        f"Starting ThreadPoolExecutor with {adjusted_concurrency} workers"
                    )
                    futures = {
                        executor.submit(fetch_klines, symbol): symbol
                        for symbol in symbols
                    }
                    for future in futures:
                        symbol = futures[future]
                        try:
                            result = future.result(timeout=30)
                            if not result.empty:
                                all_data.append(result)
                            logger.debug(f"Received result for {symbol}")
                        except FutureTimeoutError:
                            logger.error(f"Timeout fetching klines for {symbol}")
                            telegram_notifier.notify_error(
                                "Kline Fetch Timeout", f"Timeout fetching klines for {symbol}"
                            )
                        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
                            logger.error(
                                f"Network error fetching klines for {symbol}: {e}",
                                exc_info=True,
                            )
                            telegram_notifier.notify_error(
                                "Kline Fetch Error", f"Network error for {symbol}: {e}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Unexpected error fetching klines for {symbol}: {e}",
                                exc_info=True,
                            )
                            telegram_notifier.notify_error(
                                "Kline Fetch Error", f"Unexpected error for {symbol}: {e}"
                            )
            except Exception as e:
                logger.error(f"Error in ThreadPoolExecutor: {e}", exc_info=True)
                telegram_notifier.notify_error("Executor Failure", f"ThreadPoolExecutor error: {e}")

            if all_data:
                try:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    logger.debug(
                        f"Combined DataFrame has {len(combined_df)} rows, with {combined_df['symbol'].nunique()} unique symbols"
                    )
                    for symbol in combined_df["symbol"].unique():
                        logger.debug(
                            f"Symbol {symbol} has {len(combined_df[combined_df['symbol'] == symbol])} candles"
                        )
                    output_path = (
                        f"{config.config.RESULTS_FOLDER}/"
                        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_"
                        f"{config.config.PARQUET_FILENAME}"
                    )
                    if IS_GITHUB_ACTIONS:
                        logger.info("Skip saving parquet file.")
                    else:
                        save_to_local(combined_df, output_path)
                except pd.errors.EmptyDataError as e:
                    logger.error(f"Error concatenating data: {e}", exc_info=True)
                    continue
                except Exception as e:
                    logger.error(
                        f"Unexpected error processing data: {e}", exc_info=True
                    )
                    telegram_notifier.notify_error(
                        "Data Processing Error", f"Unexpected error processing data: {e}"
                    )
                    continue

                try:
                    above_threshold_data, percent_changes, order_book_metrics_list = (
                        verify_and_analyze_data(combined_df, price_monitor_manager)
                    )
                except Exception as e:
                    logger.error(f"Error analyzing data: {e}", exc_info=True)
                    telegram_notifier.notify_error("Data Analysis Error", f"Error analyzing data: {e}")
                    continue

                try:
                    manage_portfolio(
                        above_threshold_data,
                        percent_changes,
                        price_monitor_manager,
                        order_book_metrics_list,
                        combined_df=combined_df,
                    )
                except Exception as e:
                    logger.error(f"Error managing portfolio: {e}", exc_info=True)
                    telegram_notifier.notify_error("Portfolio Management Error", f"Error managing portfolio: {e}")

                try:
                    save_portfolio()
                except Exception as e:
                    logger.error(f"Error saving portfolio: {e}", exc_info=True)
                    telegram_notifier.notify_error(
                        "Portfolio Save Failure", f"Failed to save portfolio: {e}"
                    )

                try:
                    if not portfolio_lock.acquire(timeout=5):
                        logger.error("Timeout acquiring portfolio lock")
                        telegram_notifier.notify_error(
                            "Portfolio Lock Failure", "Failed to acquire portfolio lock"
                        )
                        continue
                    try:
                        total_value = portfolio["cash"] + sum(
                            asset["quantity"] * asset["current_price"]
                            for asset in portfolio["assets"].values()
                        )
                        logger.info(
                            f"{BRIGHT_BLUE}PORTFOLIO STATUS: Cash: {portfolio['cash']:.2f} EUR, "
                            f"Assets: {len(portfolio['assets'])}, "
                            f"Total Value: {total_value:.2f} EUR{RESET}"
                        )
                    finally:
                        portfolio_lock.release()
                except Exception as e:
                    logger.error(
                        f"Error calculating portfolio value: {e}", exc_info=True
                    )
                    telegram_notifier.notify_error(
                        "Portfolio Value Error", f"Error calculating portfolio value: {e}"
                    )

                try:
                    portfolio_path = (
                        "/tmp/portfolio.json" if IS_GITHUB_ACTIONS else "portfolio.json"
                    )
                    if not os.path.exists(portfolio_path):
                        raise FileNotFoundError(
                            f"Portfolio file '{portfolio_path}' does not exist"
                        )
                    print_portfolio(portfolio_path)
                except FileNotFoundError as e:
                    logger.error(f"Portfolio file error: {e}", exc_info=True)
                    telegram_notifier.notify_error("Portfolio File Error", f"Portfolio file error: {e}")
                except Exception as e:
                    logger.error(
                        f"Unexpected error printing portfolio: {e}", exc_info=True
                    )
                    telegram_notifier.notify_error(
                        "Portfolio Print Error", f"Unexpected error printing portfolio: {e}"
                    )
            else:
                logger.warning("No data collected in this cycle.")
                telegram_notifier.notify_error(
                    "No Data Collected", "No data collected in this cycle."
                )

            try:
                if not portfolio_lock.acquire(timeout=5):
                    logger.error(
                        "Timeout acquiring portfolio lock for monitor management"
                    )
                    telegram_notifier.notify_error(
                        "Monitor Lock Failure",
                        "Failed to acquire lock for monitor management",
                    )
                    continue
                try:
                    active_assets = set(portfolio["assets"].keys())
                    for symbol in list(price_monitor_manager.running.keys()):
                        if symbol not in active_assets:
                            logger.warning(f"Stopping orphaned monitor for {symbol}")
                            try:
                                price_monitor_manager.stop(symbol)
                            except Exception as e:
                                logger.error(
                                    f"Error stopping monitor for {symbol}: {e}",
                                    exc_info=True,
                                )
                                telegram_notifier.notify_error(
                                    "Monitor Stop Error", f"Error stopping monitor for {symbol}: {e}"
                                )
                    for symbol in active_assets:
                        if (
                            symbol not in price_monitor_manager.running
                            and symbol not in low_volatility_assets
                        ):
                            try:
                                price_monitor_manager.start(
                                    symbol, portfolio, portfolio_lock, combined_df
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error starting monitor for {symbol}: {e}",
                                    exc_info=True,
                                )
                                telegram_notifier.notify_error(
                                    "Monitor Start Error", f"Error starting monitor for {symbol}: {e}"
                                )
                finally:
                    portfolio_lock.release()
            except Exception as e:
                logger.error(f"Error managing price monitors: {e}", exc_info=True)
                telegram_notifier.notify_error(
                    "Price Monitor Management Error", f"Error managing price monitors: {e}"
                )

            try:
                from .state import save_state
                save_state()
            except Exception as e:
                logger.error(f"Error saving state: {e}", exc_info=True)
                telegram_notifier.notify_error("State Save Failure", f"Failed to save state: {e}")

            elapsed_time = time.time() - last_cycle_time
            sleep_time = max(0, config.config.LOOP_INTERVAL_SECONDS - elapsed_time)
            logger.info(f"{GREEN}{'=' * 256}{RESET}")
            logger.info(
                center_text(
                    f"{GREEN}Cycle completed in {elapsed_time:.2f} seconds. "
                    f"Sleeping for {sleep_time:.2f} seconds.{RESET}",
                    total_width=256,
                )
            )
            logger.info(f"{GREEN}{'=' * 256}{RESET}")
            logger.debug(f"Active threads: {threading.active_count()}")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Initiating graceful shutdown...")
        perform_shutdown(price_monitor_manager)
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
        telegram_notifier.notify_error("Critical Failure", f"Main loop crashed: {e}")
        perform_shutdown(price_monitor_manager)
        sys.exit(1)

def perform_shutdown(price_monitor_manager):
    """Handles graceful shutdown of the bot."""
    try:
        asyncio.run_coroutine_threadsafe(
            telegram_notifier.stop(), asyncio.get_event_loop()
        )
    except Exception as e:
        logger.error(
            f"Error stopping Telegram notifier during shutdown: {e}", exc_info=True
        )
    try:
        price_monitor_manager.stop_all()
    except Exception as e:
        logger.error(
            f"Error stopping price monitors during shutdown: {e}", exc_info=True
        )
    try:
        save_portfolio()
    except Exception as e:
        logger.error(f"Error saving portfolio during shutdown: {e}", exc_info=True)
    try:
        from .state import save_state
        save_state()
    except Exception as e:
        logger.error(f"Error saving state during shutdown: {e}", exc_info=True)
    logger.info("Bot stopped successfully.")
    if IS_GITHUB_ACTIONS:
        sys.exit(0)

def send_alert(subject, message):
    telegram_notifier.notify_error(subject, message)

if __name__ == "__main__":
    main()