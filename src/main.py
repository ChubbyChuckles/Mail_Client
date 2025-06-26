# main.py
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
import os

import pandas as pd

from .config import (CONCURRENT_REQUESTS, LOOP_INTERVAL_SECONDS,
                     PARQUET_FILENAME, RESULTS_FOLDER, logger)
from .data_processor import verify_and_analyze_data, colorize_value
from .exchange import bitvavo, check_rate_limit, fetch_klines
from .portfolio import manage_portfolio, save_portfolio
from .price_monitor import PriceMonitorManager
from .state import (ban_expiry_time, is_banned, low_volatility_assets,
                    portfolio, portfolio_lock)
from .storage import save_to_local
from miscellaneous.print_assets import print_portfolio
from miscellaneous.clean_up import garbage_collection

last_cycle_time = time.time()
GREEN = "\033[32m"
RESET = "\033[0m"

def watchdog(price_monitor_manager):
    global last_cycle_time
    price_monitor_manager = (
        PriceMonitorManager()
    )  # Ensure access to PriceMonitorManager instance
    while True:
        try:
            current_time = time.time()
            if current_time - last_cycle_time > LOOP_INTERVAL_SECONDS * 2:
                logger.error("Main loop appears to be hung. Checking ban status...")
                if is_banned and current_time < ban_expiry_time:
                    logger.info(
                        f"API is banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Stopping monitors and waiting..."
                    )
                    price_monitor_manager.stop_all()  # Stop all monitoring threads
                    time.sleep(min(ban_expiry_time - current_time, 60))
                else:
                    logger.error("Main loop hung without ban. Attempting recovery...")
            time.sleep(10)
        except Exception as e:
            logger.error(f"Watchdog error: {e}", exc_info=True)

def center_text(text, total_width=256):
    # Strip ANSI color codes for length calculation
    clean_text = text.replace(GREEN, "").replace(RESET, "")
    text_length = len(clean_text)
    
    # Calculate padding needed on each side
    padding = (total_width - text_length) // 2
    left_padding = " " * padding
    right_padding = " " * (total_width - text_length - padding)
    
    # Return the padded text with color codes preserved
    return f"{left_padding}{text}{right_padding}"

def main():
    # logger.info("Starting trading bot...")
    price_monitor_manager = PriceMonitorManager()
    watchdog_thread = threading.Thread(
        target=watchdog, args=(price_monitor_manager,), daemon=True
    )
    watchdog_thread.start()
    # Define ANSI color code for bright blue
    BRIGHT_BLUE = "\033[94m"
    RESET = "\033[0m"  # Resets color to default
    YELLOW = "\033[33m"

    try:
        logger.info(
            f"{GREEN}{'=' * 256}{RESET}"
        )
        logger.info(
            center_text(
                f"{GREEN}CringeTrader 1.0.4{RESET}",
                total_width=256
            )
        )
        logger.info(
            f"{GREEN}{'=' * 256}{RESET}"
        )
        garbage_collection()
        # logger.info("Cleaning up old files...")
        markets = bitvavo.load_markets()
        eur_pairs = [
            symbol
            for symbol in markets
            if symbol.endswith("/EUR") and markets[symbol].get("active", False)
        ]
        logger.info(f"Loaded {BRIGHT_BLUE}{len(markets)}{RESET} markets, {BRIGHT_BLUE}{len(eur_pairs)}{RESET} active EUR pairs")
        

        while True:
            global last_cycle_time
            last_cycle_time = time.time()
            all_data = []
            # logger.info("Fetching new data cycle...")

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
            except Exception as e:
                logger.error(f"Error fetching tickers for volume ranking: {e}")
                symbols = eur_pairs[:300]
            logger.info(f"Processing {GREEN}{len(symbols)}{RESET} EUR symbols")

            active_monitors = price_monitor_manager.active_monitors()
            adjusted_concurrency = max(
                1, min(CONCURRENT_REQUESTS - active_monitors, 20)
            )
            logger.info(
                f"{YELLOW}ACTIVE MONITORS:{RESET} {active_monitors}     |     {YELLOW}ADJUSTED CONCURRENCY:{RESET} {adjusted_concurrency}"
            )

            

            with ThreadPoolExecutor(max_workers=adjusted_concurrency) as executor:
                logger.debug(
                    f"Starting ThreadPoolExecutor with {adjusted_concurrency} workers"
                )
                results = []
                futures = {
                    executor.submit(fetch_klines, symbol): symbol for symbol in symbols
                }
                for future in futures:
                    symbol = futures[future]
                    try:
                        result = future.result(
                            timeout=30
                        )  # 30-second timeout per fetch
                        if not result.empty:
                            all_data.append(result)
                        logger.debug(f"Received result for {symbol}")
                    except FutureTimeoutError:
                        logger.error(f"Timeout fetching klines for {symbol}")
                    except Exception as e:
                        logger.error(
                            f"Error fetching klines for {symbol}: {e}", exc_info=True
                        )
                logger.debug("ThreadPoolExecutor completed")

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                output_path = f"{RESULTS_FOLDER}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{PARQUET_FILENAME}"
                save_to_local(combined_df, output_path)
                above_threshold_data, percent_changes, order_book_metrics_list = (
                    verify_and_analyze_data(combined_df, price_monitor_manager)
                )
                manage_portfolio(
                    above_threshold_data,
                    percent_changes,
                    price_monitor_manager,
                    order_book_metrics_list,
                )
                save_portfolio()
                with portfolio_lock:
                    total_value = portfolio["cash"] + sum(
                        asset["quantity"] * asset["current_price"]
                        for asset in portfolio["assets"].values()
                    )

                    
                    logger.info(
                        f"{BRIGHT_BLUE}PORTFOLIO STATUS: Cash: {portfolio['cash']:.2f} EUR, "
                        f"Assets: {len(portfolio['assets'])}, "
                        f"Total Value: {total_value:.2f} EUR{RESET}"
                    )
                try:
                    if not os.path.exists("portfolio.json"):
                        raise FileNotFoundError("Portfolio file 'portfolio.json' does not exist")
                    print_portfolio("portfolio.json")
                except FileNotFoundError as e:
                    print(f"Error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                else:
                    logger.warning("No data collected in this cycle.")

            try:
                with portfolio_lock:
                    active_assets = set(portfolio["assets"].keys())
                    for symbol in list(price_monitor_manager.running.keys()):
                        if symbol not in active_assets:
                            logger.warning(f"Stopping orphaned monitor for {symbol}")
                            price_monitor_manager.stop(symbol)
                    for symbol in active_assets:
                        if (
                            symbol not in price_monitor_manager.running
                            and symbol not in low_volatility_assets
                        ):
                            price_monitor_manager.start(
                                symbol, portfolio, portfolio_lock, combined_df
                            )
            except Exception as e:
                logger.error(f"Error managing price monitors: {e}")

            from .state import save_state

            save_state()
            elapsed_time = time.time() - last_cycle_time
            sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed_time)
            logger.info(
                f"{GREEN}{'=' * 256}{RESET}"
            )
            logger.info(
                center_text(
                    f"{GREEN}Cycle completed in {elapsed_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.{RESET}",
                    total_width=256
                )
            )
            logger.info(
                f"{GREEN}{'=' * 256}{RESET}"
            )
            logger.debug(f"Active threads: {threading.active_count()}")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping bot...")
        price_monitor_manager.stop_all()
        save_portfolio()
        logger.info("Bot stopped successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
