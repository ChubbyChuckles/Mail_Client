# main.py
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

import pandas as pd

from .config import (CONCURRENT_REQUESTS, LOOP_INTERVAL_SECONDS,
                     PARQUET_FILENAME, RESULTS_FOLDER, logger)
from .data_processor import verify_and_analyze_data
from .exchange import bitvavo, check_rate_limit, fetch_klines
from .notifications import run_async, send_telegram_message
from .portfolio import manage_portfolio, save_portfolio
from .price_monitor import PriceMonitorManager
from .state import (ban_expiry_time, is_banned, low_volatility_assets,
                    portfolio, portfolio_lock)
from .storage import save_to_local

last_cycle_time = time.time()


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
                    run_async(
                        send_telegram_message(
                            f"API banned until {datetime.utcfromtimestamp(ban_expiry_time)}. Waiting."
                        )
                    )
                    time.sleep(min(ban_expiry_time - current_time, 60))
                else:
                    logger.error("Main loop hung without ban. Attempting recovery...")
                    run_async(
                        send_telegram_message("Main loop hung. Attempting recovery.")
                    )
            time.sleep(10)
        except Exception as e:
            logger.error(f"Watchdog error: {e}", exc_info=True)


def main():
    logger.info("Starting trading bot...")
    run_async(send_telegram_message("Trading bot started."))
    price_monitor_manager = PriceMonitorManager()
    watchdog_thread = threading.Thread(
        target=watchdog, args=(price_monitor_manager,), daemon=True
    )
    watchdog_thread.start()

    try:
        markets = bitvavo.load_markets()
        eur_pairs = [
            symbol
            for symbol in markets
            if symbol.endswith("/EUR") and markets[symbol].get("active", False)
        ]
        logger.info(f"Loaded {len(markets)} markets, {len(eur_pairs)} active EUR pairs")

        while True:
            global last_cycle_time
            last_cycle_time = time.time()
            all_data = []
            logger.info("Fetching new data cycle...")

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
            logger.info(f"Processing {len(symbols)} EUR symbols")

            active_monitors = price_monitor_manager.active_monitors()
            adjusted_concurrency = max(
                1, min(CONCURRENT_REQUESTS - active_monitors, 20)
            )
            logger.info(
                f"Active monitors: {active_monitors}, Adjusted concurrency: {adjusted_concurrency}"
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
                above_threshold_data, percent_changes = verify_and_analyze_data(
                    combined_df, price_monitor_manager
                )
                manage_portfolio(
                    above_threshold_data, percent_changes, price_monitor_manager
                )
                save_portfolio()
                with portfolio_lock:
                    total_value = portfolio["cash"] + sum(
                        asset["quantity"] * asset["current_price"]
                        for asset in portfolio["assets"].values()
                    )
                    logger.info(
                        f"Portfolio Status: Cash: {portfolio['cash']:.2f} EUR, "
                        f"Assets: {len(portfolio['assets'])}, "
                        f"Total Value: {total_value:.2f} EUR"
                    )
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
                f"Cycle completed in {elapsed_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds."
            )
            logger.debug(f"Active threads: {threading.active_count()}")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping bot...")
        price_monitor_manager.stop_all()
        save_portfolio()
        run_async(send_telegram_message("Trading bot stopped."))
        from .notifications import shutdown_loop

        shutdown_loop()
        logger.info("Bot stopped successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Critical error in main loop: {e}", exc_info=True)
        run_async(send_telegram_message(f"Main loop error: {e}. Continuing execution."))
        time.sleep(5)


if __name__ == "__main__":
    main()
