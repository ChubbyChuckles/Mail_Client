# trading_bot/src/main.py
import time
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from .config import (
    logger,
    LOOP_INTERVAL_SECONDS,
    RESULTS_FOLDER,
    PARQUET_FILENAME,
    CONCURRENT_REQUESTS,
)
from .exchange import bitvavo, fetch_klines
from .data_processor import verify_and_analyze_data
from .portfolio import manage_portfolio, save_portfolio
from .state import portfolio, portfolio_lock, low_volatility_assets
from .storage import save_to_local
from .notifications import run_async, send_telegram_message
import threading
import sys
from .price_monitor import PriceMonitorManager  


def main():
    logger.info("Starting trading bot...")
    run_async(send_telegram_message("Trading bot started."))

    # Initialize price monitor manager
    price_monitor_manager = PriceMonitorManager()

    try:
        # Initialize markets
        markets = bitvavo.load_markets()
        eur_pairs = [
            symbol for symbol in markets
            if symbol.endswith('/EUR') and markets[symbol].get('active', False)
        ]
        logger.info(f"Loaded {len(markets)} markets, {len(eur_pairs)} active EUR pairs")

        while True:
            start_time = time.time()
            all_data = []
            logger.info("Fetching new data cycle...")

            # Filter top 300 symbols by volume
            try:
                tickers = bitvavo.fetch_tickers(eur_pairs)
                top_volume = sorted(
                    [(symbol, ticker['quoteVolume']) for symbol, ticker in tickers.items() if ticker.get('quoteVolume')],
                    key=lambda x: x[1],
                    reverse=True
                )[:300]
                symbols = [symbol for symbol, _ in top_volume]
            except Exception as e:
                logger.error(f"Error fetching tickers for volume ranking: {e}")
                symbols = eur_pairs[:300]
            logger.info(f"Processing {len(symbols)} EUR symbols")

            active_monitors = price_monitor_manager.active_monitors()
            adjusted_concurrency = max(1, min(CONCURRENT_REQUESTS - active_monitors, 20))
            logger.info(f"Active monitors: {active_monitors}, Adjusted concurrency: {adjusted_concurrency}")

            # Fetch OHLCV data
            with ThreadPoolExecutor(max_workers=adjusted_concurrency) as executor:
                logger.debug(f"Starting ThreadPoolExecutor with {adjusted_concurrency} workers")
                results = executor.map(fetch_klines, symbols)
                for symbol, result in zip(symbols, results):
                    logger.debug(f"Received result for {symbol}")
                    if not result.empty:
                        all_data.append(result)
                logger.debug("ThreadPoolExecutor completed")

            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                output_path = f"{RESULTS_FOLDER}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{PARQUET_FILENAME}"
                save_to_local(combined_df, output_path)

                # Analyze data and manage portfolio
                above_threshold_data, percent_changes = verify_and_analyze_data(combined_df, price_monitor_manager)
                manage_portfolio(above_threshold_data, percent_changes, price_monitor_manager)
                save_portfolio()

                # Log portfolio status
                with portfolio_lock:
                    total_value = portfolio['cash'] + sum(
                        asset['quantity'] * asset['current_price']
                        for asset in portfolio['assets'].values()
                    )
                    logger.info(
                        f"Portfolio Status: Cash: {portfolio['cash']:.2f} EUR, "
                        f"Assets: {len(portfolio['assets'])}, "
                        f"Total Value: {total_value:.2f} EUR"
                    )
            else:
                logger.warning("No data collected in this cycle.")

            # Clean up inactive monitors
            try:
                with portfolio_lock:
                    active_assets = set(portfolio['assets'].keys())
                    for symbol in list(price_monitor_manager.running.keys()):
                        if symbol not in active_assets:
                            logger.warning(f"Stopping orphaned monitor for {symbol}")
                            price_monitor_manager.stop(symbol)
                    for symbol in active_assets:
                        if symbol not in price_monitor_manager.running and symbol not in low_volatility_assets:
                            price_monitor_manager.start(symbol, portfolio, portfolio_lock, combined_df)
            except Exception as e:
                logger.error(f"Error managing price monitors: {e}")

            # Sleep until next cycle
            elapsed_time = time.time() - start_time
            sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed_time)
            logger.info(f"Cycle completed in {elapsed_time:.2f} seconds. Sleeping for {sleep_time:.2f} seconds.")
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
        run_async(send_telegram_message(f"Bot crashed: {e}"))
        price_monitor_manager.stop_all()
        save_portfolio()
        sys.exit(1)

if __name__ == "__main__":
    main()