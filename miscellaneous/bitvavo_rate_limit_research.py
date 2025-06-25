#!/usr/bin/env python3
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo
import ccxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_book_weight_probe.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BITVAVO_API_KEY") or os.getenv("APIKEY")
API_SECRET = os.getenv("BITVAVO_API_SECRET") or os.getenv("SECKEY")

if not API_KEY or not API_SECRET:
    logger.error("API_KEY or API_SECRET not found in .env file. Please set BITVAVO_API_KEY/API_SECRET or APIKEY/SECKEY.")
    exit(1)

def get_rate_limit_remaining(bitvavo):
    """Fetch remaining rate limit using python-bitvavo-api."""
    try:
        remaining = bitvavo.getRemainingLimit()
        logger.info(f"Remaining rate limit: {remaining}/1000")
        return remaining
    except Exception as e:
        logger.error(f"Error fetching rate limit: {e}")
        return None

def fetch_order_book_snapshot(symbol, depth=200, per_request_delay=0):
    """Fetch order book snapshot with dynamic delay adjustment."""
    try:
        bitvavo = ccxt.bitvavo({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': False
        })
        market = symbol.replace("/", "-")
        start_fetch = time.time()
        order_book = bitvavo.fetch_order_book(symbol, limit=depth)  # Weight: 2
        fetch_time = time.time() - start_fetch
        
        # Adjust delay to account for fetch time
        adjusted_delay = max(0, per_request_delay - fetch_time)
        time.sleep(adjusted_delay)
        total_time = fetch_time + adjusted_delay
        
        if 'bids' not in order_book or 'asks' not in order_book:
            logger.error(f"Process {os.getpid()}: Invalid order book response for {symbol}: {order_book}")
            return symbol, None, total_time
        
        bids_count = len(order_book['bids'])
        asks_count = len(order_book['asks'])
        order_book['timestamp'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Process {os.getpid()}: Fetched order book for {symbol} with depth={depth}: "
                    f"{bids_count} bids, {asks_count} asks in {fetch_time:.3f} seconds, "
                    f"adjusted delay: {adjusted_delay:.3f} seconds, total: {total_time:.3f} seconds")
    except Exception as e:
        return None
        

def main():
    logger.info(f"Starting order book weight probe at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CEST)")
    
    try:
        # Initialize Bitvavo client
        bitvavo = Bitvavo({
            'APIKEY': API_KEY,
            'APISECRET': API_SECRET
        })
        logger.info("Bitvavo client initialized successfully.")
        
        while True:
            start_time = time.time()
            
            # Probe remaining rate limit before order book fetch
            remaining_before = get_rate_limit_remaining(bitvavo)
            if remaining_before is None:
                logger.warning("Failed to fetch rate limit. Retrying in 1 second.")
                time.sleep(1)
                continue
            
            # Fetch order book to consume weight
            order_book = fetch_order_book_snapshot(symbol='BTC-EUR', depth=200)
            if order_book is None:
                logger.warning("Failed to fetch order book. Checking rate limit anyway.")
            
            # Probe remaining rate limit after order book fetch
            remaining_after = get_rate_limit_remaining(bitvavo)
            if remaining_after is not None and remaining_before is not None:
                weight_used = remaining_before - remaining_after
                logger.info(f"Weight used by order book fetch: {weight_used}")
            
            # Sleep to align with 1-second intervals
            elapsed = time.time() - start_time
            sleep_time = max(0, 1.0 - elapsed)
            time.sleep(sleep_time)
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting.")
    except Exception as e:
        logger.error(f"Critical error: {e}")
    
    logger.info(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CEST)")

if __name__ == "__main__":
    main()