#!/usr/bin/env python3
import ccxt
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from multiprocessing import Pool, Lock
from functools import partial
from python_bitvavo_api.bitvavo import Bitvavo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('order_book_snapshot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BITVAVO_API_KEY") or os.getenv("APIKEY")
API_SECRET = os.getenv("BITVAVO_API_SECRET") or os.getenv("SECKEY")
SNAPSHOTS_DIR = os.getenv("ORDER_BOOK_SNAPSHOTS_DIR", "F:\Order_Book_Data")
MAX_DEPTH = int(os.getenv("ORDER_BOOK_DEPTH", 1000))
MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))
RATE_LIMIT_WEIGHT = 900  # 90% of 1000 weight units
CYCLE_DURATION = 58  # 58 seconds for buffer
TOP_COINS_FILE = "top_coins.json"
TOP_COINS_LIMIT = 300

if not API_KEY or not API_SECRET:
    logger.error("API_KEY or API_SECRET not found in .env file")
    exit(1)

# Shared lock for API access
lock = Lock()

# Global variable to store average fetch time
average_fetch_time = 0.8  # Initial estimate (seconds)

def get_rate_limit_remaining(bitvavo):
    """Fetch remaining rate limit using python-bitvavo-api."""
    try:
        remaining = bitvavo.getRemainingLimit()
        logger.debug(f"Remaining rate limit: {remaining}/1000")
        return remaining
    except Exception as e:
        logger.error(f"Error fetching rate limit: {e}")
        return None

def wait_for_minute_reset(bitvavo):
    """Wait until the next minute reset (e.g., 18:49:00)."""
    while True:
        now = datetime.utcnow()
        seconds_until_reset = 60 - (now.second + now.microsecond / 1000000)
        if seconds_until_reset < 0:
            seconds_until_reset += 60
        
        # Check rate limit to confirm reset
        remaining = get_rate_limit_remaining(bitvavo)
        if remaining is None:
            logger.warning("Failed to fetch rate limit. Waiting 1 second.")
            time.sleep(1)
            continue
        
        if remaining >= RATE_LIMIT_WEIGHT:
            logger.info(f"Rate limit reset confirmed at {now.strftime('%Y-%m-%d %H:%M:%S')} (UTC). "
                        f"Remaining: {remaining}/1000")
            time.sleep(seconds_until_reset)  # Ensure we start at :00
            return time.time()
        
        # Sleep until close to reset time
        sleep_time = max(0, seconds_until_reset - 0.5)
        logger.debug(f"Waiting {sleep_time:.2f} seconds for next minute reset.")
        time.sleep(sleep_time)

def fetch_top_coins(bitvavo, limit=TOP_COINS_LIMIT):
    """Fetch the top coins by 24-hour volume and save to top_coins.json."""
    try:
        tickers = bitvavo.ticker24h({})
        if not tickers:
            logger.error("No ticker data received.")
            return []
        
        # Filter for EUR pairs and sort by volume
        eur_tickers = [
            ticker for ticker in tickers
            if ticker.get('market', '').endswith('-EUR')
            and ticker.get('volume', 0) is not None
        ]
        sorted_tickers = sorted(
            eur_tickers,
            key=lambda x: float(x.get('volume', 0)),
            reverse=True
        )
        
        # Extract top coins (market symbols in ccxt format, e.g., 'BTC/EUR')
        top_coins = [ticker['market'].replace('-', '/') for ticker in sorted_tickers[:limit]]
        
        # Save to top_coins.json
        try:
            with open(TOP_COINS_FILE, 'w') as f:
                json.dump({'coins': top_coins, 'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=4)
            logger.info(f"Saved {len(top_coins)} top coins to {TOP_COINS_FILE}")
        except Exception as e:
            logger.error(f"Error saving top coins to {TOP_COINS_FILE}: {e}")
        
        return top_coins
    
    except Exception as e:
        logger.error(f"Error fetching top coins: {e}")
        return []

def load_top_coins():
    """Load top coins from top_coins.json if it exists."""
    if not os.path.exists(TOP_COINS_FILE):
        logger.info(f"{TOP_COINS_FILE} does not exist. Will fetch top coins.")
        return None
    try:
        with open(TOP_COINS_FILE, 'r') as f:
            data = json.load(f)
        coins = data.get('coins', [])
        if not coins:
            logger.warning(f"No coins found in {TOP_COINS_FILE}")
            return None
        logger.info(f"Loaded {len(coins)} top coins from {TOP_COINS_FILE}")
        return coins
    except Exception as e:
        logger.error(f"Error loading top coins from {TOP_COINS_FILE}: {e}")
        return None

def fetch_order_book_snapshot(symbol, depth=MAX_DEPTH, per_request_delay=0):
    """Fetch order book snapshot with dynamic delay adjustment."""
    try:
        bitvavo = ccxt.bitvavo({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': False
        })
        market = symbol.replace("/", "-")
        start_fetch = time.time()
        with lock:
            order_book = bitvavo.fetch_order_book(symbol, limit=depth)  # Weight: 2
        fetch_time = time.time() - start_fetch
        
        # Adjust delay to account for fetch time
        adjusted_delay = max(0, per_request_delay - fetch_time)
        time.sleep(adjusted_delay)
        total_time = fetch_time + adjusted_delay
        
        if 'bids' not in order_book or 'asks' not in order_book:
            logger.error(f"Process {os.getpid()}: Invalid order book response for {symbol}: {order_book}")
            return symbol, None, total_time, fetch_time
        
        bids_count = len(order_book['bids'])
        asks_count = len(order_book['asks'])
        order_book['timestamp'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"Process {os.getpid()}: Fetched order book for {symbol} with depth={depth}: "
                    f"{bids_count} bids, {asks_count} asks in {fetch_time:.3f} seconds, "
                    f"adjusted delay: {adjusted_delay:.3f} seconds, total: {total_time:.3f} seconds")
        
        return symbol, order_book, total_time, fetch_time
    
    except ccxt.RateLimitExceeded:
        logger.error(f"Process {os.getpid()}: Rate limit exceeded for {symbol}")
        time.sleep(per_request_delay)  # Ensure slot is used
        return symbol, None, per_request_delay, 0
    except Exception as e:
        logger.error(f"Process {os.getpid()}: Error fetching order book for {symbol}: {e}")
        time.sleep(per_request_delay)
        return symbol, None, per_request_delay, 0

def save_combined_snapshots(snapshots, filename):
    """Save all snapshots to a single JSON file."""
    try:
        start_save = time.time()
        combined_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "snapshots": snapshots
        }
        with open(filename, 'w') as f:
            json.dump(combined_data, f, indent=4)
        save_time = time.time() - start_save
        logger.info(f"Saved combined snapshots to {filename} in {save_time:.3f} seconds")
        return save_time
    except Exception as e:
        logger.error(f"Error saving combined snapshots to {filename}: {e}")
        return 0

def extract_snapshot_by_symbol(filename, symbol):
    """Extract a snapshot for a specific symbol from the combined JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        market = symbol.replace("/", "-")
        snapshot = data.get("snapshots", {}).get(market)
        if snapshot:
            logger.info(f"Extracted snapshot for {symbol} from {filename}")
            return snapshot
        else:
            logger.error(f"No snapshot found for {symbol} in {TOP_COINS_FILE}")
            return None
    except Exception as e:
        logger.error(f"Error extracting snapshot for {symbol} from {filename}: {e}")
        return None

def main():
    global average_fetch_time
    logger.info(f"Starting order book snapshot script at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CEST)")
    
    try:
        # Initialize Bitvavo clients
        bitvavo_api = Bitvavo({
            'APIKEY': API_KEY,
            'APISECRET': API_SECRET
        })
        bitvavo_ccxt = ccxt.bitvavo({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'enableRateLimit': False
        })
        logger.info("Bitvavo clients initialized successfully.")
        
        # Load or fetch top coins
        eur_pairs = load_top_coins()
        if eur_pairs is None:
            logger.info("Fetching top coins by volume.")
            eur_pairs = fetch_top_coins(bitvavo_api, limit=TOP_COINS_LIMIT)
            if not eur_pairs:
                logger.error("No top coins retrieved. Exiting.")
                return
        
        logger.info(f"Processing {len(eur_pairs)} -EUR pairs: {', '.join(eur_pairs[:5])}...")
        
        # Create snapshot directory
        os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
        
        # Calculate initial per-request delay
        expected_weight = len(eur_pairs) * 2
        if expected_weight > RATE_LIMIT_WEIGHT:
            logger.error(f"Expected weight ({expected_weight}) exceeds rate limit ({RATE_LIMIT_WEIGHT}). Exiting.")
            return
        total_requests = len(eur_pairs)
        pairs_per_process = total_requests // MAX_PROCESSES + (1 if total_requests % MAX_PROCESSES else 0)
        per_request_delay = (CYCLE_DURATION / MAX_PROCESSES) / pairs_per_process if total_requests > 0 else 0
        logger.info(f"Expected weight: {expected_weight}/{RATE_LIMIT_WEIGHT}. "
                    f"Initial per-request delay: {per_request_delay:.3f} seconds for {pairs_per_process} pairs per process")
        
        # Main loop
        while True:
            # Wait for minute reset
            cycle_start = wait_for_minute_reset(bitvavo_api)
            
            # Update per-request delay based on average fetch time
            if average_fetch_time > 0:
                per_request_delay = max(0.01, ((CYCLE_DURATION / MAX_PROCESSES) - pairs_per_process * average_fetch_time) / pairs_per_process)
                logger.info(f"Updated per-request delay: {per_request_delay:.3f} seconds based on "
                            f"average fetch time: {average_fetch_time:.3f} seconds for {pairs_per_process} pairs per process")
            
            # Fetch snapshots using multiprocessing
            logger.info(f"Starting data collection at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} (UTC)")
            with Pool(processes=MAX_PROCESSES) as pool:
                fetch_partial = partial(fetch_order_book_snapshot, depth=MAX_DEPTH, per_request_delay=per_request_delay)
                results = pool.map(fetch_partial, eur_pairs)
            
            # Collect results and update average fetch time
            snapshots = {}
            total_fetch_time = 0
            successful_pairs = 0
            fetch_times = []
            for symbol, order_book, request_time, fetch_time in results:
                if order_book:
                    market = symbol.replace("/", "-")
                    snapshots[market] = order_book
                    total_fetch_time += request_time
                    successful_pairs += 1
                    if fetch_time > 0:
                        fetch_times.append(fetch_time)
                else:
                    logger.warning(f"No snapshot collected for {symbol}")
            
            # Update average fetch time
            if fetch_times:
                average_fetch_time = sum(fetch_times) / len(fetch_times)
                logger.info(f"Updated average fetch time: {average_fetch_time:.3f} seconds based on {len(fetch_times)} successful fetches")
            
            logger.info(f"Successfully fetched snapshots for {successful_pairs}/{len(eur_pairs)} pairs")
            
            # Save combined snapshots
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SNAPSHOTS_DIR, f"combined_snapshots_{timestamp}.json")
            save_time = save_combined_snapshots(snapshots, filename)
            
            # Log cycle timing
            cycle_time = time.time() - cycle_start
            logger.info(f"Data collection completed in {cycle_time:.3f} seconds "
                        f"(Fetch: {total_fetch_time:.3f} seconds, Save: {save_time:.3f} seconds)")
            
            # Check final rate limit
            final_remaining = get_rate_limit_remaining(bitvavo_api)
            if final_remaining is not None:
                logger.info(f"Rate limit after cycle: {final_remaining}/1000")
            
            # Demonstrate extraction (example for BTC-EUR)
            if "BTC-EUR" in snapshots:
                extracted = extract_snapshot_by_symbol(filename, "BTC-EUR")
                if extracted:
                    logger.info(f"Example extraction: BTC-EUR snapshot has {len(extracted['bids'])} bids, "
                                f"{len(extracted['asks'])} asks")
            
            # Check for rate limit issues
            if final_remaining is not None and final_remaining < 100:
                logger.warning(f"Low rate limit ({final_remaining}/1000). Waiting for next minute reset.")
            
            # Wait for next reset
            logger.info("Waiting for next minute reset...")
    
    except KeyboardInterrupt:
        logger.info("Script interrupted by user. Exiting.")
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
    
    logger.info(f"Script completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (CEST)")

if __name__ == "__main__":
    main()