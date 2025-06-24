import time
import logging
import requests
from urllib.parse import urlencode
from python_bitvavo_api.bitvavo import Bitvavo
from src.config import API_KEY, API_SECRET

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Bitvavo client
bitvavo = Bitvavo({
    'APIKEY': API_KEY,
    'APISECRET': API_SECRET,
    'RESTURL': 'https://api.bitvavo.com/v2',
    'WSURL': 'wss://ws.bitvavo.com/v2/'
})

# Base URL for direct REST API calls
BASE_URL = "https://api.bitvavo.com/v2"

def get_rate_limit_remaining(endpoint="/time", params=None):
    """Fetch the remaining rate limit weight using a direct REST call."""
    try:
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += "?" + urlencode(params)
        
        # Try without authentication since /time is public
        logger.debug(f"Making unauthenticated request to {url}")
        response = requests.get(url)
        response.raise_for_status()
        remaining = int(response.headers.get('bitvavo-ratelimit-remaining', -1))
        logger.info(f"Rate limit remaining for {endpoint}: {remaining}")
        return remaining
    except Exception as e:
        logger.error(f"Error fetching rate limit for {endpoint}: {e}")
        # Fallback to a different public endpoint if /time fails
        if endpoint == "/time":
            logger.info("Falling back to /markets endpoint")
            return get_rate_limit_remaining(endpoint="/markets")
        return None

def measure_action_weight(action_name, action_func):
    """Measure the weight consumed by an API action."""
    logger.info(f"Measuring weight for action: {action_name}")
    
    # Get initial rate limit
    initial_remaining = get_rate_limit_remaining()
    if initial_remaining is None:
        logger.error("Failed to get initial rate limit. Aborting.")
        return None
    
    # Wait briefly to ensure we're in a fresh rate limit window
    time.sleep(1)
    
    # Perform the action
    try:
        action_func()
        logger.info(f"Action {action_name} completed successfully")
    except Exception as e:
        logger.error(f"Error performing action {action_name}: {e}")
        return None
    
    # Get rate limit after action
    final_remaining = get_rate_limit_remaining()
    if final_remaining is None:
        logger.error("Failed to get final rate limit. Aborting.")
        return None
    
    # Calculate weight used
    if initial_remaining >= final_remaining:
        weight_used = initial_remaining - final_remaining
        logger.info(f"Weight used by {action_name}: {weight_used}")
        return weight_used
    else:
        logger.warning("Rate limit window may have reset. Weight calculation unreliable.")
        return None

# Define actions to test
actions = [
    {
        "name": "getTicker (GET /ticker/24h single market)",
        "func": lambda: bitvavo.ticker24h({'market': 'BTC-EUR'})
    },
    {
        "name": "getCandles (GET /candles)",
        "func": lambda: bitvavo.candles('BTC-EUR', '1h', {'limit': 100})
    },
    {
        "name": "getTrades (GET /trades)",
        "func": lambda: bitvavo.trades('BTC-EUR', {'limit': 100})
    },
    {
        "name": "getTicker (GET /ticker/24h all markets)",
        "func": lambda: bitvavo.ticker24h()
    }
]

def main():
    logger.info("Starting Bitvavo API weight measurement")
    
    # Measure weight for each action
    for action in actions:
        weight = measure_action_weight(action["name"], action["func"])
        if weight is not None:
            logger.info(f"Action {action['name']} consumed {weight} weight points")
        time.sleep(2)  # Avoid rate limit issues between actions
    
    logger.info("Weight measurement completed")

if __name__ == "__main__":
    main()