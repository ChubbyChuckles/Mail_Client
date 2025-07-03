import logging
import time
from urllib.parse import urlencode
import requests
from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair
OPERATOR_ID = 1001  # Unique integer for trader or algorithm
DEPTH = 100  # Depth for order book
LIMIT = 1000  # Limit for fetching orders
AMOUNT_EUR_BUY = 10  # Minimal amount for test buy order (EUR)
AMOUNT_EUR_SELL = 6  # Minimal amount for test sell order (EUR)
PERCENTAGE_BELOW = 5  # Percentage below market price for buy
PERCENTAGE_ABOVE = 5  # Percentage above market price for sell

# Load environment variables
load_dotenv()
API_KEY = os.getenv("BITVAVO_API_KEY") or os.getenv("APIKEY")
API_SECRET = os.getenv("BITVAVO_API_SECRET") or os.getenv("SECKEY")

if not API_KEY or not API_SECRET:
    logger.error(
        "API_KEY or API_SECRET not found in .env file. Please set BITVAVO_API_KEY/API_SECRET or APIKEY/SECKEY."
    )
    exit(1)

# Initialize Bitvavo client
bitvavo = Bitvavo(
    {
        "APIKEY": API_KEY,
        "APISECRET": API_SECRET,
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/",
    }
)

# Base URL for direct REST API calls
BASE_URL = "https://api.bitvavo.com/v2"

def get_rate_limit_remaining(endpoint="/time", params=None):
    """Fetch the remaining rate limit weight using a direct REST call."""
    try:
        url = f"{BASE_URL}{endpoint}"
        if params:
            url += "?" + urlencode(params)

        logger.debug(f"Making unauthenticated request to {url}")
        response = requests.get(url)
        response.raise_for_status()
        remaining = int(response.headers.get("bitvavo-ratelimit-remaining", -1))
        logger.info(f"Rate limit remaining for {endpoint}: {remaining}")
        return remaining
    except Exception as e:
        logger.error(f"Error fetching rate limit for {endpoint}: {e}")
        if endpoint == "/time":
            logger.info("Falling back to /markets endpoint")
            return get_rate_limit_remaining(endpoint="/markets")
        return None

def measure_action_weight(action_name, action_func):
    """Measure the weight consumed by an API action."""
    logger.info(f"Measuring weight for action: {action_name}")

    initial_remaining = get_rate_limit_remaining()
    if initial_remaining is None:
        logger.error("Failed to get initial rate limit. Aborting.")
        return None

    time.sleep(1)  # Ensure fresh rate limit window

    try:
        action_func()
        logger.info(f"Action {action_name} completed successfully")
    except Exception as e:
        logger.error(f"Error performing action {action_name}: {e}")
        return None

    final_remaining = get_rate_limit_remaining()
    if final_remaining is None:
        logger.error("Failed to get final rate limit. Aborting.")
        return None

    if initial_remaining >= final_remaining:
        weight_used = initial_remaining - final_remaining
        logger.info(f"Weight used by {action_name}: {weight_used}")
        return weight_used
    else:
        logger.warning(
            "Rate limit window may have reset. Weight calculation unreliable."
        )
        return None

def get_current_price():
    """Fetch the current market price for the given symbol."""
    ticker = bitvavo.tickerPrice({"market": SYMBOL})
    if ticker and "price" in ticker:
        return float(ticker["price"])
    raise Exception("Unable to fetch current price.")

def round_to_precision(value, precision):
    """Round a number to the specified number of significant digits."""
    if value == 0:
        return 0
    from math import floor, log10
    scale = precision - 1 - floor(log10(abs(value)))
    return round(value, scale if scale > 0 else 0)

def get_market_precision():
    """Get precision details for the market."""
    markets = bitvavo.markets()
    amount_precision = 8
    price_precision = 5
    min_amount = 0.0
    for market in markets:
        if market["market"] == SYMBOL:
            amount_precision = int(market.get("amountPrecision", 8))
            price_precision = int(market.get("pricePrecision", 5))
            min_amount = float(market.get("minAmount", 0.0))
            break
    return amount_precision, price_precision, min_amount

# Define actions to test
actions = [
    {
        "name": "getTicker (GET /ticker/24h single market)",
        "func": lambda: bitvavo.ticker24h({"market": SYMBOL}),
    },
    {
        "name": "getCandles (GET /candles)",
        "func": lambda: bitvavo.candles(SYMBOL, "1h", {"limit": 100}),
    },
    {
        "name": "getTrades (GET /trades)",
        "func": lambda: bitvavo.trades(SYMBOL, {"limit": 100}),
    },
    {
        "name": "getTicker (GET /ticker/24h all markets)",
        "func": lambda: bitvavo.ticker24h(),
    },
    {
        "name": "fetchOrderBook (GET /book)",
        "func": lambda: bitvavo.book(SYMBOL, {"depth": DEPTH}),
    },
    {
        "name": "listOpenOrders (GET /ordersOpen)",
        "func": lambda: bitvavo.ordersOpen({"market": SYMBOL}),
    },
    {
        "name": "fetchFilledOrders (GET /orders)",
        "func": lambda: bitvavo.getOrders({"market": SYMBOL, "limit": LIMIT}),
    },
    {
        "name": "placeLimitBuyOrder (POST /order)",
        "func": lambda: (
            lambda: bitvavo.placeOrder(
                market=SYMBOL,
                side="buy",
                orderType="limit",
                body={
                    "amount": str(
                        round_to_precision(
                            AMOUNT_EUR_BUY / (get_current_price() * (1 - PERCENTAGE_BELOW / 100)),
                            get_market_precision()[0]
                        )
                    ),
                    "price": str(
                        round_to_precision(
                            get_current_price() * (1 - PERCENTAGE_BELOW / 100),
                            get_market_precision()[1]
                        )
                    ),
                    "operatorId": str(OPERATOR_ID),
                }
            )
        )(),
    },
    {
        "name": "placeLimitSellOrder (POST /order)",
        "func": lambda: (
            lambda: (
                bitvavo.placeOrder(
                    market=SYMBOL,
                    side="sell",
                    orderType="limit",
                    body={
                        "amount": str(
                            round_to_precision(
                                AMOUNT_EUR_SELL / (get_current_price() * (1 + PERCENTAGE_ABOVE / 100)),
                                get_market_precision()[0]
                            )
                        ),
                        "price": str(
                            round_to_precision(
                                get_current_price() * (1 + PERCENTAGE_ABOVE / 100),
                                get_market_precision()[1]
                            )
                        ),
                        "operatorId": str(OPERATOR_ID),
                    }
                )
                if float(bitvavo.balance({"symbol": SYMBOL.split('-')[0]})[0].get("available", 0))
                >= round_to_precision(
                    AMOUNT_EUR_SELL / (get_current_price() * (1 + PERCENTAGE_ABOVE / 100)),
                    get_market_precision()[0]
                )
                else None
            )
        )(),
    },
    {
        "name": "cancelOrder (DELETE /order)",
        "func": lambda: (
            lambda: (
                bitvavo.cancelOrder(
                    market=SYMBOL,
                    orderId=bitvavo.ordersOpen({"market": SYMBOL})[0]["orderId"],
                    operatorId=str(OPERATOR_ID)
                )
                if bitvavo.ordersOpen({"market": SYMBOL})
                else None
            )
        )(),
    },
]

def main():
    logger.info("Starting Bitvavo API weight measurement")

    for action in actions:
        weight = measure_action_weight(action["name"], action["func"])
        if weight is not None:
            logger.info(f"Action {action['name']} consumed {weight} weight points")
        time.sleep(2)  # Avoid rate limit issues

    logger.info("Weight measurement completed")

if __name__ == "__main__":
    main()