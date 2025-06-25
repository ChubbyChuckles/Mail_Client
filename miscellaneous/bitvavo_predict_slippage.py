import logging
from urllib.parse import urlencode

import requests
from python_bitvavo_api.bitvavo import Bitvavo

from src.config import (  # Replace with your config or use env variables
    API_KEY, API_SECRET)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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


def predict_slippage(market, amount_quote, side):
    """
    Predicts the slippage percentage for a market order by analyzing the order book.

    Parameters:
    - market (str): The trading pair (e.g., 'BTC-EUR').
    - amount_quote (float): Amount in EUR to buy or sell (e.g., 5.5 EUR).
    - side (str): 'buy' or 'sell'.

    Returns:
    - dict: Predicted slippage percentage, expected price, and predicted price (or error).
    """
    try:
        if side not in ["buy", "sell"]:
            logger.error(f"Invalid side: {side}. Must be 'buy' or 'sell'.")
            return {"error": f"Invalid side: {side}"}

        # Check rate limit before proceeding
        rate_limit = get_rate_limit_remaining()
        if rate_limit is not None and rate_limit < 10:
            logger.warning(
                f"Low rate limit remaining: {rate_limit}. Aborting slippage prediction."
            )
            return {"error": f"Low rate limit remaining: {rate_limit}"}

        logger.info(
            f"Predicting slippage for {side} order of {amount_quote} EUR on {market}"
        )

        # Fetch order book with sufficient depth
        order_book = bitvavo.book(market, {"depth": 100})
        if not order_book.get("bids") or not order_book.get("asks"):
            logger.error("Failed to retrieve order book")
            return {"error": "Failed to retrieve order book"}

        # Get expected price
        if side == "buy":
            expected_price = float(order_book["asks"][0][0])  # Best ask for buys
            levels = order_book["asks"]  # Consume asks
        else:  # sell
            expected_price = float(order_book["bids"][0][0])  # Best bid for sells
            levels = order_book["bids"]  # Consume bids

        logger.info(f"Expected price: {expected_price} EUR")

        # Estimate base asset amount
        base_amount = amount_quote / expected_price
        logger.info(f"Estimated base amount: {base_amount:.8f} {market.split('-')[0]}")

        # Walk the order book to calculate predicted execution price
        total_amount = 0
        weighted_price_sum = 0
        for level in levels:
            price = float(level[0])
            amount = float(level[1])

            # Determine how much of this level is needed
            remaining_amount = base_amount - total_amount
            if remaining_amount <= 0:
                break

            amount_to_use = min(amount, remaining_amount)
            total_amount += amount_to_use
            weighted_price_sum += amount_to_use * price

            if total_amount >= base_amount:
                break

        if total_amount < base_amount:
            logger.warning(
                f"Insufficient liquidity in order book to fill {base_amount:.8f} {market.split('-')[0]}"
            )
            return {"error": "Insufficient liquidity in order book"}

        # Calculate predicted execution price
        predicted_price = weighted_price_sum / total_amount
        slippage_percent = ((predicted_price - expected_price) / expected_price) * 100

        logger.info(f"Predicted execution price: {predicted_price:.2f} EUR")
        logger.info(f"Predicted slippage: {slippage_percent:.8f}%")

        return {
            "slippage_percent": slippage_percent,
            "expected_price": expected_price,
            "predicted_price": predicted_price,
            "base_amount": base_amount,
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Example: Predict slippage for buying and selling 5.5 EUR worth of BTC
    for side in ["buy", "sell"]:
        result = predict_slippage(market="ZKJ-EUR", amount_quote=5500, side=side)
        print(result)
