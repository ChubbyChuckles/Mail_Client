import logging
import json
import os
from datetime import datetime
from urllib.parse import urlencode
import requests
from python_bitvavo_api.bitvavo import Bitvavo
from src.config import API_KEY, API_SECRET  # Replace with your config or use env variables

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

def sell_crypto_asset(symbol, amount_quote):
    """
    Sells a specified amount in EUR of a crypto asset on Bitvavo with a market order,
    calculates slippage, and saves response (including slippage) as JSON in assets_sold directory.
    
    Parameters:
    - symbol (str): The asset symbol (e.g., 'BTC' for Bitcoin).
    - amount_quote (float): Amount in EUR to sell (e.g., 5.5 EUR).
    
    Returns:
    - dict: Response with order details and slippage percentage (if available).
    """
    try:
        # Construct market pair (assuming EUR as quote currency)
        market = f"{symbol}-EUR"
        
        # Check rate limit before proceeding
        rate_limit = get_rate_limit_remaining()
        if rate_limit is not None and rate_limit < 10:
            logger.warning(f"Low rate limit remaining: {rate_limit}. Aborting order.")
            return {'error': f"Low rate limit remaining: {rate_limit}"}
        
        logger.info(f"Placing market sell order for {amount_quote} EUR of {symbol} on {market}")
        
        # Get the current market price from the order book for slippage calculation
        order_book = bitvavo.book(market, {'depth': 1})
        if not order_book.get('asks'):
            logger.error("Failed to retrieve order book")
            return {'error': 'Failed to retrieve order book'}
        
        # Use the best ask price as the expected price (for selling)
        expected_price = float(order_book['asks'][0][0])
        logger.info(f"Expected price: {expected_price} EUR")
        
        # Prepare body for market order
        body = {
            'amountQuote': str(amount_quote)
        }
        
        # Place market sell order
        response = bitvavo.placeOrder(market, 'sell', 'market', body)
        
        # Check if order was successful
        if 'orderId' not in response:
            logger.error(f"Error placing order: {response.get('error', 'Unknown error')}")
            return {'error': response.get('error', 'Unknown error')}
        
        # Extract actual price from the fills array
        fills = response.get('fills', [])
        if not fills:
            logger.warning("No fills found in order response. Cannot calculate slippage.")
            response['slip'] = None
        else:
            # Calculate weighted average price from fills
            total_amount = 0
            weighted_price_sum = 0
            for fill in fills:
                fill_amount = float(fill.get('amount', 0))
                price = float(fill.get('price', 0))
                if fill_amount == 0 or price == 0:
                    logger.warning(f"Invalid fill data: amount={fill_amount}, price={price}")
                    continue
                total_amount += fill_amount
                weighted_price_sum += fill_amount * price
            
            if total_amount == 0:
                logger.warning("Unable to calculate actual price from fills")
                response['slip'] = None
            else:
                actual_price = weighted_price_sum / total_amount
                slippage_percent = ((actual_price - expected_price) / expected_price) * 100
                logger.info(f"Actual price: {actual_price} EUR")
                logger.info(f"Slippage: {slippage_percent:.2f}%")
                response['slip'] = slippage_percent
        
        # Save response as JSON in assets_sold directory (including slippage)
        try:
            directory = "assets_sold"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Generate sensible file name: market_orderId_timestamp.json
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"{market}_{response.get('orderId')}_{timestamp}.json"
            file_path = os.path.join(directory, file_name)
            
            # Write response to JSON file
            with open(file_path, 'w') as json_file:
                json.dump(response, json_file, indent=4)
            logger.info(f"Sold order response saved to: {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON file: {str(e)}")
        
        logger.info(f"Sold order placed successfully: {response['orderId']}")
        return response
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Example: Sell 5.5 EUR worth of Bitcoin
    result = sell_crypto_asset(
        symbol='BTC',
        amount_quote=5.5
    )