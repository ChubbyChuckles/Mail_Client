import logging
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

def calculate_order_book_metrics(market, amount_quote=5.5, price_range_percent=10.0):
    """
    Derives all possible market metrics from the order book for a given market.
    
    Parameters:
    - market (str): The trading pair (e.g., 'BTC-EUR').
    - amount_quote (float): Amount in EUR for slippage prediction (default: 5.5 EUR).
    - price_range_percent (float): Price range for depth and imbalance calculations (default: 10%).
    
    Returns:
    - dict: Dictionary containing all derived metrics (or error).
    """
    try:
        # Check rate limit
        rate_limit = get_rate_limit_remaining()
        if rate_limit is not None and rate_limit < 10:
            logger.warning(f"Low rate limit remaining: {rate_limit}. Aborting metrics calculation.")
            return {'error': f"Low rate limit remaining: {rate_limit}"}
        
        logger.info(f"Calculating order book metrics for {market}")
        
        # Fetch order book with depth=100
        order_book = bitvavo.book(market, {'depth': 100})
        if not order_book.get('bids') or not order_book.get('asks'):
            logger.error("Failed to retrieve order book")
            return {'error': 'Failed to retrieve order book'}
        
        # Initialize metrics
        metrics = {
            'market': order_book['market'],
            'nonce': order_book['nonce'],
            'best_bid': None,
            'best_ask': None,
            'spread': None,
            'spread_percentage': None,
            'mid_price': None,
            'buy_depth': None,
            'sell_depth': None,
            'total_depth': None,
            'bid_levels_count': len(order_book['bids']),
            'ask_levels_count': len(order_book['asks']),
            'bid_volume': None,
            'ask_volume': None,
            'bid_value': None,
            'ask_value': None,
            'order_book_imbalance': None,
            'avg_bid_price': None,
            'avg_ask_price': None,
            'vwap_bid': None,
            'vwap_ask': None,
            'slippage_buy': None,
            'slippage_sell': None,
            'predicted_price_buy': None,
            'predicted_price_sell': None
        }
        
        # 1. Best Bid and Ask
        metrics['best_bid'] = float(order_book['bids'][0][0]) if order_book['bids'] else None
        metrics['best_ask'] = float(order_book['asks'][0][0]) if order_book['asks'] else None
        
        # 2. Spread and Mid Price
        if metrics['best_bid'] and metrics['best_ask']:
            metrics['spread'] = metrics['best_ask'] - metrics['best_bid']
            metrics['mid_price'] = (metrics['best_bid'] + metrics['best_ask']) / 2
            metrics['spread_percentage'] = (metrics['spread'] / metrics['mid_price']) * 100
        
        # 3. Depth and Volume within ±price_range_percent
        price_range_low = metrics['mid_price'] * (1 - price_range_percent / 100) if metrics['mid_price'] else None
        price_range_high = metrics['mid_price'] * (1 + price_range_percent / 100) if metrics['mid_price'] else None
        
        bid_volume = 0
        bid_value = 0
        bid_weighted_sum = 0
        ask_volume = 0
        ask_value = 0
        ask_weighted_sum = 0
        
        for bid in order_book['bids']:
            price = float(bid[0])
            amount = float(bid[1])
            if price_range_low and price >= price_range_low:
                bid_volume += amount
                bid_value += amount * price
                bid_weighted_sum += amount * price
        
        for ask in order_book['asks']:
            price = float(ask[0])
            amount = float(ask[1])
            if price_range_high and price <= price_range_high:
                ask_volume += amount
                ask_value += amount * price
                ask_weighted_sum += amount * price
        
        metrics['bid_volume'] = bid_volume
        metrics['bid_value'] = bid_value
        metrics['ask_volume'] = ask_volume
        metrics['ask_value'] = ask_value
        metrics['buy_depth'] = bid_value
        metrics['sell_depth'] = ask_value
        metrics['total_depth'] = bid_value + ask_value
        
        # 4. Order Book Imbalance
        if bid_volume + ask_volume > 0:
            metrics['order_book_imbalance'] = bid_volume / (bid_volume + ask_volume)
        
        # 5. Average Prices
        metrics['avg_bid_price'] = bid_value / bid_volume if bid_volume > 0 else None
        metrics['avg_ask_price'] = ask_value / ask_volume if ask_volume > 0 else None
        metrics['vwap_bid'] = bid_weighted_sum / bid_volume if bid_volume > 0 else None
        metrics['vwap_ask'] = ask_weighted_sum / ask_volume if ask_volume > 0 else None
        
        # 6. Slippage for Buy and Sell
        for side in ['buy', 'sell']:
            expected_price = metrics['best_ask'] if side == 'buy' else metrics['best_bid']
            levels = order_book['asks'] if side == 'buy' else order_book['bids']
            base_amount = amount_quote / expected_price if expected_price else 0
            
            total_amount = 0
            weighted_price_sum = 0
            for level in levels:
                price = float(level[0])
                amount = float(level[1])
                remaining_amount = base_amount - total_amount
                if remaining_amount <= 0:
                    break
                amount_to_use = min(amount, remaining_amount)
                total_amount += amount_to_use
                weighted_price_sum += amount_to_use * price
                if total_amount >= base_amount:
                    break
            
            if total_amount < base_amount:
                logger.warning(f"Insufficient liquidity for {side} order of {base_amount:.8f} {market.split('-')[0]}")
                metrics[f'slippage_{side}'] = None
                metrics[f'predicted_price_{side}'] = None
            else:
                predicted_price = weighted_price_sum / total_amount
                slippage_percent = ((predicted_price - expected_price) / expected_price) * 100
                metrics[f'slippage_{side}'] = slippage_percent
                metrics[f'predicted_price_{side}'] = predicted_price
        
        # Log all metrics
        logger.info(f"Market: {metrics['market']}, Nonce: {metrics['nonce']}")
        logger.info(f"Best Bid: {metrics['best_bid']:.2f} EUR")
        logger.info(f"Best Ask: {metrics['best_ask']:.2f} EUR")
        logger.info(f"Spread: {metrics['spread']:.2f} EUR ({metrics['spread_percentage']:.2f}%)")
        logger.info(f"Mid Price: {metrics['mid_price']:.2f} EUR")
        logger.info(f"Buy Depth (±{price_range_percent}%): {metrics['buy_depth']:.2f} EUR")
        logger.info(f"Sell Depth (±{price_range_percent}%): {metrics['sell_depth']:.2f} EUR")
        logger.info(f"Total Depth: {metrics['total_depth']:.2f} EUR")
        logger.info(f"Bid Volume: {metrics['bid_volume']:.8f} {market.split('-')[0]}")
        logger.info(f"Ask Volume: {metrics['ask_volume']:.8f} {market.split('-')[0]}")
        logger.info(f"Bid Value: {metrics['bid_value']:.2f} EUR")
        logger.info(f"Ask Value: {metrics['ask_value']:.2f} EUR")
        logger.info(f"Order Book Imbalance: {metrics['order_book_imbalance']:.2f} (buy pressure)")
        logger.info(f"Bid Levels Count: {metrics['bid_levels_count']}")
        logger.info(f"Ask Levels Count: {metrics['ask_levels_count']}")
        logger.info(f"Average Bid Price: {metrics['avg_bid_price']:.2f} EUR")
        logger.info(f"Average Ask Price: {metrics['avg_ask_price']:.2f} EUR")
        logger.info(f"VWAP Bid: {metrics['vwap_bid']:.2f} EUR")
        logger.info(f"VWAP Ask: {metrics['vwap_ask']:.2f} EUR")
        logger.info(f"Slippage Buy ({amount_quote} EUR): {metrics['slippage_buy']:.2f}%")
        logger.info(f"Predicted Buy Price: {metrics['predicted_price_buy']:.2f} EUR")
        logger.info(f"Slippage Sell ({amount_quote} EUR): {metrics['slippage_sell']:.2f}%")
        logger.info(f"Predicted Sell Price: {metrics['predicted_price_sell']:.2f} EUR")
        
        return metrics
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    # Calculate metrics for BTC-EUR with 5.5 EUR order size
    metrics = calculate_order_book_metrics(
        market='MOG-EUR',
        amount_quote=5.5,
        price_range_percent=10.0
    )
    print(metrics)