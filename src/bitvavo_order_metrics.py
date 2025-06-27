# trading_bot/src/bitvavo_order_metrics.py
import logging
from urllib.parse import urlencode
import requests
from python_bitvavo_api.bitvavo import Bitvavo
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from . import config
from .config import logger
from .exchange import check_rate_limit
from .order_book_buy_score import analyze_buy_decision

# Initialize Bitvavo client
bitvavo = Bitvavo(
    {
        "APIKEY": config.config.API_KEY,
        "APISECRET": config.config.API_SECRET,
        "RESTURL": "https://api.bitvavo.com/v2",
        "WSURL": "wss://ws.bitvavo.com/v2/",
    }
)

# Base URL for direct REST API calls
BASE_URL = "https://api.bitvavo.com/v2"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.RequestException, requests.HTTPError)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying bitvavo.book after {retry_state.attempt_number} attempts"
    ),
    reraise=True
)
def fetch_order_book_with_retry(market, depth=1000):
    """
    Fetches the order book for a given market with retry logic.

    Args:
        market (str): The trading pair (e.g., 'BTC-EUR').
        depth (int): Number of order book levels to fetch.

    Returns:
        dict: Order book data.

    Raises:
        requests.RequestException: For network-related errors.
        ValueError: If the order book data is invalid.
    """
    try:
        check_rate_limit(1)  # Assume order book fetch has a weight of 1
        order_book = bitvavo.book(market, {"depth": depth})
        if not isinstance(order_book, dict) or not order_book.get("bids") or not order_book.get("asks"):
            raise ValueError(f"Invalid order book data for {market}")
        return order_book
    except Exception as e:
        logger.error(f"Error fetching order book for {market}: {e}", exc_info=True)
        raise

def calculate_order_book_metrics(market, amount_quote=5.5, price_range_percent=10.0):
    """
    Derives all possible market metrics from the order book for a given market.

    Args:
        market (str): The trading pair (e.g., 'BTC-EUR').
        amount_quote (float): Amount in EUR for slippage prediction (default: 5.5 EUR).
        price_range_percent (float): Price range for depth and imbalance calculations (default: 10%).

    Returns:
        dict: Dictionary containing all derived metrics or an error message.

    Raises:
        ValueError: If inputs are invalid.
        requests.RequestException: For network-related errors during API calls.
    """
    try:
        # Input validation
        if not isinstance(market, str) or not market:
            raise ValueError(f"Invalid market: {market}")
        if not isinstance(amount_quote, (int, float)) or amount_quote <= 0:
            raise ValueError(f"Invalid amount_quote: {amount_quote}")
        if not isinstance(price_range_percent, (int, float)) or price_range_percent <= 0:
            raise ValueError(f"Invalid price_range_percent: {price_range_percent}")

        # Fetch order book
        order_book = fetch_order_book_with_retry(market, depth=1000)
        
        # Initialize metrics
        metrics = {
            "market": order_book.get("market", market),
            "nonce": order_book.get("nonce"),
            "best_bid": None,
            "best_ask": None,
            "spread": None,
            "spread_percentage": None,
            "mid_price": None,
            "buy_depth": None,
            "sell_depth": None,
            "total_depth": None,
            "bid_levels_count": len(order_book.get("bids", [])),
            "ask_levels_count": len(order_book.get("asks", [])),
            "bid_volume": None,
            "ask_volume": None,
            "bid_value": None,
            "ask_value": None,
            "order_book_imbalance": None,
            "avg_bid_price": None,
            "avg_ask_price": None,
            "vwap_bid": None,
            "vwap_ask": None,
            "slippage_buy": None,
            "slippage_sell": None,
            "predicted_price_buy": None,
            "predicted_price_sell": None,
            "total_score": None,
            "recommendation": None,
        }

        # 1. Best Bid and Ask
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        metrics["best_bid"] = float(bids[0][0]) if bids else None
        metrics["best_ask"] = float(asks[0][0]) if asks else None

        # 2. Spread and Mid Price
        if metrics["best_bid"] is not None and metrics["best_ask"] is not None:
            metrics["spread"] = metrics["best_ask"] - metrics["best_bid"]
            metrics["mid_price"] = (metrics["best_bid"] + metrics["best_ask"]) / 2
            metrics["spread_percentage"] = (
                (metrics["spread"] / metrics["mid_price"]) * 100
                if metrics["mid_price"] else None
            )

        # 3. Depth and Volume within ±price_range_percent
        price_range_low = (
            metrics["mid_price"] * (1 - price_range_percent / 100)
            if metrics["mid_price"] is not None else None
        )
        price_range_high = (
            metrics["mid_price"] * (1 + price_range_percent / 100)
            if metrics["mid_price"] is not None else None
        )

        bid_volume = 0
        bid_value = 0
        bid_weighted_sum = 0
        ask_volume = 0
        ask_value = 0
        ask_weighted_sum = 0

        for bid in bids:
            try:
                price = float(bid[0])
                amount = float(bid[1])
                if price_range_low is not None and price >= price_range_low:
                    bid_volume += amount
                    bid_value += amount * price
                    bid_weighted_sum += amount * price
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Invalid bid entry in order book for {market}: {e}", exc_info=True)
                continue

        for ask in asks:
            try:
                price = float(ask[0])
                amount = float(ask[1])
                if price_range_high is not None and price <= price_range_high:
                    ask_volume += amount
                    ask_value += amount * price
                    ask_weighted_sum += amount * price
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Invalid ask entry in order book for {market}: {e}", exc_info=True)
                continue

        metrics["bid_volume"] = bid_volume
        metrics["bid_value"] = bid_value
        metrics["ask_volume"] = ask_volume
        metrics["ask_value"] = ask_value
        metrics["buy_depth"] = bid_value
        metrics["sell_depth"] = ask_value
        metrics["total_depth"] = bid_value + ask_value if bid_value is not None and ask_value is not None else None

        # 4. Order Book Imbalance
        if bid_volume + ask_volume > 0:
            metrics["order_book_imbalance"] = bid_volume / (bid_volume + ask_volume)

        # 5. Average Prices
        metrics["avg_bid_price"] = bid_value / bid_volume if bid_volume > 0 else None
        metrics["avg_ask_price"] = ask_value / ask_volume if ask_volume > 0 else None
        metrics["vwap_bid"] = bid_weighted_sum / bid_volume if bid_volume > 0 else None
        metrics["vwap_ask"] = ask_weighted_sum / ask_volume if ask_volume > 0 else None

        # 6. Slippage for Buy and Sell
        for side in ["buy", "sell"]:
            expected_price = (
                metrics["best_ask"] if side == "buy" else metrics["best_bid"]
            )
            levels = asks if side == "buy" else bids
            base_amount = amount_quote / expected_price if expected_price else 0

            total_amount = 0
            weighted_price_sum = 0
            for level in levels:
                try:
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
                except (IndexError, ValueError, TypeError) as e:
                    logger.warning(f"Invalid {side} level in order book for {market}: {e}", exc_info=True)
                    continue

            if total_amount < base_amount:
                metrics[f"slippage_{side}"] = None
                metrics[f"predicted_price_{side}"] = None
            else:
                predicted_price = weighted_price_sum / total_amount
                slippage_percent = (
                    (predicted_price - expected_price) / expected_price * 100
                    if expected_price else None
                )
                metrics[f"slippage_{side}"] = slippage_percent
                metrics[f"predicted_price_{side}"] = predicted_price

        # 7. Total Score and Recommendation
        try:
            buy_metrics = analyze_buy_decision(order_book)
            metrics["total_score"] = buy_metrics.get("total_score")
            metrics["recommendation"] = buy_metrics.get("recommendation")
        except Exception as e:
            logger.error(f"Error analyzing buy decision for {market}: {e}", exc_info=True)
            metrics["total_score"] = None
            metrics["recommendation"] = None

        return metrics

    except ValueError as e:
        logger.error(f"Validation error for {market}: {e}", exc_info=True)
        return {"error": f"Validation error: {e}", "market": market}
    except (requests.RequestException, requests.HTTPError) as e:
        logger.error(f"Network error fetching order book for {market}: {e}", exc_info=True)
        send_alert("Order Book Fetch Failure", f"Network error for {market}: {e}")
        return {"error": f"Network error: {e}", "market": market}
    except Exception as e:
        logger.error(f"Unexpected error processing order book for {market}: {e}", exc_info=True)
        send_alert("Order Book Metrics Failure", f"Unexpected error for {market}: {e}")
        return {"error": f"Unexpected error: {e}", "market": market}

def send_alert(subject, message):
    """
    Sends an alert for critical errors (placeholder).

    Args:
        subject (str): The subject of the alert.
        message (str): The alert message.
    """
    logger.error(f"ALERT: {subject} - {message}")