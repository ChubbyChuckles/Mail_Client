#!/usr/bin/env python3
import json
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("buy_decision_analysis.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_order_book(json_file, symbol):
    """Load order book snapshot for a symbol from the combined JSON file."""
    try:
        if not os.path.exists(json_file):
            logger.error(f"File {json_file} does not exist.")
            return None
        with open(json_file, "r") as f:
            data = json.load(f)
        market = symbol.replace("/", "-")
        snapshot = data.get("snapshots", {}).get(market)
        if not snapshot:
            logger.error(f"No snapshot found for {symbol} in {json_file}")
            return None
        logger.info(f"Loaded order book snapshot for {symbol} from {json_file}")
        return snapshot
    except Exception as e:
        logger.error(f"Error loading order book from {json_file}: {e}")
        return None


def calculate_mid_price(order_book):
    """Calculate mid-price from best bid and ask."""
    try:
        best_bid = float(order_book["bids"][0][0]) if order_book["bids"] else None
        best_ask = float(order_book["asks"][0][0]) if order_book["asks"] else None
        if best_bid is None or best_ask is None:
            logger.error("No bids or asks available to calculate mid-price.")
            return None
        return (best_bid + best_ask) / 2
    except Exception as e:
        logger.error(f"Error calculating mid-price: {e}")
        return None


def calculate_liquidity_at_level(price, amount):
    """Calculate liquidity value (price × amount) at a price level."""
    return float(price) * float(amount)


def calculate_order_book_imbalance(order_book, mid_price):
    """Calculate bid-to-ask liquidity ratio within ±1% of mid-price."""
    try:
        bid_liquidity = (
            sum(
                calculate_liquidity_at_level(price, amount)
                for price, amount in order_book["bids"]
                if float(price) >= mid_price * 0.99
            )
            if order_book["bids"]
            else 0.0
        )
        ask_liquidity = (
            sum(
                calculate_liquidity_at_level(price, amount)
                for price, amount in order_book["asks"]
                if float(price) <= mid_price * 1.01
            )
            if order_book["asks"]
            else 0.0
        )

        if bid_liquidity == 0.0 and ask_liquidity == 0.0:
            logger.warning(
                "No bids or asks within ±1% of mid-price. Imbalance ratio set to None."
            )
            return None, bid_liquidity, ask_liquidity
        if ask_liquidity == 0.0:
            logger.warning(
                "No asks within ±1% of mid-price. Imbalance ratio set to None."
            )
            return None, bid_liquidity, ask_liquidity
        if bid_liquidity == 0.0:
            logger.warning(
                "No bids within ±1% of mid-price. Imbalance ratio set to 0.0."
            )
            return 0.0, bid_liquidity, ask_liquidity

        imbalance_ratio = bid_liquidity / ask_liquidity
        return imbalance_ratio, bid_liquidity, ask_liquidity
    except Exception as e:
        logger.error(f"Error calculating imbalance: {e}")
        return None, 0.0, 0.0


def detect_cascade_points(
    order_book, mid_price, liquidity_threshold=0.1, gap_threshold=0.001
):
    """Detect potential cascade points on asks within +5% of mid-price."""
    try:
        ask_cascade_points = []
        total_liquidity = (
            sum(
                calculate_liquidity_at_level(price, amount)
                for price, amount in order_book["asks"]
                if mid_price <= float(price) <= mid_price * 1.05
            )
            if order_book["asks"]
            else 0.0
        )
        prev_price = None
        for price, amount in order_book["asks"]:
            price = float(price)
            if not (mid_price <= price <= mid_price * 1.05):
                continue
            liquidity = calculate_liquidity_at_level(price, amount)
            liquidity_ratio = liquidity / total_liquidity if total_liquidity > 0 else 0
            if liquidity_ratio < liquidity_threshold:
                ask_cascade_points.append(
                    (price, amount, f"Low liquidity ({liquidity_ratio:.2%})")
                )
            if prev_price is not None:
                gap = (price - prev_price) / mid_price
                if gap > gap_threshold:
                    ask_cascade_points.append((price, amount, f"Large gap ({gap:.2%})"))
            prev_price = price
        return ask_cascade_points
    except Exception as e:
        logger.error(f"Error detecting cascade points: {e}")
        return []


def calculate_spread(order_book, mid_price):
    """Calculate bid-ask spread as percentage of mid-price."""
    try:
        best_bid = float(order_book["bids"][0][0]) if order_book["bids"] else None
        best_ask = float(order_book["asks"][0][0]) if order_book["asks"] else None
        if best_bid is None or best_ask is None:
            logger.error("No bids or asks available to calculate spread.")
            return None
        spread = (best_ask - best_bid) / mid_price
        return spread
    except Exception as e:
        logger.error(f"Error calculating spread: {e}")
        return None


def evaluate_buy_decision(order_book, mid_price):
    """Evaluate buy decision based on order book metrics."""
    try:
        # 1. Order book imbalance
        imbalance_ratio, bid_liquidity, ask_liquidity = calculate_order_book_imbalance(
            order_book, mid_price
        )
        imbalance_score = (
            0.0 if imbalance_ratio is None else min(1.0, imbalance_ratio / 1.5)
        )  # >1.2 is strong buying pressure
        logger.info(
            f"Imbalance ratio: {imbalance_ratio if imbalance_ratio is not None else 'None'} "
            f"(Bid: {bid_liquidity:.2f}, Ask: {ask_liquidity:.2f}), Score: {imbalance_score:.2f}"
        )

        # 2. Cascade vulnerability
        ask_cascade_points = detect_cascade_points(order_book, mid_price)
        cascade_count = len(ask_cascade_points)
        cascade_score = min(
            1.0, cascade_count / 20
        )  # >10 vulnerable levels is high risk/reward
        logger.info(f"Ask cascade points: {cascade_count}, Score: {cascade_score:.2f}")

        # 3. Spread
        spread = calculate_spread(order_book, mid_price)
        spread_score = (
            max(0.0, 1.0 - spread / 0.0005) if spread else 0.0
        )  # <0.05% spread is good
        logger.info(
            f"Spread: {spread*100:.4f}%{' (None)' if spread is None else ''}, Score: {spread_score:.2f}"
        )

        # Weighted total score (no price impact)
        weights = {"imbalance": 0.5, "cascade": 0.3, "spread": 0.2}
        total_score = (
            weights["imbalance"] * imbalance_score
            + weights["cascade"] * cascade_score
            + weights["spread"] * spread_score
        )
        recommendation = (
            "Strong Buy"
            if total_score > 0.7
            else "Weak Buy" if total_score > 0.5 else "No Buy"
        )
        logger.info(f"Total score: {total_score:.2f}, Recommendation: {recommendation}")

        return {
            "imbalance_ratio": imbalance_ratio,
            "cascade_count": cascade_count,
            "spread": spread,
            "total_score": total_score,
            "recommendation": recommendation,
            "ask_cascade_points": ask_cascade_points,
        }
    except Exception as e:
        logger.error(f"Error evaluating buy decision: {e}")
        return None


def analyze_buy_decision(json_file, symbol):
    """Analyze order book snapshot for buy decision."""
    try:
        # Load order book
        order_book = load_order_book(json_file, symbol)
        if not order_book:
            return

        # Calculate mid-price
        mid_price = calculate_mid_price(order_book)
        if not mid_price:
            return

        logger.info(f"Mid-price for {symbol}: {mid_price:.2f} EUR")

        # Evaluate buy decision
        buy_metrics = evaluate_buy_decision(order_book, mid_price)
        if not buy_metrics:
            return

    except Exception as e:
        logger.error(f"Error analyzing buy decision: {e}")


def main():
    # Example inputs (replace with your actual file and symbol)
    json_file = "F:\Order_Book_Data/combined_snapshots_20250625_213341.json"
    symbol = "LMWR/EUR"

    logger.info(
        f"Starting buy decision analysis for {symbol} in {json_file} with cascade points in +5% range"
    )
    analyze_buy_decision(json_file, symbol)
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
