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
        logging.FileHandler("order_book_cascade_analysis.log"),
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
        best_bid = float(order_book["bids"][0][0])
        best_ask = float(order_book["asks"][0][0])
        return (best_bid + best_ask) / 2
    except Exception as e:
        logger.error(f"Error calculating mid-price: {e}")
        return None


def calculate_liquidity_at_level(price, amount):
    """Calculate liquidity value (price × amount) at a price level."""
    return float(price) * float(amount)


def simulate_order_impact(order_book, order_size, side="buy"):
    """Simulate the price impact of a buy or sell order."""
    try:
        levels = order_book["asks"] if side == "buy" else order_book["bids"]
        remaining_size = order_size
        total_value = 0.0
        last_price = None
        cleared_levels = []

        for price, amount in levels:
            price = float(price)
            amount = float(amount)
            if remaining_size <= 0:
                break
            filled_amount = min(remaining_size, amount)
            total_value += filled_amount * price
            remaining_size -= filled_amount
            cleared_levels.append((price, filled_amount))
            last_price = price

        if remaining_size > 0:
            logger.warning(
                f"{side.capitalize()} order of {order_size} not fully filled. Remaining: {remaining_size}"
            )
            return None, cleared_levels

        final_price = last_price
        return final_price, cleared_levels
    except Exception as e:
        logger.error(f"Error simulating {side} order impact: {e}")
        return None, []


def detect_cascade_points(
    order_book, mid_price, liquidity_threshold=0.1, gap_threshold=0.001
):
    """Detect potential cascade points based on low liquidity or large price gaps."""
    cascade_points = {"bids": [], "asks": []}
    total_liquidity = 0.0
    price_range = {"bids": mid_price * 0.99, "asks": mid_price * 1.01}

    # Calculate total liquidity within ±1% of mid-price
    for side in ["bids", "asks"]:
        for price, amount in order_book[side]:
            price = float(price)
            if (side == "bids" and price >= price_range["bids"]) or (
                side == "asks" and price <= price_range["asks"]
            ):
                total_liquidity += calculate_liquidity_at_level(price, amount)

    # Check for low-liquidity levels or large gaps
    for side in ["bids", "asks"]:
        prev_price = None
        for price, amount in order_book[side]:
            price = float(price)
            if (side == "bids" and price < price_range["bids"]) or (
                side == "asks" and price > price_range["asks"]
            ):
                continue
            liquidity = calculate_liquidity_at_level(price, amount)
            liquidity_ratio = liquidity / total_liquidity if total_liquidity > 0 else 0
            reason = None

            # Check for low liquidity
            if liquidity_ratio < liquidity_threshold:
                reason = f"Low liquidity ({liquidity_ratio:.2%} of total)"

            # Check for large price gap
            if prev_price is not None:
                gap = abs(price - prev_price) / mid_price
                if gap > gap_threshold:
                    reason = f"Large gap ({gap:.2%} of mid-price)"

            if reason:
                cascade_points[side].append((price, amount, reason))

            prev_price = price

    return cascade_points


def visualize_order_book(
    order_book, symbol, mid_price, cascade_points, order_sizes=[0.1, 1, 5]
):
    """Visualize the order book with potential cascade points and order impact."""
    try:
        plt.figure(figsize=(12, 6))

        # Prepare cumulative volumes
        bid_prices = [float(p) for p, _ in order_book["bids"]]
        bid_volumes = np.cumsum([float(a) for _, a in order_book["bids"]])
        ask_prices = [float(p) for p, _ in order_book["asks"]]
        ask_volumes = np.cumsum([float(a) for _, a in order_book["asks"]])

        # Plot cumulative order book
        plt.step(
            bid_prices[::-1],
            bid_volumes[::-1],
            where="post",
            label="Bids",
            color="green",
        )
        plt.step(ask_prices, ask_volumes, where="post", label="Asks", color="red")

        # Add mid-price and ±1% lines
        plt.axvline(mid_price, color="black", linestyle="--", label="Mid-Price")
        plt.axvline(mid_price * 0.99, color="gray", linestyle=":", label="±1% Range")
        plt.axvline(mid_price * 1.01, color="gray", linestyle=":")

        # Annotate cascade points
        for side, points in cascade_points.items():
            color = "green" if side == "bids" else "red"
            for price, amount, reason in points:
                plt.plot(price, 0, marker="o", color=color, markersize=8)
                plt.annotate(
                    f"{reason}\n{price:.2f}",
                    (price, 0),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color=color,
                    fontsize=8,
                )

        # Simulate order impact for different sizes
        for order_size in order_sizes:
            for side in ["buy", "sell"]:
                final_price, cleared_levels = simulate_order_impact(
                    order_book, order_size, side
                )
                if final_price:
                    color = "blue" if side == "buy" else "purple"
                    plt.axvline(
                        final_price,
                        color=color,
                        linestyle="--",
                        alpha=0.5,
                        label=f"{side.capitalize()} {order_size} Impact",
                    )

        plt.xlabel("Price (EUR)")
        plt.ylabel("Cumulative Volume")
        plt.title(f"Order Book Analysis for {symbol} - Potential Cascading Effects")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = f'order_book_cascade_{symbol.replace("/", "-")}_{timestamp}.png'
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
        plt.show()

    except Exception as e:
        logger.error(f"Error visualizing order book: {e}")


def analyze_order_book(json_file, symbol):
    """Analyze order book snapshot for cascading effects and visualize results."""
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

        # Detect cascade points
        cascade_points = detect_cascade_points(
            order_book, mid_price, liquidity_threshold=0.1, gap_threshold=0.001
        )

        # Log cascade points
        for side, points in cascade_points.items():
            if points:
                logger.info(f"Potential cascade points for {side}:")
                for price, amount, reason in points:
                    logger.info(
                        f"  Price: {price:.2f}, Amount: {amount:.4f}, Reason: {reason}"
                    )
            else:
                logger.info(f"No potential cascade points detected for {side}")

        # Simulate order impact
        order_sizes = [0.1, 1, 5]
        for side in ["buy", "sell"]:
            logger.info(f"\nSimulating {side} order impact:")
            for order_size in order_sizes:
                final_price, cleared_levels = simulate_order_impact(
                    order_book, order_size, side
                )
                if final_price:
                    initial_price = (
                        float(order_book["asks"][0][0])
                        if side == "buy"
                        else float(order_book["bids"][0][0])
                    )
                    price_impact = (final_price - initial_price) / initial_price * 100
                    logger.info(
                        f"  Order size: {order_size}, Final price: {final_price:.2f}, "
                        f"Price impact: {price_impact:.2%}, Cleared levels: {len(cleared_levels)}"
                    )

        # Visualize results
        visualize_order_book(order_book, symbol, mid_price, cascade_points, order_sizes)

    except Exception as e:
        logger.error(f"Error analyzing order book: {e}")


def main():
    # Example inputs (replace with your actual file and symbol)
    json_file = "F:\Order_Book_Data/combined_snapshots_20250625_174942.json"
    symbol = "ETH/EUR"

    logger.info(f"Starting order book cascade analysis for {symbol} in {json_file}")
    analyze_order_book(json_file, symbol)
    logger.info("Analysis completed")


if __name__ == "__main__":
    main()
