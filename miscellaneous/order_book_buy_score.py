#!/usr/bin/env python3
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('buy_decision_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_order_book(json_file, symbol):
    """Load order book snapshot for a symbol from the combined JSON file."""
    try:
        if not os.path.exists(json_file):
            logger.error(f"File {json_file} does not exist.")
            return None
        with open(json_file, 'r') as f:
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
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
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
        bid_liquidity = sum(
            calculate_liquidity_at_level(price, amount)
            for price, amount in order_book['bids']
            if float(price) >= mid_price * 0.99
        )
        ask_liquidity = sum(
            calculate_liquidity_at_level(price, amount)
            for price, amount in order_book['asks']
            if float(price) <= mid_price * 1.01
        )
        imbalance_ratio = bid_liquidity / ask_liquidity if ask_liquidity > 0 else 1.0
        return imbalance_ratio, bid_liquidity, ask_liquidity
    except Exception as e:
        logger.error(f"Error calculating imbalance: {e}")
        return 1.0, 0.0, 0.0

def detect_cascade_points(order_book, mid_price, liquidity_threshold=0.1, gap_threshold=0.001):
    """Detect potential cascade points on asks for buy decision."""
    try:
        ask_cascade_points = []
        total_liquidity = sum(
            calculate_liquidity_at_level(price, amount)
            for price, amount in order_book['asks']
            if float(price) <= mid_price * 1.01
        )
        prev_price = None
        for price, amount in order_book['asks']:
            price = float(price)
            if price > mid_price * 1.01:
                continue
            liquidity = calculate_liquidity_at_level(price, amount)
            liquidity_ratio = liquidity / total_liquidity if total_liquidity > 0 else 0
            if liquidity_ratio < liquidity_threshold:
                ask_cascade_points.append((price, amount, "Low liquidity"))
            if prev_price is not None:
                gap = (price - prev_price) / mid_price
                if gap > gap_threshold:
                    ask_cascade_points.append((price, amount, "Large gap"))
            prev_price = price
        return ask_cascade_points
    except Exception as e:
        logger.error(f"Error detecting cascade points: {e}")
        return []

def simulate_buy_impact(order_book, order_size):
    """Simulate price impact of a buy order."""
    try:
        remaining_size = order_size
        total_value = 0.0
        last_price = None
        cleared_levels = []
        for price, amount in order_book['asks']:
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
            logger.warning(f"Buy order of {order_size} not fully filled. Remaining: {remaining_size}")
            return None, cleared_levels
        return last_price, cleared_levels
    except Exception as e:
        logger.error(f"Error simulating buy impact: {e}")
        return None, []

def calculate_spread(order_book, mid_price):
    """Calculate bid-ask spread as percentage of mid-price."""
    try:
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        spread = (best_ask - best_bid) / mid_price
        return spread
    except Exception as e:
        logger.error(f"Error calculating spread: {e}")
        return None

def evaluate_buy_decision(order_book, mid_price):
    """Evaluate buy decision based on order book metrics."""
    try:
        # 1. Order book imbalance
        imbalance_ratio, bid_liquidity, ask_liquidity = calculate_order_book_imbalance(order_book, mid_price)
        imbalance_score = min(1.0, imbalance_ratio / 1.2)  # >1.2 is strong buying pressure
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f} (Bid: {bid_liquidity:.2f}, Ask: {ask_liquidity:.2f}), Score: {imbalance_score:.2f}")

        # 2. Cascade vulnerability
        ask_cascade_points = detect_cascade_points(order_book, mid_price)
        cascade_count = len(ask_cascade_points)
        cascade_score = min(1.0, cascade_count / 10)  # >10 vulnerable levels is high risk/reward
        logger.info(f"Ask cascade points: {cascade_count}, Score: {cascade_score:.2f}")

        # 3. Price impact
        order_size = 1.0  # Evaluate 1 ETH buy
        final_price, cleared_levels = simulate_buy_impact(order_book, order_size)
        price_impact = ((final_price - float(order_book['asks'][0][0])) / float(order_book['asks'][0][0]) * 100) if final_price else 0
        impact_score = max(0.0, 1.0 - price_impact / 1.0)  # <1% impact is good
        logger.info(f"Price impact for {order_size} ETH buy: {price_impact:.2f}%, Score: {impact_score:.2f}")

        # 4. Spread
        spread = calculate_spread(order_book, mid_price)
        spread_score = max(0.0, 1.0 - spread / 0.0005) if spread else 0.0  # <0.05% spread is good
        logger.info(f"Spread: {spread*100:.4f}%, Score: {spread_score:.2f}")

        # Weighted total score
        weights = {'imbalance': 0.4, 'cascade': 0.3, 'impact': 0.2, 'spread': 0.1}
        total_score = (
            weights['imbalance'] * imbalance_score +
            weights['cascade'] * cascade_score +
            weights['impact'] * impact_score +
            weights['spread'] * spread_score
        )
        recommendation = (
            "Strong Buy" if total_score > 0.7 else
            "Weak Buy" if total_score > 0.5 else
            "No Buy"
        )
        logger.info(f"Total score: {total_score:.2f}, Recommendation: {recommendation}")

        return {
            'imbalance_ratio': imbalance_ratio,
            'cascade_count': cascade_count,
            'price_impact': price_impact,
            'spread': spread,
            'total_score': total_score,
            'recommendation': recommendation,
            'ask_cascade_points': ask_cascade_points
        }
    except Exception as e:
        logger.error(f"Error evaluating buy decision: {e}")
        return None

def visualize_buy_decision(order_book, symbol, mid_price, buy_metrics):
    """Visualize order book with buy decision indicators."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Prepare cumulative volumes
        bid_prices = [float(p) for p, _ in order_book['bids']]
        bid_volumes = np.cumsum([float(a) for _, a in order_book['bids']])
        ask_prices = [float(p) for p, _ in order_book['asks']]
        ask_volumes = np.cumsum([float(a) for _, a in order_book['asks']])
        
        # Plot cumulative order book
        plt.step(bid_prices[::-1], bid_volumes[::-1], where='post', label='Bids', color='green')
        plt.step(ask_prices, ask_volumes, where='post', label='Asks', color='red')
        
        # Add mid-price and ±1% lines
        plt.axvline(mid_price, color='black', linestyle='--', label='Mid-Price')
        plt.axvline(mid_price * 0.99, color='gray', linestyle=':', label='±1% Range')
        plt.axvline(mid_price * 1.01, color='gray', linestyle=':')
        
        # Highlight ask cascade points
        for price, amount, reason in buy_metrics.get('ask_cascade_points', []):
            plt.plot(price, 0, marker='o', color='red', markersize=8)
            plt.annotate(
                f"{reason}\n{price:.2f}",
                (price, 0),
                xytext=(5, 5),
                textcoords='offset points',
                color='red',
                fontsize=8
            )
        
        # Add buy recommendation text
        plt.text(
            mid_price * 0.99, max(bid_volumes[-1], ask_volumes[-1]) * 0.9,
            f"Recommendation: {buy_metrics['recommendation']}\nScore: {buy_metrics['total_score']:.2f}",
            bbox=dict(facecolor='white', alpha=0.8),
            fontsize=10
        )
        
        plt.xlabel('Price (EUR)')
        plt.ylabel('Cumulative Volume (ETH)')
        plt.title(f'Buy Decision Analysis for {symbol}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_file = f'buy_decision_{symbol.replace("/", "-")}_{timestamp}.png'
        plt.savefig(output_file)
        logger.info(f"Saved visualization to {output_file}")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error visualizing buy decision: {e}")

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
        
        # Visualize results
        visualize_buy_decision(order_book, symbol, mid_price, buy_metrics)
        
    except Exception as e:
        logger.error(f"Error analyzing buy decision: {e}")

def main():
    # Example inputs (replace with your actual file and symbol)
    json_file = "F:\Order_Book_Data/combined_snapshots_20250625_174942.json"
    symbol = "ETH/EUR"
    
    logger.info(f"Starting buy decision analysis for {symbol} in {json_file}")
    analyze_buy_decision(json_file, symbol)
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()