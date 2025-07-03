from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os
import time
import ntplib
import logging
import math
import statistics
import json
import os.path
from time import sleep
import requests
import hmac
import hashlib
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Grid trading configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., BTC-EUR, ETH-EUR)
TOTAL_CAPITAL_QUOTE = 500  # Total capital in quote currency (e.g., EUR)
NUM_BUY_ORDERS = 5  # Number of buy orders
NUM_SELL_ORDERS = 5  # Number of sell orders
LOSS_THRESHOLD_PERCENT = 0.1  # Stop if net unrealized loss exceeds 10%
MAX_DRAWDOWN_PERCENT = 0.2  # Stop if total capital loss exceeds 20%
CHECK_INTERVAL_SECONDS = 60  # Check price every 60 seconds
TAKER_FEE_RATE = 0.0025  # Taker fee (0.25%)
MAKER_FEE_RATE = 0.0015  # Maker fee (0.15%)
MIN_ORDER_SIZE_QUOTE = 15  # Minimum order size in EUR
ATR_PERIODS = 14  # Periods for ATR calculation
ATR_MULTIPLIER = 2  # Grid range = ATR * multiplier
STATE_FILE = f"grid_state_{SYMBOL}.json"  # File to store grid state
MIN_LIQUIDITY_QUOTE = 1000  # Minimum order book liquidity (EUR) within grid range
ORDER_BOOK_DEPTH = 100  # Number of order book levels to analyze
MAX_RETRIES = 3  # Max retries for API calls
RETRY_DELAY_SECONDS = 1  # Initial delay for exponential backoff
API_TIMEOUT_SECONDS = 10  # Timeout for API requests
MAX_RUNTIME_MINUTES = 60 * 24  # Max runtime (1 day) to prevent infinite loops
RATE_LIMIT_THRESHOLD = 100  # Pause if remaining API calls drop below this
OPERATOR_ID = 1001  # Unique integer for trader or algorithm
ATR_INTERVAL = "1h"  # Interval for ATR candles

def sync_system_time():
    """Synchronize system time with an NTP server."""
    for attempt in range(MAX_RETRIES):
        try:
            client = ntplib.NTPClient()
            response = client.request('pool.ntp.org', version=3, timeout=API_TIMEOUT_SECONDS)
            time_diff = abs(response.tx_time - time.time())
            logging.info(f"Time sync check: Difference {time_diff:.2f} seconds")
            if time_diff > 2:
                logging.warning("System clock is off by more than 2 seconds. Sync recommended.")
            return
        except Exception as e:
            logging.error(f"NTP sync failed (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.error("NTP sync failed after max retries. Continuing.")

def load_credentials():
    """Load API key and secret from .env file."""
    load_dotenv()
    api_key = os.getenv('BITVAVO_API_KEY')
    api_secret = os.getenv('BITVAVO_API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("API key or secret not found in .env file.")
    return api_key, api_secret

def round_to_precision(value, precision):
    """Round a number to the specified number of significant digits."""
    if value == 0:
        return 0
    scale = precision - 1 - math.floor(math.log10(abs(value)))
    return round(value, scale if scale > 0 else 0)

def get_market_info(bitvavo, symbol):
    """Get market-specific precision and minimum amount."""
    try:
        markets = bitvavo.markets()
        for market in markets:
            if market['market'] == symbol:
                return {
                    'amount_precision': int(market.get('amountPrecision', 8)),
                    'price_precision': int(market.get('pricePrecision', 5)),
                    'min_amount': float(market.get('minAmount', 0.0))
                }
        return {'amount_precision': 8, 'price_precision': 5, 'min_amount': 0.0}
    except Exception as e:
        logging.error(f"Failed to fetch market info: {str(e)}. Using default precision.")
        return {'amount_precision': 8, 'price_precision': 5, 'min_amount': 0.0}

def save_state(initial_price, grid_range, buy_prices, sell_prices, buy_sizes, sell_sizes):
    """Save grid state to a file."""
    state = {
        'initial_price': initial_price,
        'grid_range': grid_range,
        'buy_prices': buy_prices,
        'sell_prices': sell_prices,
        'buy_sizes': buy_sizes,
        'sell_sizes': sell_sizes
    }
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
        logging.info(f"Saved grid state to {STATE_FILE}")
    except Exception as e:
        logging.error(f"Failed to save state: {str(e)}")

def load_state():
    """Load grid state from a file."""
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        required_keys = ['initial_price', 'grid_range', 'buy_prices', 'sell_prices', 'buy_sizes', 'sell_sizes']
        if not all(key in state for key in required_keys):
            logging.error("Invalid state file: missing required keys.")
            return None
        logging.info(f"Loaded grid state from {STATE_FILE}")
        return state
    except Exception as e:
        logging.error(f"Failed to load state: {str(e)}")
        return None

def get_market_price_and_book(bitvavo, symbol):
    """Fetch current market price, bid/ask, and order book snapshot."""
    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            ticker = bitvavo.tickerPrice({'market': symbol})
            book = bitvavo.tickerBook({'market': symbol})
            price = float(ticker.get('price', 0))
            bid = float(book.get('bid', 0))
            ask = float(book.get('ask', 0))
            if price == 0 or bid == 0 or ask == 0:
                raise ValueError("Invalid price or bid/ask data received.")

            # Fetch order book using bitvavo.book
            order_book = bitvavo.book(symbol, {'depth': ORDER_BOOK_DEPTH})
            bids = [[float(price), float(amount)] for price, amount in order_book.get('bids', [])]
            asks = [[float(price), float(amount)] for price, amount in order_book.get('asks', [])]

            if not bids or not asks:
                logging.warning("Empty order book received. Using single bid/ask level.")
                bids = [[bid, 0]]
                asks = [[ask, 0]]

            return price, bid, ask, bids, asks
        except Exception as e:
            logging.error(f"Failed to fetch market data or order book (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.warning("Failed to fetch market data after max retries. Using last known price or exiting.")
    raise ValueError("Failed to fetch market data after max retries.")

def get_balance(bitvavo, symbol):
    """Get available balance for base and quote currencies."""
    base_currency = symbol.split('-')[0]
    quote_currency = symbol.split('-')[1]
    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            balances = bitvavo.balance({})
            base_balance = 0
            quote_balance = 0
            for asset in balances:
                if asset['symbol'] == base_currency:
                    base_balance = float(asset.get('available', 0))
                if asset['symbol'] == quote_currency:
                    quote_balance = float(asset.get('available', 0))
            return base_balance, quote_balance
        except Exception as e:
            logging.error(f"Failed to fetch balance (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    raise ValueError("Failed to fetch balance after max retries.")

def calculate_atr(bitvavo, symbol, periods=14):
    """Calculate Average True Range (ATR) from historical data."""
    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            candles = bitvavo.candles(symbol, ATR_INTERVAL, {'limit': periods + 1})
            if len(candles) < periods + 1:
                raise ValueError(f"Not enough candles ({len(candles)}) for ATR period ({periods}).")

            # Sort candles by timestamp (ascending)
            candles.sort(key=lambda x: int(x[0]))

            # Calculate True Range for each candle
            true_ranges = []
            for i in range(1, len(candles)):
                high = float(candles[i][2])  # High price
                low = float(candles[i][3])   # Low price
                prev_close = float(candles[i-1][4])  # Previous close price
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)

            # Calculate ATR as the average of the last 'periods' True ranges
            if len(true_ranges) < periods:
                raise ValueError(f"Not enough True Ranges ({len(true_ranges)}) for ATR period ({periods}).")
            atr = sum(true_ranges[-periods:]) / periods
            return atr
        except Exception as e:
            logging.error(f"Failed to calculate ATR (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.warning("ATR calculation failed. Using 5% fallback range.")
    return 0

def calculate_grid_levels(current_price, atr, atr_multiplier, num_buy_orders, num_sell_orders):
    """Calculate price levels using ATR-based range."""
    try:
        range_amount = atr * atr_multiplier if atr > 0 else current_price * 0.05
        if range_amount <= 0:
            raise ValueError("Invalid grid range calculated.")
        lower_bound = current_price - range_amount
        upper_bound = current_price + range_amount
        buy_step = range_amount / max(num_buy_orders, 1) if num_buy_orders > 0 else 0
        sell_step = range_amount / max(num_sell_orders, 1) if num_sell_orders > 0 else 0

        buy_prices = [round(lower_bound + i * buy_step, 2) for i in range(num_buy_orders)] if num_buy_orders > 0 else []
        sell_prices = [round(current_price + i * sell_step, 2) for i in range(1, num_sell_orders + 1)] if num_sell_orders > 0 else []
        
        return buy_prices, sell_prices, lower_bound, upper_bound
    except Exception as e:
        logging.error(f"Error calculating grid levels: {str(e)}")
        raise

def adjust_grid_for_liquidity(buy_prices, sell_prices, bids, asks, current_price, grid_range):
    """Adjust grid prices to align with high-liquidity order book levels."""
    try:
        adjusted_buy_prices = []
        adjusted_sell_prices = []

        for price in buy_prices:
            closest_bid = min(bids, key=lambda x: abs(x[0] - price), default=[price, 0])[0]
            adjusted_price = round(min(closest_bid, price - 0.01), 2)
            adjusted_buy_prices.append(adjusted_price)

        for price in sell_prices:
            closest_ask = min(asks, key=lambda x: abs(x[0] - price), default=[price, 0])[0]
            adjusted_price = round(max(closest_ask, price + 0.01), 2)
            adjusted_sell_prices.append(adjusted_price)

        total_bid_volume = sum(amount * price for price, amount in bids if current_price - grid_range <= price <= current_price)
        total_ask_volume = sum(amount * price for price, amount in asks if current_price <= price <= current_price + grid_range)
        if total_bid_volume + total_ask_volume < MIN_LIQUIDITY_QUOTE:
            logging.warning(f"Insufficient liquidity ({total_bid_volume + total_ask_volume:.2f} EUR) in grid range. Using unadjusted prices.")
            return buy_prices, sell_prices
        return adjusted_buy_prices, adjusted_sell_prices
    except Exception as e:
        logging.error(f"Error adjusting grid for liquidity: {str(e)}. Using unadjusted prices.")
        return buy_prices, sell_prices

def calculate_order_sizes(bitvavo, symbol, total_capital_quote, num_buy_orders, num_sell_orders, bids, asks, buy_prices, sell_prices):
    """Calculate order sizes using geometric progression, liquidity, and available balance."""
    try:
        # Get available balances
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        base_balance, quote_balance = get_balance(bitvavo, symbol)
        
        total_orders = max(num_buy_orders + num_sell_orders, 1)
        base_size = total_capital_quote / total_orders
        if base_size < MIN_ORDER_SIZE_QUOTE:
            min_capital = MIN_ORDER_SIZE_QUOTE * total_orders
            raise ValueError(f"Order size {base_size:.2f} EUR is below minimum {MIN_ORDER_SIZE_QUOTE} EUR. Increase TOTAL_CAPITAL_QUOTE to at least {min_capital} EUR or reduce orders.")

        progression_factor = 1.2
        buy_sizes = []
        sell_sizes = []

        # Calculate initial order sizes with geometric progression and liquidity
        for i in range(num_buy_orders):
            size = base_size * (progression_factor ** (num_buy_orders - 1 - i))
            closest_bid = min(bids, key=lambda x: abs(x[0] - buy_prices[i]), default=[buy_prices[i], 0])[1]
            liquidity_factor = min(closest_bid * buy_prices[i] / base_size, 1.0) if closest_bid > 0 else 1.0
            buy_sizes.append(max(size * liquidity_factor, MIN_ORDER_SIZE_QUOTE))
        for i in range(num_sell_orders):
            size = base_size * (progression_factor ** i)
            closest_ask = min(asks, key=lambda x: abs(x[0] - sell_prices[i]), default=[sell_prices[i], 0])[1]
            liquidity_factor = min(closest_ask * sell_prices[i] / base_size, 1.0) if closest_ask > 0 else 1.0
            sell_sizes.append(max(size * liquidity_factor, MIN_ORDER_SIZE_QUOTE))

        # Calculate total required capital for buy and sell orders
        total_buy_size = sum(buy_sizes)
        total_sell_size = sum(sell_sizes)
        required_quote = total_buy_size
        required_base = sum(size / price for price, size in zip(sell_prices, sell_sizes) if size > 0)

        # Scale buy order sizes if quote balance is insufficient
        if required_quote > quote_balance:
            logging.warning(f"Insufficient quote balance ({quote_balance:.2f} {quote_currency}) for buy orders. Required: {required_quote:.2f}. Scaling buy order sizes.")
            scale_factor = quote_balance / required_quote if required_quote > 0 else 0
            buy_sizes = [max(size * scale_factor, MIN_ORDER_SIZE_QUOTE) for size in buy_sizes]
            total_buy_size = sum(buy_sizes)
            logging.info(f"Scaled buy order sizes: {[round(s, 2) for s in buy_sizes]} {quote_currency}")

        # Scale sell order sizes if base balance is insufficient
        if required_base > base_balance:
            logging.warning(f"Insufficient base balance ({base_balance:.8f} {base_currency}) for sell orders. Required: {required_base:.8f}. Scaling sell order sizes.")
            scale_factor = base_balance / required_base if required_base > 0 else 0
            sell_sizes = [max(size * scale_factor, MIN_ORDER_SIZE_QUOTE) for size in sell_sizes]
            total_sell_size = sum(sell_sizes)
            logging.info(f"Scaled sell order sizes: {[round(s, 2) for s in sell_sizes]} {quote_currency}")

        # Ensure total capital does not exceed total_capital_quote
        total_size = total_buy_size + total_sell_size
        if total_size > total_capital_quote:
            scale = total_capital_quote / total_size if total_size > 0 else 0
            buy_sizes = [size * scale for size in buy_sizes]
            sell_sizes = [size * scale for size in sell_sizes]
            logging.info(f"Adjusted order sizes to fit total capital: Buy sizes {[round(s, 2) for s in buy_sizes]}, Sell sizes {[round(s, 2) for s in sell_sizes]}")

        return buy_sizes, sell_sizes
    except Exception as e:
        logging.error(f"Error calculating order sizes: {str(e)}")
        raise

def recover_grid_state(bitvavo, symbol, total_capital_quote, num_buy_orders, num_sell_orders):
    """Recover grid state from open orders and trade history."""
    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            orders = bitvavo.ordersOpen({'market': symbol})
            if not orders:
                logging.info("No open orders found. Starting fresh grid.")
                return None, None, None, None, None, None

            current_price, _, _, bids, asks = get_market_price_and_book(bitvavo, symbol)
            atr = calculate_atr(bitvavo, symbol, ATR_PERIODS)
            grid_range = atr * ATR_MULTIPLIER if atr > 0 else current_price * 0.05

            state = load_state()
            if state:
                initial_price = state['initial_price']
                grid_range = state['grid_range']
                buy_prices = state['buy_prices']
                sell_prices = state['sell_prices']
                buy_sizes = state['buy_sizes']
                sell_sizes = state['sell_sizes']
            else:
                order_prices = [float(order['price']) for order in orders if order.get('status') == 'new']
                if order_prices:
                    initial_price = statistics.mean(order_prices)
                else:
                    api_key, api_secret = load_credentials()
                    filled_orders = try_manual_request(api_key, api_secret, symbol, [])
                    initial_price = statistics.mean([float(order['price']) for order in filled_orders if order.get('price')]) if filled_orders else current_price
                buy_prices, sell_prices, _, _ = calculate_grid_levels(initial_price, atr, ATR_MULTIPLIER, num_buy_orders, num_sell_orders)
                buy_sizes, sell_sizes = calculate_order_sizes(bitvavo, symbol, total_capital_quote, num_buy_orders, num_sell_orders, bids, asks, buy_prices, sell_prices)

            order_ids = [order['orderId'] for order in orders if order.get('status') == 'new']
            logging.info(f"Recovered {len(order_ids)} open orders.")
            return order_ids, initial_price, buy_prices, sell_prices, buy_sizes, sell_sizes
        except Exception as e:
            logging.error(f"Failed to recover grid state (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.error("Failed to recover grid state after max retries. Starting fresh.")
    return None, None, None, None, None, None

def place_grid_orders(bitvavo, symbol, buy_prices, sell_prices, buy_sizes, sell_sizes, bid, ask, base_balance):
    """Place buy and sell limit orders, preferring maker fees, with adjusted sizes."""
    order_ids = []
    base_currency = symbol.split('-')[0]
    market_info = get_market_info(bitvavo, symbol)
    amount_precision = market_info['amount_precision']
    price_precision = market_info['price_precision']
    min_amount = market_info['min_amount']

    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            
            # Place buy orders
            for price, size in zip(buy_prices, buy_sizes):
                amount = size / price
                amount = round(amount, amount_precision)
                if amount < min_amount:
                    logging.warning(f"Buy order amount {amount} {base_currency} at price {price} below minimum {min_amount}. Skipping.")
                    continue
                if size < MIN_ORDER_SIZE_QUOTE:
                    logging.warning(f"Buy order size {size:.2f} EUR at price {price} below minimum {MIN_ORDER_SIZE_QUOTE}. Skipping.")
                    continue
                maker_price = round_to_precision(min(price, bid - 0.01), price_precision)
                response = bitvavo.placeOrder(
                    market=symbol,
                    side='buy',
                    orderType='limit',
                    body={
                        'amount': str(amount),
                        'price': str(maker_price),
                        'operatorId': str(OPERATOR_ID)
                    }
                )
                if 'orderId' in response:
                    order_ids.append(response['orderId'])
                    fee = size * MAKER_FEE_RATE
                    logging.info(f"Placed buy order: {amount} {base_currency} @ {maker_price}, Est. fee: {fee:.2f} EUR")
                else:
                    logging.error(f"Failed to place buy order: {response}")

            # Place sell orders if sufficient base balance
            required_base = sum(size / price for price, size in zip(sell_prices, sell_sizes) if size > 0)
            if base_balance >= required_base:
                for price, size in zip(sell_prices, sell_sizes):
                    amount = size / price
                    amount = round(amount, amount_precision)
                    if amount < min_amount:
                        logging.warning(f"Sell order amount {amount} {base_currency} at price {price} below minimum {min_amount}. Skipping.")
                        continue
                    if size < MIN_ORDER_SIZE_QUOTE:
                        logging.warning(f"Sell order size {size:.2f} EUR at price {price} below minimum {MIN_ORDER_SIZE_QUOTE}. Skipping.")
                        continue
                    maker_price = round_to_precision(max(price, ask + 0.01), price_precision)
                    response = bitvavo.placeOrder(
                        market=symbol,
                        side='sell',
                        orderType='limit',
                        body={
                            'amount': str(amount),
                            'price': str(maker_price),
                            'operatorId': str(OPERATOR_ID)
                        }
                    )
                    if 'orderId' in response:
                        order_ids.append(response['orderId'])
                        fee = size * MAKER_FEE_RATE
                        logging.info(f"Placed sell order: {amount} {base_currency} @ {maker_price}, Est. fee: {fee:.2f} EUR")
                    else:
                        logging.error(f"Failed to place sell order: {response}")
            else:
                logging.warning(f"Insufficient {base_currency} ({base_balance:.8f}) for full sell orders. Adjusted sizes used.")

            return order_ids
        except Exception as e:
            logging.error(f"Error placing orders (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    raise ValueError("Failed to place orders after max retries.")

def cancel_all_orders(bitvavo, symbol, order_ids):
    """Cancel all open orders if order_ids is defined and not empty."""
    if not order_ids:
        logging.info("No orders to cancel.")
        return
    for order_id in order_ids:
        for attempt in range(MAX_RETRIES):
            try:
                if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                    logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                    sleep(60)
                response = bitvavo.cancelOrder(
                    market=symbol,
                    orderId=order_id,
                    operatorId=str(OPERATOR_ID)
                )
                if 'orderId' in response:
                    logging.info(f"Cancelled order: {response['orderId']}")
                else:
                    logging.error(f"Failed to cancel order {order_id}: {response}")
                break
            except Exception as e:
                logging.error(f"Failed to cancel order {order_id} (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    sleep(RETRY_DELAY_SECONDS * (2 ** attempt))

def calculate_profit_loss(bitvavo, symbol, order_ids, current_price, initial_price, initial_capital):
    """Calculate net profit/loss including fees and check drawdown."""
    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)

            # Load API credentials
            api_key, api_secret = load_credentials()

            # Fetch filled orders using manual request with pagination
            all_filled_orders = try_manual_request(api_key, api_secret, symbol, order_ids)

            # Fetch open orders
            open_orders = bitvavo.ordersOpen({'market': symbol})
            unrealized_loss = 0
            realized_profit = 0
            total_fees = 0
            total_invested = 0

            # Calculate unrealized loss from open orders
            for order in open_orders:
                if order['orderId'] in order_ids and order.get('status') == 'new':
                    order_price = float(order['price'])
                    amount = float(order['amount'])
                    order_value = amount * order_price
                    fee = order_value * MAKER_FEE_RATE
                    total_fees += fee
                    total_invested += order_value
                    if order['side'] == 'buy':
                        loss = (order_price - current_price) * amount
                    else:
                        loss = (current_price - order_price) * amount
                    unrealized_loss += loss

            # Calculate realized profit from filled orders
            for order in all_filled_orders:
                if order['orderId'] in order_ids:
                    trade_price = float(order.get('price', 0))
                    amount = float(order.get('filledAmount', 0))
                    fee = float(order.get('feePaid', 0))
                    total_fees += fee
                    if order['side'] == 'buy':
                        profit = (current_price - trade_price) * amount - fee
                    else:
                        profit = (trade_price - initial_price) * amount - fee
                    realized_profit += profit

            net_unrealized_loss_percent = abs(unrealized_loss) / total_invested if total_invested > 0 else 0
            total_pnl = realized_profit - unrealized_loss - total_fees
            drawdown_percent = -total_pnl / initial_capital if initial_capital > 0 else 0

            logging.info(f"Realized profit: {realized_profit:.2f} EUR, Unrealized loss: {unrealized_loss:.2f} EUR, Total fees: {total_fees:.2f} EUR, Drawdown: {drawdown_percent:.2%}")
            return net_unrealized_loss_percent, total_pnl, drawdown_percent, total_fees

        except Exception as e:
            logging.error(f"Error calculating profit/loss (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.error("Failed to calculate profit/loss after max retries. Returning zeros.")
    return 0, 0, 0, 0

def try_manual_request(api_key, api_secret, symbol, order_ids):
    """Fetch filled orders using manual API request with pagination."""
    all_filled_orders = []
    order_id_from = None
    limit = 1000  # Max orders per request

    while True:
        try:
            url = "https://api.bitvavo.com/v2/orders"
            timestamp = str(int(time.time() * 1000))
            query_params = f"market={symbol}&limit={limit}"
            if order_id_from:
                query_params += f"&orderIdFrom={order_id_from}"
            string_to_sign = f"{timestamp}GET/v2/orders?{query_params}"
            signature = hmac.new(
                api_secret.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            headers = {
                "Bitvavo-Access-Key": api_key,
                "Bitvavo-Access-Signature": signature,
                "Bitvavo-Access-Timestamp": timestamp,
                "Content-Type": "application/json"
            }

            response = requests.get(f"{url}?{query_params}", headers=headers)
            response.raise_for_status()
            orders = response.json()

            if isinstance(orders, dict) and "errorCode" in orders:
                raise Exception(f"Manual API error: {orders.get('errorCode', 'Unknown')}: {orders.get('error', 'No details provided')}")

            filled_orders = [order for order in orders if order.get("status") == "filled" and (not order_ids or order['orderId'] in order_ids)]
            all_filled_orders.extend(filled_orders)

            if len(orders) < limit:
                break  # No more orders to fetch
            order_id_from = orders[-1].get("orderId")  # Use last order ID for next batch

        except Exception as e:
            logging.error(f"Manual request failed: {str(e)}")
            raise

    if not all_filled_orders:
        logging.info(f"No filled orders found for {symbol}.")
    else:
        logging.info(f"Fetched {len(all_filled_orders)} filled orders for {symbol}.")
    return all_filled_orders

def rebalance_grid(bitvavo, symbol, order_ids, buy_prices, sell_prices, buy_sizes, sell_sizes, bid, ask, current_price, base_balance, bids, asks):
    """Rebalance grid by replacing executed orders."""
    market_info = get_market_info(bitvavo, symbol)
    amount_precision = market_info['amount_precision']
    price_precision = market_info['price_precision']
    min_amount = market_info['min_amount']

    for attempt in range(MAX_RETRIES):
        try:
            if bitvavo.getRemainingLimit() < RATE_LIMIT_THRESHOLD:
                logging.warning(f"API rate limit low ({bitvavo.getRemainingLimit()}). Pausing for 60 seconds.")
                sleep(60)
            orders = bitvavo.ordersOpen({'market': symbol})
            active_order_prices = [float(order['price']) for order in orders if order['orderId'] in order_ids and order['status'] == 'new']
            
            new_order_ids = []
            for price, size in zip(buy_prices, buy_sizes):
                if not any(abs(p - price) < 0.01 for p in active_order_prices):
                    amount = size / price
                    amount = round(amount, amount_precision)
                    if amount < min_amount:
                        logging.warning(f"Buy order amount {amount} {symbol.split('-')[0]} at price {price} below minimum {min_amount}. Skipping.")
                        continue
                    if size < MIN_ORDER_SIZE_QUOTE:
                        logging.warning(f"Buy order size {size:.2f} EUR at price {price} below minimum {MIN_ORDER_SIZE_QUOTE}. Skipping.")
                        continue
                    maker_price = round_to_precision(min(price, bid - 0.01), price_precision)
                    response = bitvavo.placeOrder(
                        market=symbol,
                        side='buy',
                        orderType='limit',
                        body={
                            'amount': str(amount),
                            'price': str(maker_price),
                            'operatorId': str(OPERATOR_ID)
                        }
                    )
                    if 'orderId' in response:
                        new_order_ids.append(response['orderId'])
                        logging.info(f"Rebalanced buy order: {amount} {symbol.split('-')[0]} @ {maker_price}")

            required_base = sum(size / price for price, size in zip(sell_prices, sell_sizes) if size > 0)
            if base_balance >= required_base:
                for price, size in zip(sell_prices, sell_sizes):
                    if not any(abs(p - price) < 0.01 for p in active_order_prices):
                        amount = size / price
                        amount = round(amount, amount_precision)
                        if amount < min_amount:
                            logging.warning(f"Sell order amount {amount} {symbol.split('-')[0]} at price {price} below minimum {min_amount}. Skipping.")
                            continue
                        if size < MIN_ORDER_SIZE_QUOTE:
                            logging.warning(f"Sell order size {size:.2f} EUR at price {price} below minimum {MIN_ORDER_SIZE_QUOTE}. Skipping.")
                            continue
                        maker_price = round_to_precision(max(price, ask + 0.01), price_precision)
                        response = bitvavo.placeOrder(
                            market=symbol,
                            side='sell',
                            orderType='limit',
                            body={
                                'amount': str(amount),
                                'price': str(maker_price),
                                'operatorId': str(OPERATOR_ID)
                            }
                        )
                        if 'orderId' in response:
                            new_order_ids.append(response['orderId'])
                            logging.info(f"Rebalanced sell order: {amount} {symbol.split('-')[0]} @ {maker_price}")
            else:
                logging.warning(f"Insufficient {symbol.split('-')[0]} ({base_balance:.8f}) for sell order rebalancing.")
            
            return new_order_ids
        except Exception as e:
            logging.error(f"Error rebalancing grid (attempt {attempt + 1}/{MAX_RETRIES}): {str(e)}")
            if attempt < MAX_RETRIES - 1:
                sleep(RETRY_DELAY_SECONDS * (2 ** attempt))
    logging.error("Failed to rebalance grid after max retries. Returning empty list.")
    return []

def grid_trading(symbol, total_capital_quote, num_buy_orders, num_sell_orders):
    """Main grid trading function with recovery and order book analysis."""
    order_ids = []
    start_time = time.time()
    try:
        # Initialize Bitvavo client
        api_key, api_secret = load_credentials()
        bitvavo = Bitvavo({
            'APIKEY': api_key,
            'APISECRET': api_secret,
            'DEBUGGING': True,
            'RESTURL': 'https://api.bitvavo.com/v2',
            'WSURL': 'wss://ws.bitvavo.com/v2/',
            'ACCESSWINDOW': 10000,
            'TIMEOUT': API_TIMEOUT_SECONDS
        })

        # Sync time
        sync_system_time()

        # Get market info
        market_info = get_market_info(bitvavo, symbol)

        # Attempt to recover state
        recovered = recover_grid_state(bitvavo, symbol, total_capital_quote, num_buy_orders, num_sell_orders)
        if recovered[0] is not None:
            order_ids, initial_price, buy_prices, sell_prices, buy_sizes, sell_sizes = recovered
            logging.info("Resuming grid from recovered state.")
        else:
            # Get initial data for fresh start
            current_price, bid, ask, bids, asks = get_market_price_and_book(bitvavo, symbol)
            base_balance, quote_balance = get_balance(bitvavo, symbol)
            atr = calculate_atr(bitvavo, symbol, ATR_PERIODS)
            grid_range = atr * ATR_MULTIPLIER if atr > 0 else current_price * 0.05
            logging.info(f"Current price: {current_price}, Bid: {bid}, Ask: {ask}, ATR: {atr}, Grid range: ±{grid_range:.2f}")

            # Calculate grid levels
            buy_prices, sell_prices, lower_bound, upper_bound = calculate_grid_levels(
                current_price, atr, ATR_MULTIPLIER, num_buy_orders, num_sell_orders
            )
            
            # Adjust grid for liquidity
            adjusted_buy_prices, adjusted_sell_prices = adjust_grid_for_liquidity(
                buy_prices, sell_prices, bids, asks, current_price, grid_range
            )
            buy_prices = [round_to_precision(p, market_info['price_precision']) for p in adjusted_buy_prices]
            sell_prices = [round_to_precision(p, market_info['price_precision']) for p in adjusted_sell_prices]

            # Calculate order sizes with liquidity and balance adjustment
            buy_sizes, sell_sizes = calculate_order_sizes(
                bitvavo, symbol, total_capital_quote, num_buy_orders, num_sell_orders, bids, asks, buy_prices, sell_prices
            )
            logging.info(f"Buy order sizes: {[round(s, 2) for s in buy_sizes]} EUR")
            logging.info(f"Sell order sizes: {[round(s, 2) for s in sell_sizes]} EUR")

            # Validate balance for buy orders
            required_quote = sum(buy_sizes)
            if quote_balance < required_quote:
                logging.warning(f"Insufficient {symbol.split('-')[1]} ({quote_balance:.2f}) for buy orders after scaling. Need {required_quote:.2f}.")
                raise ValueError(f"Insufficient {symbol.split('-')[1]} balance after scaling.")

            # Place initial grid orders
            order_ids = place_grid_orders(
                bitvavo, symbol, buy_prices, sell_prices, buy_sizes, sell_sizes, bid, ask, base_balance
            )
            initial_price = current_price

            # Save state
            save_state(initial_price, grid_range, buy_prices, sell_prices, buy_sizes, sell_sizes)

        # Monitor and manage grid
        initial_capital = total_capital_quote
        lower_bound = min(buy_prices) if buy_prices else initial_price * (1 - 0.05)
        upper_bound = max(sell_prices) if sell_prices else initial_price * (1 + 0.05)
        while True:
            if (time.time() - start_time) / 60 > MAX_RUNTIME_MINUTES:
                logging.warning("Max runtime exceeded. Cancelling orders and stopping.")
                cancel_all_orders(bitvavo, symbol, order_ids)
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                break

            time.sleep(CHECK_INTERVAL_SECONDS)
            current_price, bid, ask, bids, asks = get_market_price_and_book(bitvavo, symbol)
            base_balance, quote_balance = get_balance(bitvavo, symbol)
            logging.info(f"Current price: {current_price}, Base balance: {base_balance}, Quote balance: {quote_balance}")

            # Check price range
            if current_price < lower_bound or current_price > upper_bound:
                logging.warning("Price outside grid range. Cancelling orders.")
                cancel_all_orders(bitvavo, symbol, order_ids)
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                break

            # Check profit/loss and drawdown
            unrealized_loss_percent, total_pnl, drawdown_percent, total_fees = calculate_profit_loss(
                bitvavo, symbol, order_ids, current_price, initial_price, initial_capital
            )
            if unrealized_loss_percent > LOSS_THRESHOLD_PERCENT or drawdown_percent > MAX_DRAWDOWN_PERCENT:
                logging.warning(f"Stopping: Unrealized loss {unrealized_loss_percent:.2%} or Drawdown {drawdown_percent:.2%} exceeded")
                cancel_all_orders(bitvavo, symbol, order_ids)
                if os.path.exists(STATE_FILE):
                    os.remove(STATE_FILE)
                break

            # Rebalance grid
            new_order_ids = rebalance_grid(
                bitvavo, symbol, order_ids, buy_prices, sell_prices, buy_sizes, sell_sizes, bid, ask, current_price, base_balance, bids, asks
            )
            order_ids.extend(new_order_ids)

    except Exception as e:
        logging.error(f"Grid trading error: {str(e)}")
        cancel_all_orders(bitvavo, symbol, order_ids)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        raise
    finally:
        cancel_all_orders(bitvavo, symbol, order_ids)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        logging.info("Script terminated. All orders cancelled and state cleared.")

def main():
    grid_trading(SYMBOL, TOTAL_CAPITAL_QUOTE, NUM_BUY_ORDERS, NUM_SELL_ORDERS)

if __name__ == "__main__":
    main()