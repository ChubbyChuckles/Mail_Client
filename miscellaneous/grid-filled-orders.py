from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import requests
import hmac
import hashlib
import json

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
LIMIT = 1000  # Max orders per request (API limit)

def fetch_filled_orders():
    try:
        # Load environment variables from .env file
        if not os.path.exists(".env"):
            raise Exception(".env file not found.")
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        # Validate API credentials
        if not api_key or not api_secret:
            raise Exception("API key or secret not found in .env file.")
        if len(api_key.strip()) == 0 or len(api_secret.strip()) == 0:
            raise Exception("API key or secret is empty or contains only whitespace.")
        api_key = api_key.strip()
        api_secret = api_secret.strip()

        # Initialize Bitvavo client
        bitvavo = Bitvavo({
            'APIKEY': api_key,
            'APISECRET': api_secret
        })

        # Test public endpoint to verify library
        try:
            markets = bitvavo.markets()
            if not any(market.get("market") == SYMBOL for market in markets):
                raise Exception(f"Invalid market: {SYMBOL}")
        except Exception as e:
            print(f"Public endpoint test failed: {str(e)}")
            raise Exception("Failed to verify market. Check library installation or network.")

        # Initialize variables for pagination
        all_filled_orders = []
        order_id_from = None

        print(f"Fetching filled orders for {SYMBOL}...")

        while True:
            # Prepare request parameters
            params = {"market": SYMBOL, "limit": LIMIT}
            if order_id_from:
                params["orderIdFrom"] = order_id_from

            # Fetch orders
            orders = bitvavo.getOrders(params)

            # Check for error response
            if isinstance(orders, dict) and "errorCode" in orders:
                raise Exception(f"API error: {orders.get('errorCode', 'Unknown')}: {orders.get('error', 'No details provided')}")

            # Validate response
            if not isinstance(orders, list):
                print(f"Unexpected response format: {orders}")
                raise Exception("API response is not a list.")

            # Filter filled orders
            filled_orders = [order for order in orders if isinstance(order, dict) and order.get("status") == "filled"]

            if not filled_orders:
                print("No more filled orders found in this batch.")
                break

            all_filled_orders.extend(filled_orders)

            # Check for pagination
            if len(orders) < LIMIT:
                break  # No more orders to fetch
            order_id_from = orders[-1].get("orderId")  # Use last order ID for next batch

        if not all_filled_orders:
            print(f"No filled orders found for {SYMBOL}.")
            return

        # Display filled orders
        print(f"\nFilled orders for {SYMBOL}:")
        print("=" * 60)
        for order in all_filled_orders:
            order_id = order.get("orderId", "N/A")
            side = order.get("side", "N/A")
            order_type = order.get("orderType", "N/A")
            amount = float(order.get("filledAmount", 0))
            price = float(order.get("price", 0)) if order.get("price") else "N/A"
            filled_timestamp = int(order.get("printing", 0))
            filled_time = (datetime.fromtimestamp(filled_timestamp / 1000)
                          .strftime("%Y-%m-%d %H:%M:%S") if filled_timestamp else "N/A")
            fee = float(order.get("feePaid", 0))
            fee_currency = order.get("feeCurrency", "N/A")

            print(f"Order ID: {order_id}")
            print(f"Side: {side.capitalize()}")
            print(f"Type: {order_type.capitalize()}")
            print(f"Amount: {amount:.8f} {SYMBOL.split('-')[0]}")
            print(f"Price: {price:.2f} EUR" if price != "N/A" else "Price: N/A (Market Order)")
            print(f"Fee: {fee:.8f} {fee_currency}")
            print(f"Filled Time: {filled_time}")
            print("-" * 60)

        print(f"Total filled orders: {len(all_filled_orders)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if "309" in str(e) or "403" in str(e):
            print("Signature or permission error detected. Attempting manual request...")
            try_manual_request(api_key, api_secret)
        else:
            print("Debugging tips:")
            print("- Verify .env file exists and contains valid BITVAVO_API_KEY and BITVAVO_API_SECRET.")
            print("- Ensure API key is activated via email and has 'View' or 'Trade' permissions.")
            print("- Synchronize system clock (e.g., `sudo ntpdate pool.ntp.org` on Linux/Mac, or 'Sync now' on Windows).")
            print("- Update python-bitvavo-api: `pip install --upgrade python-bitvavo-api`")
            print("- Check Bitvavo API status: https://status.bitvavo.com/")
            print("- Test with a public endpoint (e.g., bitvavo.markets()) to rule out credential issues.")

def try_manual_request(api_key, api_secret):
    try:
        print("Attempting manual API request to fetch filled orders...")
        url = "https://api.bitvavo.com/v2/orders"
        timestamp = str(int(time.time() * 1000))
        query_params = f"market={SYMBOL}&limit={LIMIT}"
        # Correct signature format per Bitvavo API documentation
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

        filled_orders = [order for order in orders if order.get("status") == "filled"]

        if not filled_orders:
            print("No filled orders found in manual request.")
            return

        print(f"\nFilled orders for {SYMBOL} (manual request):")
        print("=" * 60)
        for order in filled_orders:
            order_id = order.get("orderId", "N/A")
            side = order.get("side", "N/A")
            order_type = order.get("orderType", "N/A")
            amount = float(order.get("filledAmount", 0))
            price = float(order.get("price", 0)) if order.get("price") else "N/A"
            filled_timestamp = int(order.get("updated", 0))
            filled_time = (datetime.fromtimestamp(filled_timestamp / 1000)
                          .strftime("%Y-%m-%d %H:%M:%S") if filled_timestamp else "N/A")
            fee = float(order.get("feePaid", 0))
            fee_currency = order.get("feeCurrency", "N/A")

            print(f"Order ID: {order_id}")
            print(f"Side: {side.capitalize()}")
            print(f"Type: {order_type.capitalize()}")
            print(f"Amount: {amount:.8f} {SYMBOL.split('-')[0]}")
            print(f"Price: {price:.2f} EUR" if price != "N/A" else "Price: N/A (Market Order)")
            print(f"Fee: {fee:.8f} {fee_currency}")
            print(f"Filled Time: {filled_time}")
            print("-" * 60)

        print(f"Total filled orders: {len(filled_orders)}")

    except Exception as e:
        print(f"Manual request failed: {str(e)}")
        print("Manual request debugging tips:")
        print("- Verify API key and secret in .env (no typos, spaces, or quotes).")
        print("- Ensure API key has 'View' or 'Trade' permissions in Bitvavo settings.")
        print("- Check system clock synchronization (compare with https://time.is/).")
        print("- Verify no IP restrictions are set on the API key.")
        print("- Check Bitvavo API status: https://status.bitvavo.com/")
        print("- Contact Bitvavo support if the issue persists.")

if __name__ == "__main__":
    fetch_filled_orders()