import os
from datetime import datetime

from dotenv import load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
OPERATOR_ID = (
    1001  # Unique integer for trader or algorithm (e.g., 1001 for a human trader)
)


def list_open_orders():
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        if not api_key or not api_secret:
            raise Exception("API key or secret not found in .env file.")

        # Initialize Bitvavo client
        bitvavo = Bitvavo({"APIKEY": api_key, "APISECRET": api_secret})

        # Fetch open orders for the specified symbol
        orders = bitvavo.ordersOpen({"market": SYMBOL})

        if not orders:
            print(f"No open orders found for {SYMBOL}.")
            return

        # Display orders
        print(f"\nOpen orders for {SYMBOL}:")
        print("-" * 50)
        for order in orders:
            order_id = order.get("orderId", "N/A")
            side = order.get("side", "N/A")
            order_type = order.get("orderType", "N/A")
            amount = float(order.get("amount", 0))
            price = float(order.get("price", 0)) if order.get("price") else "N/A"
            created_timestamp = int(order.get("created", 0))
            created_time = (
                datetime.fromtimestamp(created_timestamp / 1000).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                if created_timestamp
                else "N/A"
            )

            print(f"Order ID: {order_id}")
            print(f"Side: {side.capitalize()}")
            print(f"Type: {order_type.capitalize()}")
            print(f"Amount: {amount} {SYMBOL.split('-')[0]}")
            print(
                f"Price: {price} EUR" if price != "N/A" else "Price: N/A (Market Order)"
            )
            print(f"Created: {created_time}")
            print("-" * 50)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    list_open_orders()
