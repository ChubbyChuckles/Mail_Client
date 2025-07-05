import os

from dotenv import load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
DEPTH = 100  # Number of bids and asks to fetch (e.g., 10 for 10 bids and 10 asks)


def fetch_order_book():
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        if not api_key or not api_secret:
            print(
                "Warning: API key or secret not found in .env file. Attempting to use public endpoint."
            )

        # Initialize Bitvavo client
        bitvavo = Bitvavo({"APIKEY": api_key or "", "APISECRET": api_secret or ""})

        # Validate depth
        if not isinstance(DEPTH, int) or DEPTH <= 0:
            raise Exception("Depth must be a positive integer.")

        # Fetch order book snapshot
        order_book = bitvavo.book(SYMBOL, {"depth": DEPTH})

        if not order_book:
            raise Exception(f"Failed to fetch order book for {SYMBOL}.")

        # Extract bids and asks
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])

        if not bids and not asks:
            print(f"No order book data available for {SYMBOL}.")
            return

        # Display order book
        print(f"\nOrder Book Snapshot for {SYMBOL} (Depth: {DEPTH})")
        print("=" * 50)
        print("Bids (Buy Orders):")
        print("-" * 50)
        if bids:
            print(f"{'Price (EUR)':<15} {'Amount (Base)':<15}")
            for bid in bids:
                price = float(bid[0])
                amount = float(bid[1])
                print(f"{price:<15.2f} {amount:<15.8f} {SYMBOL.split('-')[0]}")
        else:
            print("No bids available.")

        print("\nAsks (Sell Orders):")
        print("-" * 50)
        if asks:
            print(f"{'Price (EUR)':<15} {'Amount (Base)':<15}")
            for ask in asks:
                price = float(ask[0])
                amount = float(ask[1])
                print(f"{price:<15.2f} {amount:<15.8f} {SYMBOL.split('-')[0]}")
        else:
            print("No asks available.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    fetch_order_book()
