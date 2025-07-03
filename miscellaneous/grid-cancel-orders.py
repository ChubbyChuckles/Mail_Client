from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
OPERATOR_ID = 1001  # Unique integer for trader or algorithm (e.g., 1001 for a human trader)

def cancel_all_orders():
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        if not api_key or not api_secret:
            raise Exception("API key or secret not found in .env file.")

        # Initialize Bitvavo client
        bitvavo = Bitvavo({
            "APIKEY": api_key,
            "APISECRET": api_secret
        })

        # Fetch open orders for the specified symbol
        orders = bitvavo.ordersOpen({"market": SYMBOL})

        if not orders:
            print(f"No open orders found for {SYMBOL}.")
            return

        # Cancel each open order
        print(f"\nCanceling open orders for {SYMBOL}:")
        print("-" * 50)
        for order in orders:
            order_id = order.get("orderId", None)
            side = order.get("side", "N/A")
            amount = float(order.get("amount", 0))
            price = float(order.get("price", 0)) if order.get("price") else "N/A"

            if order_id:
                print(f"Canceling order ID: {order_id} ({side.capitalize()}, Amount: {amount} {SYMBOL.split('-')[0]}, Price: {price} EUR)")
                try:
                    # Cancel the order with operatorId
                    response = bitvavo.cancelOrder(
                        market=SYMBOL,
                        orderId=order_id,
                        operatorId=str(OPERATOR_ID)
                    )
                    if "orderId" in response:
                        print(f"Successfully canceled order ID: {response['orderId']}")
                    else:
                        print(f"Failed to cancel order ID: {order_id}")
                        print(response)
                except Exception as e:
                    print(f"Error canceling order ID: {order_id}: {str(e)}")
                print("-" * 50)
            else:
                print(f"Skipping invalid order (no orderId found): {order}")
                print("-" * 50)

        # Verify all orders are canceled
        remaining_orders = bitvavo.ordersOpen({"market": SYMBOL})
        if not remaining_orders:
            print(f"All open orders for {SYMBOL} have been canceled.")
        else:
            print(f"Warning: Some orders may not have been canceled. {len(remaining_orders)} open orders remain.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    cancel_all_orders()