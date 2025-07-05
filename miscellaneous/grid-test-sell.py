import os

from dotenv import load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
PERCENTAGE_ABOVE = 5  # Percentage above current market price (e.g., 5 for 5%)
AMOUNT_EUR = 6  # Amount to sell in EUR
OPERATOR_ID = (
    1001  # Unique integer for trader or algorithm (e.g., 1001 for a human trader)
)


def get_current_price(bitvavo, symbol):
    """Fetch the current market price for the given symbol."""
    try:
        ticker = bitvavo.tickerPrice({"market": symbol})
        if ticker and "price" in ticker:
            return float(ticker["price"])
        else:
            raise Exception("Unable to fetch current price.")
    except Exception as e:
        raise Exception(f"Error fetching price: {str(e)}")


def round_to_precision(value, precision):
    """Round a number to the specified number of significant digits."""
    if value == 0:
        return 0
    from math import floor, log10

    scale = precision - 1 - floor(log10(abs(value)))
    return round(value, scale if scale > 0 else 0)


def place_limit_sell_order():
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        if not api_key or not api_secret:
            raise Exception("API key or secret not found in .env file.")

        # Initialize Bitvavo client
        bitvavo = Bitvavo({"APIKEY": api_key, "APISECRET": api_secret})

        # Get current market price
        current_price = get_current_price(bitvavo, SYMBOL)
        print(f"Current market price for {SYMBOL}: {current_price} EUR")

        # Get market info for precision
        markets = bitvavo.markets()
        amount_precision = 8  # Default amount precision
        price_precision = 5  # Default price precision
        min_amount = 0.0  # Default minimum amount
        for market in markets:
            if market["market"] == SYMBOL:
                amount_precision = int(market.get("amountPrecision", 8))
                price_precision = int(market.get("pricePrecision", 5))
                min_amount = float(market.get("minAmount", 0.0))
                break

        # Calculate limit price (percentage above current price)
        limit_price = current_price * (1 + PERCENTAGE_ABOVE / 100)
        # Round to the market's price precision (significant digits)
        limit_price = round_to_precision(limit_price, price_precision)
        print(
            f"Rounded limit price to {price_precision} significant digits: {limit_price} EUR"
        )

        # Calculate amount in base currency (e.g., BTC)
        amount_base = AMOUNT_EUR / limit_price
        # Round amount to the required precision
        amount_base = round(amount_base, amount_precision)

        # Check if amount meets minimum requirement
        if amount_base < min_amount:
            raise Exception(
                f"Calculated amount {amount_base} {SYMBOL.split('-')[0]} is below minimum amount {min_amount} {SYMBOL.split('-')[0]}"
            )

        # Check balance
        balance = bitvavo.balance({"symbol": SYMBOL.split("-")[0]})
        available_base = 0.0
        if balance:
            available_base = float(balance[0].get("available", 0))
        if available_base < amount_base:
            raise Exception(
                f"Insufficient balance: {available_base} {SYMBOL.split('-')[0]} available, {amount_base} required"
            )

        print(
            f"Placing limit sell order at {limit_price} EUR for {amount_base} {SYMBOL.split('-')[0]}"
        )

        # Place a limit sell order
        response = bitvavo.placeOrder(
            market=SYMBOL,
            side="sell",
            orderType="limit",
            body={
                "amount": str(amount_base),
                "price": str(limit_price),
                "operatorId": str(OPERATOR_ID),  # Include operatorId
            },
        )

        # Check the response
        if "orderId" in response:
            print(f"Limit sell order placed successfully!")
            print(f"Order ID: {response['orderId']}")
            print(f"Symbol: {response['market']}")
            print(f"Amount: {response['amount']} {SYMBOL.split('-')[0]}")
            print(f"Price: {response['price']} EUR")
        else:
            print("Failed to place order.")
            print(response)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    place_limit_sell_order()
