from python_bitvavo_api.bitvavo import Bitvavo
from dotenv import load_dotenv
import os
from datetime import datetime

# Configuration
SYMBOL = "BTC-EUR"  # Trading pair (e.g., "BTC-EUR", "ETH-EUR")
CANDLE_LIMIT = 50  # Number of candles to fetch
ATR_PERIOD = 14  # Period for ATR calculation
INTERVAL = "1m"  # 1-minute candles

def calculate_atr(candles, period):
    """Calculate the Average True Range (ATR) for the given candles over the specified period."""
    if len(candles) < period + 1:
        raise Exception(f"Not enough candles ({len(candles)}) for ATR period ({period}).")

    # Calculate True Range for each candle
    true_ranges = []
    for i in range(1, len(candles)):
        high = float(candles[i][2])  # High price
        low = float(candles[i][3])   # Low price
        prev_close = float(candles[i-1][4])  # Previous close price

        # True Range = max(high - low, |high - prev_close|, |low - prev_close|)
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        true_ranges.append(tr)

    # Calculate ATR as the average of the last 'period' True Ranges
    if len(true_ranges) < period:
        raise Exception(f"Not enough True Ranges ({len(true_ranges)}) for ATR period ({period}).")
    
    atr = sum(true_ranges[-period:]) / period
    return atr

def fetch_candles_and_calculate_atr():
    try:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.getenv("BITVAVO_API_KEY")
        api_secret = os.getenv("BITVAVO_API_SECRET")

        if not api_key or not api_secret:
            print("Warning: API key or secret not found in .env file. Attempting to use public endpoint.")

        # Initialize Bitvavo client
        bitvavo = Bitvavo({
            "APIKEY": api_key or "",
            "APISECRET": api_secret or ""
        })

        # Validate inputs
        if CANDLE_LIMIT < ATR_PERIOD:
            raise Exception(f"Candle limit ({CANDLE_LIMIT}) must be at least as large as ATR period ({ATR_PERIOD}).")
        if INTERVAL not in ["1m", "5m", "15m", "30m", "1h", "4h", "6h", "8h", "12h", "1d"]:
            raise Exception(f"Invalid interval: {INTERVAL}. Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d")

        # Fetch the last 50 one-minute candles
        candles = bitvavo.candles(SYMBOL, INTERVAL, {"limit": CANDLE_LIMIT})

        if not candles:
            raise Exception(f"No candle data available for {SYMBOL}.")

        # Sort candles by timestamp (ascending) to ensure correct order
        candles.sort(key=lambda x: int(x[0]))

        # Display candle data
        print(f"\nLast {CANDLE_LIMIT} one-minute candles for {SYMBOL}:")
        print("=" * 60)
        print(f"{'Timestamp':<20} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<10}")
        print("-" * 60)
        for candle in candles:
            timestamp = int(candle[0]) / 1000
            formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{formatted_time:<20} {float(candle[1]):<10.2f} {float(candle[2]):<10.2f} "
                  f"{float(candle[3]):<10.2f} {float(candle[4]):<10.2f} {float(candle[5]):<10.8f}")

        # Calculate ATR for the latest 14 candles
        atr = calculate_atr(candles, ATR_PERIOD)
        print(f"\nATR ({ATR_PERIOD} periods) for {SYMBOL}: {atr:.2f} EUR")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    fetch_candles_and_calculate_atr()