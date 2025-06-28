# config.py
import logging
import os
import sys
import io
from datetime import datetime
from dotenv import load_dotenv

# Set session start time for log file naming
SESSION_START_TIME = datetime.utcnow()

# Set up logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

log_dir = "trading_logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(
    log_dir, f"trading_{SESSION_START_TIME.strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class EvaluationLogHandler(logging.Handler):
    def emit(self, record):
        if record.levelno == logging.INFO and "Evaluation Decision" in record.msg:
            with open(log_filename, "a", encoding="utf-8") as f:
                formatted_message = self.format(record)
                f.write(f"{formatted_message}\n")

evaluation_handler = EvaluationLogHandler()
evaluation_handler.setFormatter(log_formatter)
logger.addHandler(evaluation_handler)

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    if sys.stdout.fileno():
        sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
except (AttributeError, OSError, io.UnsupportedOperation):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.reload()
            self._initialized = True

    def parse_float_env(self, var_name, default):
        value = os.getenv(var_name, str(default))
        try:
            return float(value.split("#")[0].strip())
        except ValueError as e:
            logger.error(f"Invalid value for {var_name}: {value}. Using default: {default}")
            return float(default)

    def reload(self):
        load_dotenv(override=True)
        self.API_KEY = os.getenv("BITVAVO_API_KEY")
        self.API_SECRET = os.getenv("BITVAVO_API_SECRET")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
        self.CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", 10))
        self.RATE_LIMIT_WEIGHT = int(os.getenv("RATE_LIMIT_WEIGHT", 1000))
        self.CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", 10))
        self.CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "1m")
        self.RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "data_1m_pq_alot")
        self.PARQUET_FILENAME = os.getenv("PARQUET_FILENAME", "bitvavo_1min_candles_eur.parquet")
        self.PRICE_INCREASE_THRESHOLD = float(os.getenv("PRICE_INCREASE_THRESHOLD", 1.0))
        self.MIN_VOLUME_EUR = float(os.getenv("MIN_VOLUME_EUR", 10000))
        self.VOLUME_DROP_THRESHOLD = float(os.getenv("VOLUME_DROP_THRESHOLD", 0.5))
        self.PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_VALUE", 10000))
        self.ALLOCATION_PER_TRADE = float(os.getenv("ALLOCATION_PER_TRADE", 0.1))
        self.BUY_FEE = float(os.getenv("BUY_FEE", 0.0015))
        self.SELL_FEE = float(os.getenv("SELL_FEE", 0.0025))
        self.TRAILING_STOP_FACTOR = float(os.getenv("TRAILING_STOP_FACTOR", 1.0))
        self.TRAILING_STOP_FACTOR_EARLY = float(os.getenv("TRAILING_STOP_FACTOR_EARLY", 1.5))
        self.ADJUSTED_PROFIT_TARGET = float(os.getenv("ADJUSTED_PROFIT_TARGET", 0.015))
        self.PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", 0.05))
        self.MIN_HOLDING_MINUTES = float(os.getenv("MIN_HOLDING_MINUTES", 5))
        self.TIME_STOP_MINUTES = int(os.getenv("TIME_STOP_MINUTES", 90))
        self.CAT_LOSS_THRESHOLD = float(os.getenv("CAT_LOSS_THRESHOLD", -0.08))
        self.CAT_LOSS_ATR_MULTIPLIER = float(os.getenv("CAT_LOSS_ATR_MULTIPLIER", 2.0))
        self.MOMENTUM_CONFIRM_MINUTES = int(os.getenv("MOMENTUM_CONFIRM_MINUTES", 3))
        self.MOMENTUM_THRESHOLD = float(os.getenv("MOMENTUM_THRESHOLD", -0.25))
        self.PORTFOLIO_FILE = os.getenv("PORTFOLIO_FILE", "portfolio.json")
        self.LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", 60))
        self.MAX_ACTIVE_ASSETS = int(os.getenv("MAX_ACTIVE_ASSETS", 7))
        self.ASSET_THRESHOLD = int(self.MAX_ACTIVE_ASSETS * 0.6)
        self.INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 20))
        self.PROFIT_TARGET_MULTIPLIER = float(os.getenv("PROFIT_TARGET_MULTIPLIER", 2.0))
        self.BUY_TRADES_CSV = os.getenv("BUY_TRADES_CSV", "buy_trades.csv")
        self.FINISHED_TRADES_CSV = os.getenv("FINISHED_TRADES_CSV", "finished_trades.csv")
        self.ORDER_BOOK_METRICS_CSV = os.getenv("ORDER_BOOK_METRICS_CSV", "order_book_metrics.csv")
        self.AMOUNT_QUOTE = self.parse_float_env("AMOUNT_QUOTE", 5.5)
        self.PRICE_RANGE_PERCENT = self.parse_float_env("PRICE_RANGE_PERCENT", 10.0)
        self.MAX_SLIPPAGE_BUY = self.parse_float_env("MAX_SLIPPAGE_BUY", 0.05)
        self.MAX_SLIPPAGE_SELL = self.parse_float_env("MAX_SLIPPAGE_SELL", -0.05)
        self.MIN_TOTAL_SCORE = self.parse_float_env("MIN_TOTAL_SCORE", 0.7)
        self.RSI_OVERSOLD_THRESHOLD=int(os.getenv("RSI_OVERSOLD_THRESHOLD", 30))
        self.RSI_OVERSOLD_PERIOD=int(os.getenv("RSI_OVERSOLD_PERIOD", 14))
        self.RSI_OVERBOUGHT_THRESHOLD=int(os.getenv("RSI_OVERBOUGHT_THRESHOLD", 70))
        self.RSI_OVERBOUGHT_PERIOD=int(os.getenv("RSI_OVERBOUGHT_PERIOD", 14))
        self.BOLLINGER_LOWER_PERIOD=int(os.getenv("BOLLINGER_LOWER_PERIOD", 20))
        self.BOLLINGER_LOWER_STD_DEV=self.parse_float_env("BOLLINGER_LOWER_STD_DEV", 2.0)
        self.BOLLINGER_UPPER_PERIOD=int(os.getenv("BOLLINGER_UPPER_PERIOD", 20))
        self.BOLLINGER_UPPER_STD_DEV=self.parse_float_env("BOLLINGER_UPPER_STD_DEV", 2.0)
        self.MACD_BULLISH_FAST_PERIOD=int(os.getenv("MACD_BULLISH_FAST_PERIOD", 12))
        self.MACD_BULLISH_SLOW_PERIOD=int(os.getenv("MACD_BULLISH_SLOW_PERIOD", 26))
        self.MACD_BULLISH_SIGNAL_PERIOD=int(os.getenv("MACD_BULLISH_SIGNAL_PERIOD", 9))
        self.MACD_BEARISH_FAST_PERIOD=int(os.getenv("MACD_BEARISH_FAST_PERIOD", 12))
        self.MACD_BEARISH_SLOW_PERIOD=int(os.getenv("MACD_BEARISH_SLOW_PERIOD", 26))
        self.MACD_BEARISH_SIGNAL_PERIOD=int(os.getenv("MACD_BEARISH_SIGNAL_PERIOD", 9))
        self.GRID_ENABLED=bool(os.getenv("GRID_ENABLED", False))
        self.GRID_RANGE=self.parse_float_env("GRID_RANGE", 0.05)
        self.GRID_STEP=self.parse_float_env("GRID_STEP", 0.01)
        self.GRID_MAX_ORDERS=int(os.getenv("GRID_MAX_ORDERS", 4)) # Maximum number of grid orders
        self.GRID_BB_PERIOD=int(os.getenv("GRID_BB_PERIOD", 20)) # Bollinger Bands period for grid trading
        self.GRID_BB_STD_DEV=self.parse_float_env("GRID_BB_STD_DEV", 1.0) # Bollinger Bands standard deviation for grid trading
        self.DCA_ENABLED=bool(os.getenv("DCA_ENABLED", False))
        self.DCA_DROP_THRESHOLD=self.parse_float_env("DCA_DROP_THRESHOLD", 0.05) # Percentage drop to trigger DCA
        self.DCA_ALLOCATION=self.parse_float_env("DCA_ALLOCATION", 0.02) # Percentage of portfolio to allocate for DCA
        self.VOLATILITY_SIZING_ENABLED=bool(os.getenv("VOLATILITY_SIZING_ENABLED", False))
        self.VOLATILITY_ATR_PERIOD=int(os.getenv("VOLATILITY_ATR_PERIOD", 14)) # ATR period for volatility sizing
        self.VOLATILITY_MAX_ALLOCATION_MULTIPLIER=self.parse_float_env("VOLATILITY_MAX_ALLOCATION_MULTIPLIER", 1.5) # Multiplier for ATR to
        self.VOLATILITY_MIN_ALLOCATION_MULTIPLIER=self.parse_float_env("VOLATILITY_MIN_ALLOCATION_MULTIPLIER", 0.5) # Minimum allocation multiplier for volatility sizing
        self.MAX_DRAWDOWN=self.parse_float_env("MAX_DRAWDOWN", 0.1) # Maximum drawdown allowed before stopping trading

        if self.MAX_ACTIVE_ASSETS < 1:
            logger.warning(
                f"MAX_ACTIVE_ASSETS is {self.MAX_ACTIVE_ASSETS}, must be >= 1. Setting to 7."
            )
            self.MAX_ACTIVE_ASSETS = 7
            self.ASSET_THRESHOLD = int(self.MAX_ACTIVE_ASSETS * 0.6)

        if not self.TELEGRAM_BOT_TOKEN or not self.TELEGRAM_CHAT_ID:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing. Telegram notifications disabled."
            )

        # logger.info("Configuration reloaded from .env file")

# Instantiate the singleton config object
config = Config()

# Replace reload_config function to use the singleton
def reload_config():
    config.reload()