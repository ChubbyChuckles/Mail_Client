# config.py
import io
import logging
import os
import sys
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
    if hasattr(sys.stdout, "reconfigure"):
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
            logger.error(
                f"Invalid value for {var_name}: {value}. Using default: {default}"
            )
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
        self.PARQUET_FILENAME = os.getenv(
            "PARQUET_FILENAME", "bitvavo_1min_candles_eur.parquet"
        )
        self.PRICE_INCREASE_THRESHOLD = float(
            os.getenv("PRICE_INCREASE_THRESHOLD", 1.0)
        )
        self.MIN_VOLUME_EUR = float(os.getenv("MIN_VOLUME_EUR", 10000))
        self.PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_VALUE", 10000))
        self.ALLOCATION_PER_TRADE = float(os.getenv("ALLOCATION_PER_TRADE", 0.1))
        self.BUY_FEE = float(os.getenv("BUY_FEE", 0.0015))
        self.SELL_FEE = float(os.getenv("SELL_FEE", 0.0025))
        self.TRAILING_STOP_FACTOR = float(os.getenv("TRAILING_STOP_FACTOR", 1.0))
        self.TRAILING_STOP_FACTOR_EARLY = float(
            os.getenv("TRAILING_STOP_FACTOR_EARLY", 1.5)
        )
        self.ADJUSTED_PROFIT_TARGET = float(os.getenv("ADJUSTED_PROFIT_TARGET", 0.015))
        self.PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", 0.05))
        self.MIN_HOLDING_MINUTES = float(os.getenv("MIN_HOLDING_MINUTES", 5))
        self.TIME_STOP_MINUTES = int(os.getenv("TIME_STOP_MINUTES", 90))
        self.CAT_LOSS_THRESHOLD = float(os.getenv("CAT_LOSS_THRESHOLD", -0.08))
        self.MOMENTUM_CONFIRM_MINUTES = int(os.getenv("MOMENTUM_CONFIRM_MINUTES", 3))
        self.MOMENTUM_THRESHOLD = float(os.getenv("MOMENTUM_THRESHOLD", -0.25))
        self.PORTFOLIO_FILE = os.getenv("PORTFOLIO_FILE", "portfolio.json")
        self.LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", 60))
        self.MAX_ACTIVE_ASSETS = int(os.getenv("MAX_ACTIVE_ASSETS", 7))
        self.ASSET_THRESHOLD = int(self.MAX_ACTIVE_ASSETS * 0.6)
        self.INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 20))
        self.PROFIT_TARGET_MULTIPLIER = float(
            os.getenv("PROFIT_TARGET_MULTIPLIER", 2.0)
        )
        self.BUY_TRADES_CSV = os.getenv("BUY_TRADES_CSV", "buy_trades.csv")
        self.FINISHED_TRADES_CSV = os.getenv(
            "FINISHED_TRADES_CSV", "finished_trades.csv"
        )
        self.ORDER_BOOK_METRICS_CSV = os.getenv(
            "ORDER_BOOK_METRICS_CSV", "order_book_metrics.csv"
        )
        self.AMOUNT_QUOTE = self.parse_float_env("AMOUNT_QUOTE", 5.5)
        self.PRICE_RANGE_PERCENT = self.parse_float_env("PRICE_RANGE_PERCENT", 10.0)
        self.MAX_SLIPPAGE_BUY = self.parse_float_env("MAX_SLIPPAGE_BUY", 0.05)
        self.MAX_SLIPPAGE_SELL = self.parse_float_env("MAX_SLIPPAGE_SELL", -0.05)
        self.MIN_TOTAL_SCORE = self.parse_float_env("MIN_TOTAL_SCORE", 0.7)
        self.USE_RSI = bool(os.getenv("USE_RSI", True))
        self.RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
        self.RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", 70))
        self.RSI_MIN_SCORE = int(os.getenv("RSI_MIN_SCORE", 30))
        self.USE_BOLLINGER_BANDS = bool(
            os.getenv("USE_BOLLINGER_BANDS", True)
        )  # Enable/disable Bollinger Bands for buy decisions
        self.BOLLINGER_PERIOD = int(
            os.getenv("BOLLINGER_PERIOD", 20)
        )  # Period for Bollinger Bands calculation
        self.BOLLINGER_STD_DEV = self.parse_float_env(
            "BOLLINGER_STD_DEV", 2.0
        )  # Standard deviation multiplier for Bollinger Bands

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
