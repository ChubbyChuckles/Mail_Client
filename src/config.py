# config.py
import io
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Determine runtime environment
IS_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"

# Set session start time for log file naming
SESSION_START_TIME = datetime.utcnow()

# Set up logging
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

# Adjust log directory and behavior based on environment
if IS_GITHUB_ACTIONS:
    # In GitHub Actions, avoid writing logs to disk to save space
    log_dir = "/tmp/trading_logs"  # Use temporary directory
    log_filename = None  # Disable file logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
else:
    # Local PC: Write logs to disk
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
        if IS_GITHUB_ACTIONS:
            # Skip file writes in GitHub Actions
            return
        if record.levelno == logging.INFO and "Evaluation Decision" in record.msg:
            with open(log_filename, "a", encoding="utf-8") as f:
                formatted_message = self.format(record)
                f.write(f"{formatted_message}\n")


if not IS_GITHUB_ACTIONS:
    evaluation_handler = EvaluationLogHandler()
    evaluation_handler.setFormatter(log_formatter)
    logger.addHandler(evaluation_handler)

# Force line buffering for stdout
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
        # Adjust concurrency for GitHub Actions to avoid resource limits
        if IS_GITHUB_ACTIONS:
            self.CONCURRENT_REQUESTS = min(
                self.CONCURRENT_REQUESTS, 10
            )  # Lower concurrency
        self.RATE_LIMIT_WEIGHT = int(os.getenv("RATE_LIMIT_WEIGHT", 1000))
        self.CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", 50))
        self.CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "1m")
        # Adjust paths for GitHub Actions
        self.RESULTS_FOLDER = os.getenv(
            "RESULTS_FOLDER",
            "/tmp/data_1m_pq_alot" if IS_GITHUB_ACTIONS else "data_1m_pq_alot",
        )
        os.makedirs(self.RESULTS_FOLDER, exist_ok=True)
        self.PARQUET_FILENAME = os.getenv(
            "PARQUET_FILENAME", "bitvavo_1min_candles_eur.parquet"
        )
        self.PRICE_INCREASE_THRESHOLD = float(
            os.getenv("PRICE_INCREASE_THRESHOLD", 1.00)
        )
        self.MIN_VOLUME_EUR = float(os.getenv("MIN_VOLUME_EUR", 5000))
        self.PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_VALUE", 10000))
        self.ALLOCATION_PER_TRADE = float(os.getenv("ALLOCATION_PER_TRADE", 0.1))
        self.BUY_FEE = float(os.getenv("BUY_FEE", 0.0015))
        self.SELL_FEE = float(os.getenv("SELL_FEE", 0.0025))
        self.TRAILING_STOP_FACTOR = float(os.getenv("TRAILING_STOP_FACTOR", 2.0))
        self.TRAILING_STOP_FACTOR_EARLY = float(
            os.getenv("TRAILING_STOP_FACTOR_EARLY", 3)
        )
        self.ADJUSTED_PROFIT_TARGET = float(os.getenv("ADJUSTED_PROFIT_TARGET", 0.025))
        self.PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", 0.05))
        self.MIN_HOLDING_MINUTES = float(os.getenv("MIN_HOLDING_MINUTES", 6))
        self.TIME_STOP_MINUTES = int(os.getenv("TIME_STOP_MINUTES", 180))
        self.CAT_LOSS_THRESHOLD = float(os.getenv("CAT_LOSS_THRESHOLD", 0.08))
        self.MOMENTUM_CONFIRM_MINUTES = int(os.getenv("MOMENTUM_CONFIRM_MINUTES", 3))
        self.MOMENTUM_THRESHOLD = float(os.getenv("MOMENTUM_THRESHOLD", -0.25))
        # Adjust portfolio file path for GitHub Actions
        self.PORTFOLIO_FILE = os.getenv(
            "PORTFOLIO_FILE",
            "/tmp/portfolio.json" if IS_GITHUB_ACTIONS else "portfolio.json",
        )
        self.LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", 60))
        self.MAX_ACTIVE_ASSETS = int(os.getenv("MAX_ACTIVE_ASSETS", 8))
        self.ASSET_THRESHOLD = int(self.MAX_ACTIVE_ASSETS * 0.8)
        self.INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 20))
        self.PROFIT_TARGET_MULTIPLIER = float(
            os.getenv("PROFIT_TARGET_MULTIPLIER", 4.0)
        )
        # Adjust CSV file paths for GitHub Actions
        self.BUY_TRADES_CSV = os.getenv(
            "BUY_TRADES_CSV",
            "/tmp/buy_trades.csv" if IS_GITHUB_ACTIONS else "buy_trades.csv",
        )
        self.FINISHED_TRADES_CSV = os.getenv(
            "FINISHED_TRADES_CSV",
            "/tmp/finished_trades.csv" if IS_GITHUB_ACTIONS else "finished_trades.csv",
        )
        self.ORDER_BOOK_METRICS_CSV = os.getenv(
            "ORDER_BOOK_METRICS_CSV",
            (
                "/tmp/order_book_metrics.csv"
                if IS_GITHUB_ACTIONS
                else "order_book_metrics.csv"
            ),
        )
        self.AMOUNT_QUOTE = self.parse_float_env("AMOUNT_QUOTE", 1000.0)
        self.PRICE_RANGE_PERCENT = self.parse_float_env("PRICE_RANGE_PERCENT", 10.0)
        self.MAX_SLIPPAGE_BUY = self.parse_float_env("MAX_SLIPPAGE_BUY", 0.025)
        self.MAX_SLIPPAGE_SELL = self.parse_float_env("MAX_SLIPPAGE_SELL", -0.05)
        self.MIN_TOTAL_SCORE = self.parse_float_env("MIN_TOTAL_SCORE", 0.75)
        self.USE_RSI = bool(os.getenv("USE_RSI", True))
        self.RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
        self.RSI_OVERBOUGHT = int(os.getenv("RSI_OVERBOUGHT", 70))
        self.RSI_MIN_SCORE = int(os.getenv("RSI_MIN_SCORE", 30))
        self.USE_BOLLINGER_BANDS = bool(os.getenv("USE_BOLLINGER_BANDS", True))
        self.BOLLINGER_PERIOD = int(os.getenv("BOLLINGER_PERIOD", 20))
        self.BOLLINGER_STD_DEV = self.parse_float_env("BOLLINGER_STD_DEV", 2.0)

        self.MIN_BULLISH_INDICATOR = self.parse_float_env("MIN_BULLISH_INDICATOR", 0.7) # Default 0.6

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

        logger.info(
            f"Configuration reloaded from .env file. Running in {'GitHub Actions' if IS_GITHUB_ACTIONS else 'Local PC'} mode."
        )


# Instantiate the singleton config object
config = Config()


def reload_config():
    config.reload()
