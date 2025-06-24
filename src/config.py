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

# Create trading_logs directory if it doesn't exist
log_dir = "trading_logs"
os.makedirs(log_dir, exist_ok=True)

# Create a new log file for this session in trading_logs folder
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


# Custom handler for evaluation logs
class EvaluationLogHandler(logging.Handler):
    def emit(self, record):
        if record.levelno == logging.INFO and "Evaluation Decision" in record.msg:
            with open(log_filename, "a", encoding="utf-8") as f:
                f.write(f"{record.asctime} [INFO] {record.msg}\n")


evaluation_handler = EvaluationLogHandler()
evaluation_handler.setFormatter(log_formatter)
logger.addHandler(evaluation_handler)

# Ensure unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

# Load environment variables
load_dotenv()

# Configuration variables from .env
API_KEY = os.getenv("BITVAVO_API_KEY")
API_SECRET = os.getenv("BITVAVO_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", 30))
RATE_LIMIT_WEIGHT = int(os.getenv("RATE_LIMIT_WEIGHT", 1000))
CANDLE_LIMIT = int(os.getenv("CANDLE_LIMIT", 10))
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "1m")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "data_1m_pq_alot")
PARQUET_FILENAME = os.getenv("PARQUET_FILENAME", "bitvavo_1min_candles_eur.parquet")
PRICE_INCREASE_THRESHOLD = float(os.getenv("PRICE_INCREASE_THRESHOLD", 1.0))
MIN_VOLUME_EUR = float(os.getenv("MIN_VOLUME_EUR", 10000))
GOOGLE_SHEETS_CREDENTIALS = os.getenv("GOOGLE_SHEETS_CREDENTIALS", "creds.json")
SPREADSHEET_NAME = os.getenv("SPREADSHEET_NAME", "Bitvavo_Trading_Results")
ACTIVE_ASSETS_SHEET = os.getenv("ACTIVE_ASSETS_SHEET", "Active_Assets")
TOP_COINS_SHEET = os.getenv("TOP_COINS_SHEET", "Top_Coins")
FINISHED_TRADES_SHEET = os.getenv("FINISHED_TRADES_SHEET", "Finished_Trades")
PORTFOLIO_VALUE = float(os.getenv("PORTFOLIO_VALUE", 10000))
ALLOCATION_PER_TRADE = float(os.getenv("ALLOCATION_PER_TRADE", 0.1))
BUY_FEE = float(os.getenv("BUY_FEE", 0.0015))
SELL_FEE = float(os.getenv("SELL_FEE", 0.0025))
TRAILING_STOP_FACTOR = float(os.getenv("TRAILING_STOP_FACTOR", 1.0))
TRAILING_STOP_FACTOR_EARLY = float(os.getenv("TRAILING_STOP_FACTOR_EARLY", 1.5))
ADJUSTED_PROFIT_TARGET = float(os.getenv("ADJUSTED_PROFIT_TARGET", 0.015))
PROFIT_TARGET = float(os.getenv("PROFIT_TARGET", 0.05))
MIN_HOLDING_MINUTES = float(os.getenv("MIN_HOLDING_MINUTES", 5))
TIME_STOP_MINUTES = int(os.getenv("TIME_STOP_MINUTES", 30))
CAT_LOSS_THRESHOLD = float(os.getenv("CAT_LOSS_THRESHOLD", -0.08))
MOMENTUM_CONFIRM_MINUTES = int(os.getenv("MOMENTUM_CONFIRM_MINUTES", 3))
MOMENTUM_THRESHOLD = float(os.getenv("MOMENTUM_THRESHOLD", -0.25))
PORTFOLIO_FILE = os.getenv("PORTFOLIO_FILE", "portfolio.json")
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", 70))
MAX_ACTIVE_ASSETS = int(os.getenv("MAX_ACTIVE_ASSETS", 20))
ASSET_THRESHOLD = int(MAX_ACTIVE_ASSETS * 0.6)
INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", 20))
SHEETS_WRITE_INTERVAL = int(os.getenv("SHEETS_WRITE_INTERVAL", 300))
PROFIT_TARGET_MULTIPLIER = float(os.getenv("PROFIT_TARGET_MULTIPLIER", 2.0))
BUY_TRADES_CSV = os.getenv("BUY_TRADES_CSV", "buy_trades.csv")

# Validate MAX_ACTIVE_ASSETS
if MAX_ACTIVE_ASSETS < 1:
    logger.warning(
        f"MAX_ACTIVE_ASSETS is {MAX_ACTIVE_ASSETS}, must be >= 1. Setting to 20."
    )
    MAX_ACTIVE_ASSETS = 20
    ASSET_THRESHOLD = int(MAX_ACTIVE_ASSETS * 0.6)

# Warn if Telegram credentials are missing
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    logger.warning(
        "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing. Telegram notifications disabled."
    )
